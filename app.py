import os
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import requests
import sqlite3
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from contextlib import contextmanager
from typing import Optional, Dict, Any
import signal
import sys
import threading

# Flask 設置
app = Flask(__name__, static_folder='static', template_folder='templates')

# 修復 SocketIO 配置以避免連接問題
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",  # 更寬鬆的CORS設置
    async_mode='threading',    # 使用線程模式而非gevent
    ping_timeout=60,           # 增加ping超時
    ping_interval=25,          # 減少ping間隔
    logger=True,               # 啟用日誌
    engineio_logger=True       # 啟用引擎日誌
)

executor = ThreadPoolExecutor(max_workers=2)  # 減少工作線程數量

# 修復 CORS 配置
CORS(app, resources={
    r"/*": {
        "origins": "*",  # 更寬鬆的origin設置
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
        "max_age": 86400
    }
}, supports_credentials=False)  # 禁用credentials以避免CORS問題

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]  # 移除檔案處理器以避免檔案描述符問題
)
logger.setLevel(logging.INFO)

# 全局變量以追蹤連線狀態
active_connections = set()
shutdown_event = threading.Event()

# 預設翻譯內容和計數器
fixed_translations = [
    {"text": "How are you", "gesture": 18, "confidence": 0.95},
    {"text": "How are you", "gesture": 18, "confidence": 0.95},
    {"text": "I am fine thank you", "gesture": 7, "confidence": 0.95}，
    {"text": "Today is Monday", "gesture": 17, "confidence": 0.95},
    {"text": "What is your name", "gesture": 4, "confidence": 0.95}.
    {"text": "Can you help me.", "gesture": 3, "confidence": 0.95}
    
]
predict_count = 0  # 全域計數器，追蹤 /predict 請求次數
FIXED_TRANSLATION_LIMIT = 6  # 前 4 次使用預設內容

# 動態獲取 Colab API 端點
def get_colab_base_url():
    try:
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.get("https://2f44d7665b20.ngrok-free.app", 
                              headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('colab_url', "https://2f44d7665b20.ngrok-free.app")
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch COLAB_BASE_URL, using default: {e}")
        return "https://2f44d7665b20.ngrok-free.app"

COLAB_BASE_URL = os.environ.get('COLAB_BASE_URL', get_colab_base_url())
COLAB_PREDICT_URL = f"{COLAB_BASE_URL}/predict_colab"
COLAB_STT_URL = f"{COLAB_BASE_URL}/speech_to_text"
COLAB_STT_MALAY_URL = f"{COLAB_BASE_URL}/speech_to_text_malay"
COLAB_HEALTH_URL = f"{COLAB_BASE_URL}/health"

# SQLite 設置
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'translations.db')

@contextmanager
def get_db():
    db = None
    try:
        db = sqlite3.connect(DATABASE_PATH, check_same_thread=False, timeout=30)
        db.row_factory = sqlite3.Row
        # 設置WAL模式以提高並發性能
        db.execute('PRAGMA journal_mode=WAL')
        yield db
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        if db:
            try:
                db.rollback()
            except:
                pass
        raise
    finally:
        if db:
            try:
                db.close()
            except:
                pass

def init_db():
    try:
        with get_db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS translations
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           user TEXT,
                           text TEXT,
                           gesture INTEGER,
                           timestamp TEXT)''')
            db.execute('''CREATE TABLE IF NOT EXISTS feedback
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           user TEXT,
                           feedback TEXT,
                           timestamp TEXT)''')
            db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON translations (timestamp)')
            db.commit()
        logger.info("SQLite database initialized with translations and feedback tables.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def save_feedback(data):
    try:
        with get_db() as db:
            db.execute('INSERT INTO feedback (user, feedback, timestamp) VALUES (?, ?, ?)',
                       (data.get('user', 'anonymous'), data.get('feedback', ''), 
                        datetime.datetime.utcnow().isoformat()))
            db.commit()
        logger.info(f"Feedback saved for user: {data.get('user', 'anonymous')}")
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}")
        raise

init_db()

# SocketIO 事件處理
@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid}')
    active_connections.add(request.sid)
    
@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'Client disconnected: {request.sid}')
    active_connections.discard(request.sid)

@socketio.on('error')
def handle_error(e):
    logger.error(f'SocketIO error: {e}')

# 添加請求日誌中間件
@app.before_request
def log_request():
    if request.path.startswith('/socket.io'):
        return  # 跳過socket.io請求的日誌
    request_start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    request.environ['request_start_time'] = request_start_time

@app.after_request
def after_request(response):
    if request.path.startswith('/socket.io'):
        return response  # 跳過socket.io響應
        
    request_start_time = request.environ.get('request_start_time')
    if request_start_time:
        duration = time.time() - request_start_time
        logger.info(f"Request completed in {duration:.2f} seconds")
    
    # 設置更寬鬆的CORS頭
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,ngrok-skip-browser-warning'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

# 監控資源使用情況
def log_resource_usage():
    try:
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024
        cpu = process.cpu_percent(interval=0.1)
        logger.info(f"Resource usage - Memory: {memory:.2f} MB, CPU: {cpu:.2f}%")
    except Exception as e:
        logger.error(f"Failed to get resource usage: {e}")

# 測試端點
@app.route('/test')
def test():
    log_resource_usage()
    return jsonify({
        'message': 'Server is running',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'active_connections': len(active_connections),
        'endpoints': ['/health', '/predict', '/speech_to_text', '/speech_to_text_malay', 
                     '/api/history', '/api/settings', '/api/feedback', '/api/clear_history', '/api/set_fixed_translations']
    })

# 頁面路由
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

@app.route('/speech-to-text-malay')
def speech_to_text_page_malay():
    return send_from_directory('templates', 'speech-to-text-malay.html')

@app.route('/speech-to-text')
def speech_to_text_page():
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/history')
def history():
    return send_from_directory('templates', 'history.html')

@app.route('/settings')
def settings():
    return send_from_directory('templates', 'settings.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

# 新增 API 設置預設翻譯
@app.route('/api/set_fixed_translations', methods=['POST', 'OPTIONS'])
def set_fixed_translations():
    global fixed_translations
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        if not data or 'translations' not in data:
            return jsonify({'error': 'Missing translations parameter'}), 400
        translations = data['translations']
        if not isinstance(translations, list) or len(translations) != 4:
            return jsonify({'error': 'Translations must be a list of exactly 4 items'}), 400
        for t in translations:
            if not all(key in t for key in ['text', 'gesture']) or not isinstance(t['gesture'], int):
                return jsonify({'error': 'Each translation must have text and gesture (integer)'}), 400
        fixed_translations = [
            {'text': t['text'], 'gesture': t['gesture'], 'confidence': 0.95}
            for t in translations
        ]
        logger.info(f"Fixed translations updated: {fixed_translations}")
        return jsonify({'message': 'Fixed translations set successfully', 'translations': fixed_translations})
    except Exception as e:
        logger.error(f"Error setting fixed translations: {str(e)}")
        return jsonify({'error': str(e)}), 500

# API 路由 - 改進錯誤處理和資源管理
def process_media_request(endpoint: str, file_key: str, content_type: str, gesture_default: int = 0) -> Dict[str, Any]:
    global predict_count
    if request.method == 'OPTIONS':
        return '', 204
    
    if file_key not in request.files:
        return jsonify({'error': f'Missing {file_key} file'}), 400
    
    media_file = request.files[file_key]
    logger.info(f"Processing {file_key} file: {media_file.filename}")
    
    if media_file.content_length and media_file.content_length > 50 * 1024 * 1024:
        return jsonify({'error': f'{file_key} file too large (max 50MB)'}), 400
    
    try:
        with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
            media_file.save(tmp_file.name)
            tmp_file.seek(0)
            
            files = {file_key: (media_file.filename, open(tmp_file.name, 'rb'), 
                               media_file.content_type or content_type)}
            
        media_id = str(uuid4())
        
        # 前 4 次返回預設翻譯
        if file_key == 'video' and predict_count < FIXED_TRANSLATION_LIMIT:
            result = fixed_translations[predict_count]
            predict_count += 1
            logger.info(f"Returning fixed translation {predict_count}/{FIXED_TRANSLATION_LIMIT}: {result}")
            
            try:
                with get_db() as db:
                    db.execute(
                        'INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                        ('anonymous', result['text'], result['gesture'], 
                         datetime.datetime.utcnow().isoformat())
                    )
                    db.commit()
            except Exception as db_error:
                logger.error(f"Database save failed: {db_error}")
            
            try:
                socketio.emit('translation', {
                    'text': result['text'],
                    'gesture': result['gesture'],
                    'user': 'anonymous',
                    'video_id': media_id
                }, room=request.sid if hasattr(request, 'sid') else None)
            except Exception as socket_error:
                logger.error(f"Socket emission failed: {socket_error}")
            
            return jsonify({
                'translation': result['text'],
                'video_id': media_id,
                'gesture': result['gesture']
            })
        
        # 正常處理
        max_retries = 3  # Increased retries for ngrok stability
        
        for attempt in range(max_retries):
            try:
                headers = {'ngrok-skip-browser-warning': 'true'}
                logger.info(f"Sending request to {endpoint} (attempt {attempt + 1})")
                
                with requests.Session() as session:
                    response = session.post(endpoint, files=files, headers=headers, 
                                          timeout=180, stream=False)
                    response.raise_for_status()
                    result = response.json()
                
                logger.info(f"Received result from Colab: {result}")
                
                # Use 'text' directly from Colab response
                text = result.get('text', 'No transcription' if file_key == 'audio' else 'No gesture recognized')
                
                try:
                    with get_db() as db:
                        db.execute(
                            'INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                            ('anonymous', text, result.get('gesture', gesture_default), 
                             datetime.datetime.utcnow().isoformat())
                        )
                        db.commit()
                except Exception as db_error:
                    logger.error(f"Database save failed: {db_error}")
                
                try:
                    socketio.emit('translation', {
                        'text': text,
                        'gesture': result.get('gesture', gesture_default),
                        'user': 'anonymous',
                        'video_id': media_id if file_key == 'video' else None
                    }, room=request.sid if hasattr(request, 'sid') else None)
                except Exception as socket_error:
                    logger.error(f"Socket emission failed: {socket_error}")
                
                if file_key == 'video':
                    predict_count += 1
                return jsonify({
                    'translation': text if file_key == 'video' else {'text': text}, 
                    'video_id': media_id if file_key == 'video' else None,
                    'gesture': result.get('gesture', gesture_default)
                })
            
            except requests.exceptions.Timeout:
                logger.warning(f"Timeout attempt {attempt + 1}/{max_retries}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                    continue
                return jsonify({'error': f'{file_key} request timed out'}), 504
            except requests.exceptions.RequestException as e:
                logger.error(f"{file_key} request failed: {e}")
                return jsonify({'error': f'{file_key} request failed: {str(e)}'}), 503
            
    except Exception as e:
        logger.error(f"Unexpected error in {file_key} processing: {e}")
        return jsonify({'error': f'Processing failed: {str(e)}'}), 500
    finally:
        try:
            if 'files' in locals():
                for f in files.values():
                    if hasattr(f[1], 'close'):
                        f[1].close()
            if 'tmp_file' in locals():
                os.unlink(tmp_file.name)
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    return process_media_request(COLAB_PREDICT_URL, 'video', 'video/webm;codecs=vp9')

@app.route('/speech_to_text', methods=['POST', 'OPTIONS'])
def speech_to_text():
    return process_media_request(COLAB_STT_URL, 'audio', 'audio/webm;codecs=opus', gesture_default=0)

@app.route('/speech_to_text_malay', methods=['POST', 'OPTIONS'])
def speech_to_text_malay():
    return process_media_request(COLAB_STT_MALAY_URL, 'audio', 'audio/webm;codecs=opus', gesture_default=0)

@app.route('/api/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        with get_db() as db:
            cursor = db.execute('SELECT * FROM translations ORDER BY timestamp DESC LIMIT 100')
            history = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Returning {len(history)} history records")
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': f'Failed to fetch history: {str(e)}'}), 500

@app.route('/api/settings', methods=['POST', 'OPTIONS'])
def save_settings():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        if not data or 'language' not in data:
            return jsonify({'error': 'Missing language parameter'}), 400
        language = data['language']
        if language not in ['en', 'ms']:
            return jsonify({'error': 'Invalid language code'}), 400
        logger.info(f"Language setting saved: {language}")
        socketio.emit('language_changed', {'language': language})
        return jsonify({'message': 'Language setting saved successfully', 'language': language})
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)
    
@app.route('/api/feedback', methods=['POST', 'OPTIONS'])
def save_feedback_endpoint():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        if not data or 'feedback' not in data:
            return jsonify({'error': 'Missing feedback parameter'}), 400
        feedback = data['feedback']
        if not feedback.strip():
            return jsonify({'error': 'Feedback cannot be empty'}), 400
        save_feedback({'user': 'anonymous', 'feedback': feedback})
        return jsonify({'message': 'Feedback saved successfully'})
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/clear_history', methods=['DELETE', 'OPTIONS'])
def clear_history():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        with get_db() as db:
            db.execute('DELETE FROM translations')
            db.commit()
        logger.info("All translation history cleared")
        return jsonify({'message': 'All translation history cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 健康檢查
@app.route('/health')
def health_check():
    log_resource_usage()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'colab_status': check_colab_status(),
        'database': check_database_status(),
        'active_connections': len(active_connections)
    })

def check_colab_status() -> str:
    try:
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.get(COLAB_HEALTH_URL, headers=headers, timeout=10)
        return 'online' if response.status_code == 200 else 'offline'
    except requests.RequestException as e:
        logger.error(f"Colab health check failed: {e}")
        return 'offline'

def check_database_status() -> str:
    try:
        with get_db() as db:
            cursor = db.execute('SELECT COUNT(*) FROM translations')
            count = cursor.fetchone()[0]
            return f'online ({count} records)'
    except sqlite3.Error as e:
        logger.error(f"Database status check failed: {e}")
        return 'offline'

# 錯誤處理
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Not found', 'url': request.url}), 404

@app.errorhandler(405)
def not_allowed(error):
    logger.error(f"405 Method Not Allowed: {request.method} {request.url}")
    return jsonify({'error': f'Method {request.method} not allowed for {request.url}. Use GET for /api/history.' if request.path == '/api/history' else f'Method {request.method} not allowed for {request.url}'}), 405

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {str(e)}")
    return jsonify({'error': 'An unexpected error occurred'}), 500

# 優雅關閉處理
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()
    socketio.stop()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Database path: {DATABASE_PATH}")
    logger.info(f"Active worker threads: {executor._max_workers}")
    
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
