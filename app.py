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

# 修復 SocketIO 配置 - 使用 eventlet 作為異步模式
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='eventlet',      # 改為 eventlet 模式
    ping_timeout=120,           # 增加 ping 超時
    ping_interval=25,
    logger=False,               # 禁用詳細日志以減少輸出
    engineio_logger=False,      # 禁用引擎日志
    transports=['websocket', 'polling']  # 允許回退到 polling
)

# 減少線程池大小
executor = ThreadPoolExecutor(max_workers=2)

# 修復 CORS 配置
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
        "max_age": 86400
    }
}, supports_credentials=False)

# 設置日誌 - 減少日志級別
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.WARNING,  # 改為 WARNING 級別
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger.setLevel(logging.WARNING)

# 全局變量
active_connections = set()
shutdown_event = threading.Event()

# 動態獲取 Colab API 端點 - 添加緩存
_colab_base_url_cache = None
_colab_cache_timestamp = 0
CACHE_DURATION = 300  # 5分鐘緩存

def get_colab_base_url():
    global _colab_base_url_cache, _colab_cache_timestamp
    
    current_time = time.time()
    if _colab_base_url_cache and (current_time - _colab_cache_timestamp) < CACHE_DURATION:
        return _colab_base_url_cache
    
    try:
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.get("https://8603c16d2d79.ngrok-free.app/health", 
                              headers=headers, timeout=5)  # 減少超時時間
        response.raise_for_status()
        data = response.json()
        _colab_base_url_cache = data.get('colab_url', "https://8603c16d2d79.ngrok-free.app")
        _colab_cache_timestamp = current_time
        return _colab_base_url_cache
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch COLAB_BASE_URL, using default: {e}")
        default_url = "https://8603c16d2d79.ngrok-free.app"
        _colab_base_url_cache = default_url
        _colab_cache_timestamp = current_time
        return default_url

COLAB_BASE_URL = os.environ.get('COLAB_BASE_URL', get_colab_base_url())
COLAB_PREDICT_URL = f"{COLAB_BASE_URL}/predict_colab"
COLAB_STT_URL = f"{COLAB_BASE_URL}/speech_to_text"
COLAB_STT_MALAY_URL = f"{COLAB_BASE_URL}/speech_to_text_malay"
COLAB_HEALTH_URL = f"{COLAB_BASE_URL}/health"

GESTURE_MAPPING = {
    1: "Hi, How are you?",
    2: "I am fine, thank you.",
    3: "What is your name",
    4: "Excuse me, what is the time now?",
    5: "I am hungry and want to eat.",
    6: "Can you help me.",
    7: "I need help.",
    8: "Have a nice day.",
    9: "Thank You for your help.",
    10: "How much is this?",
    11: "See you tomorrow",
    12: "I am going to buy something.",
    13: "Do you want to play together.",
    14: "Where are you going?",
    15: "Where is toilet",
    16: "Toilet is turn right in front",
    17: "What day is it today?",
    18: "Today is Monday.",
    19: "Yes of course.",
    20: "what are you doing now?",
    21: "I am working",
    22: "Are you free tomorrow afternoon.",
    23: "Sorry i am busy tomorrow",
    24: "I have a little headache",
    25: "What is your name?",
    26: "I want a glass of water",
    27: "This is too expensive.",
    28: "Can i sit here?",
    29: "I need to rest."
}

# SQLite 設置 - 優化數據庫配置
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'translations.db')

@contextmanager
def get_db():
    db = None
    try:
        db = sqlite3.connect(DATABASE_PATH, check_same_thread=False, timeout=10)  # 減少超時
        db.row_factory = sqlite3.Row
        # 設置性能優化
        db.execute('PRAGMA journal_mode=WAL')
        db.execute('PRAGMA synchronous=NORMAL')
        db.execute('PRAGMA cache_size=1000')
        db.execute('PRAGMA temp_store=MEMORY')
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
            db.execute('CREATE INDEX IF NOT EXISTS idx_user ON translations (user)')
            db.commit()
        logger.warning("SQLite database initialized with translations and feedback tables.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def save_feedback(data):
    try:
        with get_db() as db:
            db.execute('INSERT INTO feedback (user, feedback, timestamp) VALUES (?, ?, ?)',
                       (data.get('user', 'anonymous'), data.get('feedback', ''), 
                        datetime.datetime.utcnow().isoformat()))
            db.commit()
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}")
        raise

init_db()

# SocketIO 事件處理 - 添加錯誤處理
@socketio.on('connect')
def handle_connect():
    try:
        active_connections.add(request.sid)
        logger.warning(f'Client connected: {request.sid}')
    except Exception as e:
        logger.error(f'Connect error: {e}')
    
@socketio.on('disconnect')
def handle_disconnect():
    try:
        active_connections.discard(request.sid)
        logger.warning(f'Client disconnected: {request.sid}')
    except Exception as e:
        logger.error(f'Disconnect error: {e}')

@socketio.on('error')
def handle_error(e):
    logger.error(f'SocketIO error: {e}')

@socketio.on_error_default
def default_error_handler(e):
    logger.error(f'SocketIO default error: {e}')

# 移除請求日誌中間件以減少開銷
@app.after_request
def after_request(response):
    if not request.path.startswith('/socket.io'):
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,ngrok-skip-browser-warning'
        response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
        response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

# 監控資源使用情況 - 簡化版本
def log_resource_usage():
    try:
        process = psutil.Process()
        memory = process.memory_info().rss / 1024 / 1024
        if memory > 200:  # 只在內存超過200MB時記錄
            logger.warning(f"High memory usage: {memory:.2f} MB")
    except Exception:
        pass

# 測試端點
@app.route('/test')
def test():
    return jsonify({
        'message': 'Server is running',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'active_connections': len(active_connections),
        'status': 'healthy'
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

# API 路由 - 優化版本
def process_media_request(endpoint: str, file_key: str, content_type: str, gesture_default: int = 0) -> Dict[str, Any]:
    if request.method == 'OPTIONS':
        return '', 204
    
    if file_key not in request.files:
        return jsonify({'error': f'Missing {file_key} file'}), 400
    
    media_file = request.files[file_key]
    
    # 檢查文件大小限制（30MB - 減少限制）
    if media_file.content_length and media_file.content_length > 30 * 1024 * 1024:
        return jsonify({'error': f'{file_key} file too large (max 30MB)'}), 400
    
    temp_path = None
    try:
        # 創建臨時文件
        with tempfile.NamedTemporaryFile(delete=False, suffix=f'.{file_key}') as tmp_file:
            temp_path = tmp_file.name
            media_file.save(tmp_file.name)
        
        # 準備文件上傳
        with open(temp_path, 'rb') as f:
            files = {file_key: (media_file.filename, f, 
                               media_file.content_type or content_type)}
            
            media_id = str(uuid4())
            headers = {'ngrok-skip-browser-warning': 'true'}
            
            # 單次請求，減少重試
            try:
                response = requests.post(endpoint, files=files, headers=headers, 
                                      timeout=90, stream=False)  # 減少超時時間
                response.raise_for_status()
                result = response.json()
                
                text = result.get('text', 'No transcription' if file_key == 'audio' else 
                                 GESTURE_MAPPING.get(result.get('gesture', gesture_default), 'Unknown gesture'))
                if file_key == 'video' and result.get('predictions'):
                    text = GESTURE_MAPPING.get(result.get('predictions', [0])[0], text)
                
                # 異步保存到數據庫
                def save_to_db():
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
                
                # 在線程池中執行數據庫保存
                executor.submit(save_to_db)
                
                # 發送Socket.IO事件
                try:
                    socketio.emit('translation', {
                        'text': text,
                        'gesture': result.get('gesture', gesture_default),
                        'user': 'anonymous',
                        'video_id': media_id if file_key == 'video' else None
                    })
                except Exception as socket_error:
                    logger.error(f"Socket emission failed: {socket_error}")
                
                return jsonify({
                    'translation': text if file_key == 'video' else {'text': text}, 
                    'video_id': media_id if file_key == 'video' else None,
                    'gesture': result.get('gesture', gesture_default)
                })
                
            except requests.exceptions.Timeout:
                return jsonify({'error': f'{file_key} request timed out'}), 504
            except requests.exceptions.RequestException as e:
                logger.error(f"{file_key} request failed: {e}")
                return jsonify({'error': f'{file_key} service unavailable'}), 503
            
    except Exception as e:
        logger.error(f"Unexpected error in {file_key} processing: {e}")
        return jsonify({'error': f'Processing failed'}), 500
    finally:
        # 清理臨時文件
        if temp_path and os.path.exists(temp_path):
            try:
                os.unlink(temp_path)
            except:
                pass

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
            cursor = db.execute('SELECT * FROM translations ORDER BY timestamp DESC LIMIT 50')  # 減少限制
            history = [dict(row) for row in cursor.fetchall()]
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {e}")
        return jsonify({'error': 'Failed to fetch history'}), 500

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
        
        socketio.emit('language_changed', {'language': language})
        return jsonify({'message': 'Language setting saved successfully', 'language': language})
    except Exception as e:
        logger.error(f"Error saving settings: {str(e)}")
        return jsonify({'error': 'Settings save failed'}), 500

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
        return jsonify({'error': 'Feedback save failed'}), 500

@app.route('/api/clear_history', methods=['DELETE', 'OPTIONS'])
def clear_history():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        with get_db() as db:
            db.execute('DELETE FROM translations')
            db.commit()
        return jsonify({'message': 'All translation history cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'error': 'Clear history failed'}), 500

# 健康檢查 - 簡化版本
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'active_connections': len(active_connections)
    })

def check_colab_status() -> str:
    try:
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.get(COLAB_HEALTH_URL, headers=headers, timeout=5)
        return 'online' if response.status_code == 200 else 'offline'
    except:
        return 'offline'

# 錯誤處理 - 簡化版本
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

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
    logger.warning(f"Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()
    try:
        socketio.stop()
    except:
        pass
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    logger.warning("Starting Flask application...")
    logger.warning(f"Database path: {DATABASE_PATH}")
    
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
