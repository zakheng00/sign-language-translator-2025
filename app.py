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
from flask_socketio import SocketIO, emit
from contextlib import contextmanager
from typing import Optional, Dict, Any

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')

socketio = SocketIO(app, cors_allowed_origins=["https://sign-language-translator-2025.onrender.com"])
executor = ThreadPoolExecutor(max_workers=4)

# 修復 CORS 配置
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://sign-language-translator-2025.onrender.com",
            "https://*.ngrok-free.app",
            "https://*.ngrok.io",
            "http://localhost:*",
            "http://127.0.0.1:*"
        ],
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
        "max_age": 86400
    }
}, supports_credentials=True)


# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler('app.log')]
)
logger.setLevel(logging.INFO)

# Colab API 端點
COLAB_BASE_URL = os.environ.get('COLAB_BASE_URL', "https://7d1d03ca5ce8.ngrok-free.app")  # 使用 Colab 的 ngrok URL
COLAB_PREDICT_URL = f"{COLAB_BASE_URL}/predict_colab"
COLAB_STT_URL = f"{COLAB_BASE_URL}/speech_to_text"
COLAB_HEALTH_URL = f"{COLAB_BASE_URL}/health"

# 手語映射表
GESTURE_MAPPING = {
    18: "Hello",
    11: "Thank You",
    7: "I Love You",
    8: "Yes",
    19: "Good Bye",
    16: "Sorry",
    0: "Unknown"
}

# --- SQLite 設置 ---
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'translations.db')

@contextmanager
def get_db():
    db = None
    try:
        db = sqlite3.connect(DATABASE_PATH, check_same_thread=False, timeout=10)
        db.row_factory = sqlite3.Row
        yield db
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        if db:
            db.rollback()
        raise
    finally:
        if db:
            db.close()

def init_db():
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS translations
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user TEXT,
                       text TEXT,
                       gesture INTEGER,
                       timestamp TEXT)''')
        db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON translations (timestamp)')
        db.commit()
    logger.info("SQLite database initialized with index.")

init_db()

# 添加請求日誌中間件
@app.before_request
def log_request():
    request_start_time = time.time()
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Origin: {request.headers.get('Origin', 'No Origin')}")
    logger.info(f"User-Agent: {request.headers.get('User-Agent', 'No User-Agent')}")
    logger.info(f"Content-Length: {request.content_length}")
    request.environ['request_start_time'] = request_start_time

@app.after_request
def after_request(response):
    request_start_time = request.environ.get('request_start_time')
    if request_start_time:
        duration = time.time() - request_start_time
        logger.info(f"Request completed in {duration:.2f} seconds")
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,ngrok-skip-browser-warning'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

# 監控資源使用情況
def log_resource_usage():
    process = psutil.Process()
    memory = process.memory_info().rss / 1024 / 1024
    cpu = process.cpu_percent(interval=1)
    logger.info(f"Resource usage - Memory: {memory:.2f} MB, CPU: {cpu:.2f}%")

# --- 測試端點 ---
@app.route('/test')
def test():
    log_resource_usage()
    return jsonify({
        'message': 'Server is running',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'endpoints': ['/health', '/predict', '/speech_to_text', '/api/history']
    })

# --- 頁面路由 ---
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

@app.route('/speech-to-text')
def speech_to_text_page():
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/history')
def history():
    return send_from_directory('templates', 'history.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

# --- API 路由 ---
def process_media_request(endpoint: str, file_key: str, content_type: str, gesture_default: int = 0) -> Dict[str, Any]:
    if request.method == 'OPTIONS':
        return '', 204
    
    if file_key not in request.files:
        return jsonify({'error': f'Missing {file_key} file'}), 400
    
    media_file = request.files[file_key]
    logger.info(f"Processing {file_key} file: {media_file.filename}, content_length: {media_file.content_length}, content_type: {media_file.content_type}")
    
    if media_file.content_length is None or media_file.content_length == 0:
        logger.warning(f"{file_key} file content_length is invalid ({media_file.content_length})")
        try:
            media_data = media_file.read(100)
            logger.info(f"{file_key} file sample: {media_data}")
            if not media_data:
                return jsonify({'error': f'Empty {file_key} data'}), 400
            media_file.seek(0)
        except Exception as e:
            logger.error(f"Failed to read {file_key} file: {e}")
            return jsonify({'error': f'Invalid {file_key} file'}), 400
    
    try:
        files = {file_key: (media_file.filename, media_file, media_file.content_type or content_type)}
        media_id = str(uuid4())
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {'ngrok-skip-browser-warning': 'true'}
                logger.info(f"Sending request to {endpoint}")
                response = requests.post(endpoint, files=files, headers=headers, timeout=120)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Received result from Colab: {result}")
                
                text = result.get('text', 'No transcription' if file_key == 'audio' else 
                                 GESTURE_MAPPING.get(result.get('gesture', gesture_default), 'Unknown gesture'))
                if file_key == 'video' and result.get('predictions'):
                    text = GESTURE_MAPPING.get(result.get('predictions', [0])[0], text)
                
                with get_db() as db:
                    db.execute(
                        'INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                        ('anonymous', text, result.get('gesture', gesture_default), datetime.datetime.utcnow().isoformat())
                    )
                    db.commit()
                
                logger.info(f"Emitting {file_key} translation globally: {text}")
                socketio.emit('translation', {
                    'text': text,
                    'gesture': result.get('gesture', gesture_default),
                    'user': 'anonymous',
                    'video_id': media_id if file_key == 'video' else None
                })
                return jsonify({'translation': text if file_key == 'video' else {'text': text}, 
                               'video_id': media_id if file_key == 'video' else None,
                               'gesture': result.get('gesture', gesture_default)})
            
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Timeout attempt {attempt + 1}/{max_retries}, retrying...")
                    time.sleep(2 ** attempt)
                    continue
                logger.error(f"Max retries reached for {file_key} request")
                return jsonify({'error': f'{file_key} request timed out after max retries'}), 500
            except requests.exceptions.RequestException as e:
                logger.error(f"{file_key} request failed: {e}")
                return jsonify({'error': f'{file_key} request failed: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in {file_key} processing: {e}")
        return jsonify({'error': f'Unexpected error processing {file_key}: {str(e)}'}), 500

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    return process_media_request(COLAB_PREDICT_URL, 'video', 'video/webm;codecs=vp9')

@app.route('/speech_to_text', methods=['POST', 'OPTIONS'])
def speech_to_text():
    return process_media_request(COLAB_STT_URL, 'audio', 'audio/webm;codecs=opus', gesture_default=0)

# --- 歷史記錄 API ---
@app.route('/api/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return '', 204
    
    logger.info(f"Received history request from origin: {request.headers.get('Origin', 'No Origin')}")
    try:
        with get_db() as db:
            cursor = db.execute('SELECT * FROM translations ORDER BY timestamp DESC LIMIT 100')
            history = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Returning {len(history)} history records")
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- 健康檢查 ---
@app.route('/health')
def health_check():
    log_resource_usage()
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'colab_status': check_colab_status(),
        'database': check_database_status()
    })

def check_colab_status() -> str:
    try:
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.get(COLAB_HEALTH_URL, headers=headers, timeout=5)
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

# --- 錯誤處理 ---
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Not found', 'url': request.url}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error', 'details': str(error)}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Database path: {DATABASE_PATH}")
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
