import os
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import requests
import json
import sqlite3
import datetime
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, emit
from contextlib import contextmanager

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')

# 修復 CORS 配置 - 允許更多來源
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
        "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"]
    }
})

# 使用 eventlet 作為異步模式
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_timeout=300, ping_interval=60)
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Colab API 端點
COLAB_BASE_URL = "https://e965ae982731.ngrok-free.app"  # 更新為最新 ngrok URL
COLAB_PREDICT_URL = f"{COLAB_BASE_URL}/predict_colab"
COLAB_STT_URL = f"{COLAB_BASE_URL}/speech_to_text"

# 手語映射表
GESTURE_MAPPING = {
    18: "Hello",
    11: "Thank You",
    7: "I Love You",
    8: "Yes",
    19: "Good Bye",
    16: "Sorry"
}

# --- SQLite 設置 ---
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'translations.db')

@contextmanager
def get_db():
    """上下文管理器，確保數據庫連接正確關閉"""
    db = None
    try:
        db = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
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
    """初始化數據庫"""
    with get_db() as db:
        db.execute('''CREATE TABLE IF NOT EXISTS translations
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user TEXT,
                       text TEXT,
                       gesture INTEGER,
                       timestamp TEXT)''')
        db.commit()
    logger.info("SQLite database initialized.")

# 確保數據庫初始化
init_db()

# 添加 CORS 處理中間件
@app.before_request
def handle_preflight():
    if request.method == "OPTIONS":
        response = jsonify({})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add('Access-Control-Allow-Headers', "*")
        response.headers.add('Access-Control-Allow-Methods', "*")
        response.headers.add('ngrok-skip-browser-warning', 'true')
        return response

@app.after_request
def after_request(response):
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Access-Control-Allow-Headers'] = 'Content-Type,Authorization,ngrok-skip-browser-warning'
    response.headers['Access-Control-Allow-Methods'] = 'GET,PUT,POST,DELETE,OPTIONS'
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

# --- 測試端點 ---
@app.route('/test')
def test():
    return jsonify({
        'message': 'Server is running',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'endpoints': ['/predict', '/speech_to_text']
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

@app.route('/favicon.ico')
def favicon():
    return '', 204

# --- API 路由 ---
@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'}), 400
    
    video_file = request.files['video']
    try:
        logger.info(f"Processing video file: {video_file.filename}, content_length: {video_file.content_length}, content_type: {video_file.content_type}")
        files = {'video': (video_file.filename, video_file, video_file.content_type or 'video/webm')}
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                headers = {'ngrok-skip-browser-warning': 'true'}
                response = requests.post(COLAB_PREDICT_URL, files=files, headers=headers, timeout=60)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Received prediction from Colab: {result}")
                gesture = result.get('gesture', 0)
                predictions = result.get('predictions', [])
                text = GESTURE_MAPPING.get(gesture, 'Unknown gesture') if gesture else 'No translation'
                if predictions:
                    text = GESTURE_MAPPING.get(predictions[0], text)
                # 保存到數據庫
                with get_db() as db:
                    db.execute(
                        'INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                        ('anonymous', text, gesture, datetime.datetime.utcnow().isoformat())
                    )
                    db.commit()
                room = request.args.get('room', 'default')
                logger.info(f"Emitting translation to room: {room}, text: {text}")
                socketio.emit('translation', {
                    'text': text,
                    'gesture': gesture,
                    'user': 'anonymous',
                    'sid': 'server',
                    'room': room
                }, room=room)
                return jsonify({'translation': text, 'gesture': gesture})
            except requests.exceptions.RequestException as e:
                logger.error(f"Colab request failed (attempt {attempt + 1}/{max_retries}): {e}, Response: {getattr(response, 'text', 'No response')}")
                if attempt < max_retries - 1:
                    time.sleep(2)
                    continue
                raise
                
    except Exception as e:
        logger.error(f"Unexpected error in predict: {e}")
        return jsonify({'error': f'Failed to process video on Colab: {str(e)}'}), 500

@app.route('/speech_to_text', methods=['POST', 'OPTIONS'])
def speech_to_text():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'audio' not in request.files:
        return jsonify({'error': 'Missing audio file'}), 400
    
    audio_file = request.files['audio']
    try:
        logger.info(f"Processing audio file: {audio_file.filename}, content_length: {audio_file.content_length}, content_type: {audio_file.content_type}")
        files = {'audio': (audio_file.filename, audio_file, audio_file.content_type or 'audio/webm;codecs=opus')}
        
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.post(COLAB_STT_URL, files=files, headers=headers, timeout=60)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received speech to text result from Colab: {result}")
        text = result.get('text', 'No transcription')
        # 保存到數據庫
        with get_db() as db:
            db.execute(
                'INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                ('anonymous', text, 0, datetime.datetime.utcnow().isoformat())
            )
            db.commit()
        room = request.args.get('room', 'default')
        logger.info(f"Emitting translation to room: {room}, text: {text}")
        socketio.emit('translation', {
            'text': text,
            'user': 'anonymous',
            'sid': 'server',
            'room': room
        }, room=room)
        return jsonify({'text': text})
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab for speech to text: {e}, Response: {getattr(response, 'text', 'No response')}")
        return jsonify({'error': f'Failed to process audio on Colab: {str(e)}'}), 500

# --- SocketIO 事件處理 ---
@socketio.on('connect')
def handle_connect():
    logger.info(f"Client connected, sid: {request.sid}")
    join_room('default')  # 默認加入 'default' 房間

@socketio.on('join')
def handle_join(data):
    room = data.get('room', 'default')
    join_room(room)
    logger.info(f"Client {request.sid} joined room: {room}")
    emit('joined', {'room': room, 'sid': request.sid}, room=room)

@socketio.on('translation')
def handle_translation(data):
    logger.info(f"Received translation event: {data}")
    room = data.get('room', 'default')
    emit('translation', data, room=room)

# --- 錯誤處理 ---
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Not found', 'url': request.url}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Database path: {DATABASE_PATH}")
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
