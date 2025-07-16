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

# Colab API 端點 (請確認 URL 是否有效)
COLAB_URL = "https://16215c21960a.ngrok-free.app/predict_colab"
COLAB_STT_URL = "https://16215c21960a.ngrok-free.app/speech_to_text"

# 手語映射表
GESTURE_MAPPING = {
    18: "Hello",
    11: "Thank You",
    7: "I Love You",
    8: "Yes",
    19: "Good Bye",
    16: "Sorry"
}

# --- 測試端點 ---
@app.route('/test')
def test():
    return jsonify({
        'message': 'Server is running',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'endpoints': [
            '/health',
            '/predict',
            '/speech_to_text'
        ]
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

@app.route('/history-page')
def history_page():
    return send_from_directory('templates', 'history.html')

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
                response = requests.post(COLAB_URL, files=files, timeout=60)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Received prediction from Colab: {result}")
                gesture = result.get('gesture', 0)
                predictions = result.get('predictions', [])
                text = GESTURE_MAPPING.get(gesture, 'Unknown gesture') if gesture else 'No translation'
                if predictions:
                    text = GESTURE_MAPPING.get(predictions[0], text)
                room = request.args.get('room', 'default')
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
        
        response = requests.post(COLAB_STT_URL, files=files, timeout=60)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received speech to text result from Colab: {result}")
        text = result.get('text', 'No transcription')
        room = request.args.get('room', 'default')
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

# --- 健康檢查 ---
@app.route('/health')
def health_check():
    return jsonify({
        'status': 'healthy',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'colab_status': check_colab_status(),
        'database': check_database_status()
    })

def check_colab_status():
    """檢查 Colab 服務狀態"""
    try:
        response = requests.get(f"{COLAB_URL.replace('/predict_colab', '/health')}", timeout=5)
        return 'online' if response.status_code == 200 else 'offline'
    except:
        return 'offline'

def check_database_status():
    """檢查數據庫狀態"""
    try:
        return 'offline'  # 移除 SQLite 後返回 offline
    except:
        return 'offline'

# --- 錯誤處理 ---
@app.errorhandler(404)
def not_found(error):
    logger.warning(f"404 error: {request.url}")
    return jsonify({'error': 'Not found', 'url': request.url}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

# --- SocketIO 事件 ---
@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected, SID: {request.sid}")

@socketio.on('join')
def on_join(data):
    room = data.get('room', 'default')
    logger.info(f"User joined room: {room}, SID: {request.sid}")
    join_room(room)
    emit('connect_status', {'message': f'User {request.sid} connected to room {room}'}, room=room)

@socketio.on('send_message')
def on_send_message(data):
    room = data.get('room', 'default')
    message = data.get('message', '')
    sid = data.get('sid', 'anonymous')
    logger.info(f"Message received in room {room} from SID {sid}: {message}")
    emit('receive_message', {'user': sid, 'message': message, 'sid': request.sid}, room=room)

@socketio.on('translation')
def on_translation(data):
    logger.info(f"Received translation event: {data}")
    room = data.get('room', 'default')
    emit('translation', data, room=room)

@socketio.on('disconnect')
def on_disconnect():
    logger.info(f"Client disconnected, SID: {request.sid}")

# 添加請求日誌中間件
@app.before_request
def log_request():
    logger.info(f"Request: {request.method} {request.url}")
    logger.info(f"Origin: {request.headers.get('Origin', 'No Origin')}")
    logger.info(f"User-Agent: {request.headers.get('User-Agent', 'No User-Agent')}")

# 添加響應頭中間件
@app.after_request
def after_request(response):
    response.headers['ngrok-skip-browser-warning'] = 'true'
    return response

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
