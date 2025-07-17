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
from flask import Flask, render_template
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
COLAB_URL = "https://1aa98e78d123.ngrok-free.app/predict_colab"
COLAB_STT_URL = "https://1aa98e78d123.ngrok-free.app/speech_to_text"

# 手語映射表
GESTURE_MAPPING = {
    18: "Hello",
    11: "Thank You",
    7: "I Love You",
    8: "Yes",
    19: "Good Bye",
    16: "Sorry"
}

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
        'endpoints': [
            '/health',
            '/predict',
            '/speech_to_text'
        ]
    })

@app.route('/history')
def history():
    return render_template('history.html')

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
                # 添加 ngrok 頭部
                headers = {'ngrok-skip-browser-warning': 'true'}
                response = requests.post(COLAB_URL, files=files, headers=headers, timeout=60)
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
        
        # 添加 ngrok 頭部
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.post(COLAB_STT_URL, files=files, headers=headers, timeout=60)
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
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.get(f"{COLAB_URL.replace('/predict_colab', '/health')}", headers=headers, timeout=5)
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
