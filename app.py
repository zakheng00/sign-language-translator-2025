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
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={
    r"/predict": {"origins": "https://sign-language-translator-2025.onrender.com"},
    r"/speech_to_text": {"origins": "https://sign-language-translator-2025.onrender.com"}
})  # 精確限制 CORS 來源
socketio = SocketIO(app, async_mode='eventlet', ping_timeout=20, ping_interval=25, cors_allowed_origins="https://sign-language-translator-2025.onrender.com") # 限制 Socket.IO 來源
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Colab API 端點 (替換為實際 NGROK URL)
COLAB_URL = "https://4fe696e97fec.ngrok-free.app/predict_colab"  # 根據最新 Colab URL 更新
COLAB_STT_URL = "https://4fe696e97fec.ngrok-free.app/speech_to_text"  # 語音轉文字端點

# --- 路由 ---
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
def speech_to_text():
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        logger.error("Missing video file in request")
        return jsonify({'error': 'Missing video file'}), 400
    video_file = request.files['video']
    try:
        logger.info(f"Processing video file: {video_file.filename}")
        files = {'video': (video_file.filename, video_file, 'video/mp4')}
        response = requests.post(COLAB_URL, files=files, timeout=600)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received prediction from Colab: {result}")
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab: {e}")
        return jsonify({'error': 'Failed to process video on Colab'}), 500

# --- 語音轉文字路由 (新增，指定唯一端點名稱) ---
@app.route('/speech_to_text', methods=['POST'], endpoint='stt')
def speech_to_text():
    if 'audio' not in request.files:
        logger.error("Missing audio file in request")
        return jsonify({'error': 'Missing audio file'}), 400
    audio_file = request.files['audio']
    try:
        logger.info(f"Processing audio file: {audio_file.filename}")
        files = {'audio': (audio_file.filename, audio_file, 'audio/webm;codecs=opus')}
        response = requests.post(COLAB_STT_URL, files=files, timeout=600)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received speech to text result from Colab: {result}")
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab for speech to text: {e}")
        return jsonify({'error': 'Failed to process audio on Colab'}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)