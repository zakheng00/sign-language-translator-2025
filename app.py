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
from pymongo import MongoClient
import datetime

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={
    r"/*": {
        "origins": "https://sign-language-translator-2025.onrender.com",
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "Cache-Control"],
    }
}, supports_credentials=True)
socketio = SocketIO(app, cors_allowed_origins="*")  # 使用預設 threading 模式
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Colab API 端點 (替換為實際 NGROK URL)
COLAB_URL = "https://67387521b9d9.ngrok-free.app/predict_colab"  # 根據最新 Colab URL 更新
COLAB_STT_URL = "https://67387521b9d9.ngrok-free.app/speech_to_text"  # 根據最新 Colab URL 更新

# --- MongoDB 設置 ---
def get_db():
    client = MongoClient('mongodb+srv://zakheng00:Bong2000@@cluster0.xxxxx.mongodb.net/?retryWrites=true&w=majority')
    db = client['sign_language_db']
    return db['translations']

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
def speech_to_text_page():
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/history')
def history():
    return send_from_directory('templates', 'history.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204  # 忽略 favicon 請求

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
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
        logger.error(f"Failed to connect to Colab for prediction: {e}")
        return jsonify({'error': 'Failed to process video on Colab'}), 500

@app.route('/speech_to_text', methods=['POST'])
def speech_to_text():
    if 'audio' not in request.files:
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

# --- 新增歷史記錄相關路由 ---
@app.route('/save_history', methods=['POST'])
def save_history():
    logger.info(f"Received save request from origin: {request.headers.get('Origin')}")
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        db = get_db()
        data['timestamp'] = data.get('timestamp', str(datetime.datetime.utcnow()))
        db.insert_one(data)
        return jsonify({'message': 'History saved successfully', 'data': data}), 200
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history():
    logger.info(f"Received history request from origin: {request.headers.get('Origin')}")
    try:
        db = get_db()
        history = list(db.find().sort('timestamp', -1))
        for record in history:
            record['_id'] = str(record['_id'])  # 將 ObjectId 轉為字符串
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
