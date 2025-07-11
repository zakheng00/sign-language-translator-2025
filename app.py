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
from flask_socketio import SocketIO, join_room, leave_room, emit

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  # 啟用 CORS，允許所有來源（可根據需要限制）
socketio = SocketIO(app, cors_allowed_origins="*")  # 使用預設 threading 模式
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Colab API 端點 (替換為實際 NGROK URL)
COLAB_URL = "https://a4d2baab3f8e.ngrok-free.app/predict_colab"  # 更新為 Colab 的最新 WebSocket URL

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


@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'}), 400
    video_file = request.files['video']
    try:
        # 將視頻發送到 Colab
        files = {'video': (video_file.filename, video_file, 'video/mp4')}
        response = requests.post(f"{COLAB_URL}/predict_colab", files=files, timeout=600)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received prediction from Colab: {result}")
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab: {e}")
        return jsonify({'error': 'Failed to process video on Colab'}), 500

# --- SocketIO 事件處理 ---
@socketio.on('join_room')
def on_join(data):
    room = data.get('room', request.sid)  # 默認使用客戶端 ID 作為房間
    join_room(room)
    logger.info(f"Client joined room: {room}")
    emit('message', {'msg': f'Joined room {room}'}, room=room)
    # 轉發到 Colab
    requests.post(COLAB_URL, json={'join_room': data})

@socketio.on('leave_room')
def on_leave(data):
    room = data.get('room', request.sid)
    leave_room(room)
    logger.info(f"Client left room: {room}")
    emit('message', {'msg': f'Left room {room}'}, room=room)
    # 轉發到 Colab
    requests.post(COLAB_URL, json={'leave_room': data})

@socketio.on('video_frame')
def handle_video_frame(data):
    try:
        # 轉發實時幀到 Colab
        response = requests.post(COLAB_URL, json={'video_frame': data})
        response.raise_for_status()
        result = response.json()
        room = request.sid  # 默認發送到客戶端所在的房間
        emit('prediction', result, room=room)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab for frame: {e}")
        emit('prediction', {'error': str(e)}, room=request.sid)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)