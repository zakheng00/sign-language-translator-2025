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

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app, resources={r"/*": {"origins": "*"}})  # 開發時使用 *, 生產環境限制
socketio = SocketIO(app, cors_allowed_origins=["*"], async_mode='eventlet')  # 使用 eventlet 支持 WebSocket
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Colab API 端點 (替換為實際 NGROK URL)
COLAB_URL = "https://0ae5c8df1dae.ngrok-free.app/predict_colab"  # 根據最新 Colab URL 更新
COLAB_STT_URL = "https://0ae5c8df1dae.ngrok-free.app/speech_to_text"  # 根據最新 Colab URL 更新

# --- SQLite 設置 ---
def get_db():
    try:
        # 改進環境檢測
        is_colab = 'google.colab' in str(getattr(__import__('google.colab', fromlist=['']), '', ''))
        db_path = '/content/drive/My Drive/translations.db' if is_colab else os.path.join(os.path.dirname(__file__), 'translations.db')
        db = sqlite3.connect(db_path, check_same_thread=False)
        db.execute('''CREATE TABLE IF NOT EXISTS translations
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user TEXT,
                       text TEXT,
                       gesture INTEGER,
                       timestamp TEXT)''')
        return db
    except (sqlite3.OperationalError, ImportError) as e:
        logger.error(f"SQLite error: {str(e)}")
        raise

def init_db():
    with get_db() as db:
        db.commit()
    logger.info("SQLite database initialized.")

# 確保數據庫初始化
init_db()

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

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
        return '', 204
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

@app.route('/speech_to_text', methods=['POST', 'OPTIONS'])
def speech_to_text():
    if request.method == 'OPTIONS':
        return '', 204
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
@app.route('/save_history', methods=['POST', 'OPTIONS'])
def save_history():
    if request.method == 'OPTIONS':
        return '', 204
    logger.info(f"Received save request from origin: {request.headers.get('Origin')}")
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        data['timestamp'] = data.get('timestamp', str(datetime.datetime.utcnow()))
        with get_db() as db:
            db.execute('INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                       (data.get('user', 'Unknown'), data.get('text', ''), data.get('gesture', 0), data['timestamp']))
            db.commit()
        return jsonify({'message': 'History saved successfully', 'data': data}), 200
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return '', 204
    logger.info(f"Received history request from origin: {request.headers.get('Origin')}")
    try:
        with get_db() as db:
            cursor = db.execute('SELECT * FROM translations ORDER BY timestamp DESC')
            history = [{'id': row[0], 'user': row[1], 'text': row[2], 'gesture': row[3], 'timestamp': row[4]} for row in cursor.fetchall()]
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# --- WebSocket 事件 ---
@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('message', {'msg': '🟢 Connected to chat room', 'sid': request.sid})

@socketio.on('join_room')
def on_join(data):
    room = data.get('room', 'default')
    join_room(room)
    logger.info(f"Client {request.sid} joined room: {room}")
    emit('joined_room', {'msg': f'✅ Joined {room}', 'sid': request.sid}, room=room)

@socketio.on('message')
def handle_message(data):
    room = data.get('room', 'default')
    logger.info(f"Received message from {data.get('sid', 'Unknown')} in room {room}: {data['msg']}")
    emit('message', {'msg': data['msg'], 'sid': data.get('sid', 'Unknown')}, room=room, broadcast=True)

@socketio.on('translation')
def handle_translation(data):
    room = data.get('room', 'default')
    logger.info(f"Received translation from {data.get('sid', 'Unknown')} in room {room}: {data}")
    emit('translation', data, room=room, broadcast=True)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
