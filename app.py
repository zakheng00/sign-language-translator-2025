from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import requests
import base64
from io import BytesIO
import logging
import os

# 初始化 Flask 應用
app = Flask(__name__, static_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True)

# 配置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 根據 Colab ngrok URL 更新
COLAB_URL = "https://77279c3e3907.ngrok-free.app/predict_colab"  # 請替換為實際 Colab ngrok URL

# 靜態路由
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

# HTTP predict 路由
@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'}), 400
    video_file = request.files['video']
    try:
        files = {'video': (video_file.filename, video_file, 'video/mp4')}
        response = requests.post(f"{COLAB_URL}/predict_colab", files=files, timeout=600)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received prediction from Colab: {result}")
        return jsonify(result)
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab: {e}")
        return jsonify({'error': 'Failed to process video on Colab'}), 500

# Socket.IO 事件
@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('message', {'msg': 'Connected to chat room'})

@socketio.on('send_video')
def handle_video(data):
    user = data.get('user', 'Anonymous')
    logger.info(f"Received video from user: {user}")
    try:
        video_data = data['video']
        files = {'video': ('video.mp4', BytesIO(base64.b64decode(video_data.split(',')[1])), 'video/mp4')}
        response = requests.post(COLAB_URL, files=files, timeout=30)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Translation result: {result}")
        emit('translation', {'gesture': result.get('gesture'), 'user': user, 'error': result.get('error')}, broadcast=True)
    except requests.RequestException as e:
        logger.error(f"API request failed: {str(e)}")
        emit('translation', {'error': f"API error: {str(e)}", 'user': user}, broadcast=True)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        emit('translation', {'error': f"Processing error: {str(e)}", 'user': user}, broadcast=True)

if __name__ == '__main__':
    logger.info("Starting server...")
    socketio.run(app, host='0.0.0.0', port=5000)