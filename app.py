from flask import Flask, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
import aiohttp
import base64
from io import BytesIO
import logging
import os
from eventlet import Timeout
import json

# 初始化 Flask 應用
app = Flask(__name__, static_folder='templates')
socketio = SocketIO(app, cors_allowed_origins="*", logger=True, engineio_logger=True, async_mode='eventlet', ping_timeout=180, ping_interval=45)

# 配置日誌
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# 根據 Colab ngrok URL 更新
COLAB_URL = "https://574fa12359f8.ngrok-free.app/predict_colab"  # 請確認此 URL 有效

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

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static'), 'favicon.ico', mimetype='image/vnd.microsoft.icon')

# HTTP predict 路由
@app.route('/predict', methods=['POST'])
async def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'}), 400
    video_file = request.files['video']
    try:
        with Timeout(600, False):
            files = {'video': (video_file.filename, video_file, 'video/mp4')}
            logger.info(f"Sending request to Colab: {COLAB_URL}")
            async with aiohttp.ClientSession() as session:
                async with session.post(COLAB_URL, data=files, timeout=aiohttp.ClientTimeout(total=600)) as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.info(f"Received prediction from Colab: {result}")
                    return jsonify(result)
    except aiohttp.ClientTimeout:
        logger.error("Colab API request timed out")
        return jsonify({'error': 'Colab API request timed out'}), 500
    except aiohttp.ClientError as e:
        logger.error(f"Failed to connect to Colab: {str(e)}")
        return jsonify({'error': f'Failed to process video on Colab: {str(e)}'}), 500
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        return jsonify({'error': f'Processing error: {str(e)}'}), 500

# Socket.IO 事件
@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('message', {'msg': 'Connected to chat room'})

@socketio.on('send_video')
async def handle_video(data):
    user = data.get('user', 'Anonymous')
    logger.info(f"Received video from user: {user}, size: {len(data['video'])}")
    try:
        with Timeout(180, False):
            video_data = data['video']
            files = {'video': ('video.mp4', base64.b64decode(video_data), 'video/mp4')}
            logger.info(f"Sending request to Colab: {COLAB_URL}")
            async with aiohttp.ClientSession() as session:
                async with session.post(COLAB_URL, data=files, timeout=aiohttp.ClientTimeout(total=180)) as response:
                    response.raise_for_status()
                    result = await response.json()
                    logger.info(f"Translation result: {result}")
                    emit('translation', {'gesture': result.get('gesture'), 'user': user, 'error': result.get('error')}, broadcast=True)
    except aiohttp.ClientTimeout:
        logger.error("Colab API request timed out")
        emit('translation', {'error': 'Colab API request timed out', 'user': user}, broadcast=True)
    except aiohttp.ClientError as e:
        logger.error(f"API request failed: {str(e)}")
        emit('translation', {'error': f"API error: {str(e)}", 'user': user}, broadcast=True)
    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        emit('translation', {'error': f"Processing error: {str(e)}", 'user': user}, broadcast=True)

@socketio.on('live_translation')
async def handle_live_translation(data):
    user = data.get('user', 'Anonymous')
    logger.info(f"Received live translation data from user: {user}")
    try:
        with Timeout(30, False):
            landmarks = json.loads(data['data'])
            # 模擬回應（需替換為實際 Colab API 調用）
            result = {'gesture': 'Test Live Gesture', 'error': None}
            logger.info(f"Live translation result: {result}")
            emit('translation', {'gesture': result['gesture'], 'user': user, 'error': result.get('error')}, broadcast=True)
    except Exception as e:
        logger.error(f"Error processing live translation: {str(e)}")
        emit('translation', {'error': f"Live translation error: {str(e)}", 'user': user}, broadcast=True)

if __name__ == '__main__':
    logger.info("Starting server...")
    socketio.run(app, host='0.0.0.0', port=5000)