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
from flask_socketio import SocketIO
from contextlib import contextmanager

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')

# 精簡 CORS 配置
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://sign-language-translator-2025.onrender.com",
            "http://localhost:*",
            "http://127.0.0.1:*"
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

socketio = SocketIO(app, cors_allowed_origins="*")
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Colab API 端點
COLAB_URL = "https://5bad14badabc.ngrok-free.app/predict_colab"
COLAB_STT_URL = "https://5bad14badabc.ngrok-free.app/speech_to_text"

# 手語映射表（示例，需根據 Colab 定義調整）
GESTURE_MAPPING = {
    18: "Hello",  # 示例映射
    11: "Thank You",
    7: "I Love You",
    8: "Yes",
    23: "Good"
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
    """初始化數據庫並添加索引"""
    with get_db() as db:
        db.execute('''
            CREATE TABLE IF NOT EXISTS translations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user TEXT,
                text TEXT,
                gesture INTEGER,
                timestamp TEXT,
                video_id TEXT,
                audio_text TEXT
            )
        ''')
        db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON translations (timestamp)')
        db.commit()
    logger.info("SQLite database initialized with index.")

# 確保數據庫初始化
init_db()

def save_history(data):
    """保存翻譯歷史記錄到 SQLite 數據庫"""
    try:
        with get_db() as db:
            cursor = db.execute(
                'INSERT INTO translations (user, text, gesture, timestamp, video_id, audio_text) '
                'VALUES (?, ?, ?, ?, ?, ?)',
                (
                    data.get('user', 'anonymous'),
                    data.get('text', ''),
                    data.get('gesture', 0),
                    data.get('timestamp', datetime.datetime.utcnow().isoformat()),
                    data.get('video_id', None),
                    data.get('audio_text', None)
                )
            )
            db.commit()
            record_id = cursor.lastrowid
            logger.info(f"History saved with ID: {record_id}")
            return record_id
    except sqlite3.Error as e:
        logger.error(f"Error saving history: {str(e)}")
        raise

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

# --- 測試端點 ---
@app.route('/test')
def test():
    return jsonify({
        'message': 'Server is running',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'endpoints': [
            '/api/history',
            '/api/save_history',
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
    if video_file.content_length == 0:
        return jsonify({'error': 'Empty video file uploaded', 'video_id': str(uuid4())}), 400
    
    try:
        logger.info(f"Processing video file: {video_file.filename}, size: {video_file.content_length} bytes")
        files = {'video': (video_file.filename, video_file, 'video/mp4')}
        video_id = str(uuid4())
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(COLAB_URL, files=files, timeout=60)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Colab response for predict: {result}")
                gesture = result.get('gesture', 0)
                predictions = result.get('predictions', [])
                text = GESTURE_MAPPING.get(gesture, 'Unknown gesture') if gesture else 'No translation'
                if predictions:
                    text = GESTURE_MAPPING.get(predictions[0], text)  # 優先使用第一個預測
                frame_count = result.get('frame_count', len(predictions) if predictions else 0)
                if frame_count < 60:  # 假設要求至少 60 幀 (2 秒 @ 30 FPS)
                    logger.warning(f"Insufficient frames: got {frame_count}, padding may affect result")
                save_history({
                    'user': 'anonymous',
                    'text': text,
                    'gesture': gesture,
                    'timestamp': datetime.datetime.utcnow().isoformat(),
                    'video_id': video_id
                })
                socketio.emit('translation', {
                    'text': text,
                    'gesture': gesture,
                    'user': 'anonymous',
                    'sid': 'server',
                    'room': request.args.get('room', 'default')
                }, room=request.args.get('room', 'default'))
                return jsonify({'translation': text, 'video_id': video_id, 'gesture': gesture})
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Timeout attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    continue
                logger.error("Max retries reached for Colab request")
                return jsonify({'error': 'Colab request timed out', 'video_id': video_id}), 504
            except requests.exceptions.RequestException as e:
                logger.error(f"Colab request failed: {e}")
                return jsonify({'error': 'Failed to process video on Colab', 'video_id': video_id}), 500
                
    except Exception as e:
        logger.error(f"Unexpected error in predict: {e}")
        return jsonify({'error': str(e), 'video_id': video_id}), 500

@app.route('/speech_to_text', methods=['POST', 'OPTIONS'])
def speech_to_text():
    if request.method == 'OPTIONS':
        return '', 204
    
    if 'audio' not in request.files:
        return jsonify({'error': 'Missing audio file'}), 400
    
    audio_file = request.files['audio']
    try:
        logger.info(f"Processing audio file: {audio_file.filename}, size: {audio_file.content_length} bytes")
        files = {'audio': (audio_file.filename, audio_file, 'audio/webm;codecs=opus')}
        
        response = requests.post(COLAB_STT_URL, files=files, timeout=60)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Colab response for speech_to_text: {result}")
        text = result.get('text', 'No transcription')
        save_history({
            'user': 'anonymous',
            'text': text,
            'gesture': 0,
            'timestamp': datetime.datetime.utcnow().isoformat(),
            'audio_text': text
        })
        socketio.emit('translation', {
            'text': text,
            'user': 'anonymous',
            'sid': 'server',
            'room': request.args.get('room', 'default')
        }, room=request.args.get('room', 'default'))
        return jsonify({'text': text})
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab for speech to text: {e}")
        return jsonify({'error': 'Failed to process audio on Colab'}), 500
    except Exception as e:
        logger.error(f"Unexpected error in speech_to_text: {e}")
        return jsonify({'error': str(e)}), 500

# --- 歷史記錄 API ---
@app.route('/api/history', methods=['GET', 'OPTIONS'])
@app.route('/history', methods=['GET', 'OPTIONS'])
def get_history():
    if request.method == 'OPTIONS':
        return '', 204
    
    logger.info(f"Received history request from origin: {request.headers.get('Origin')}")
    try:
        with get_db() as db:
            cursor = db.execute('SELECT * FROM translations ORDER BY timestamp DESC LIMIT 100')
            history = [dict(row) for row in cursor.fetchall()]
        logger.info(f"Returning {len(history)} history records")
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_history', methods=['POST', 'OPTIONS'])
def save_history_api():
    if request.method == 'OPTIONS':
        return '', 204
    logger.info(f"Received save request from origin: {request.headers.get('Origin')}, data: {request.get_json()}")
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        record_id = save_history(data)
        return jsonify({
            'message': 'History saved successfully',
            'id': record_id,
            'data': data
        }), 200
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}")
        return jsonify({'error': f'Failed to save history: {str(e)}'}), 500

@app.route('/api/delete_history/<int:record_id>', methods=['DELETE', 'OPTIONS'])
def delete_history(record_id):
    if request.method == 'OPTIONS':
        return '', 204
    
    logger.info(f"Deleting history record: {record_id}")
    try:
        with get_db() as db:
            cursor = db.execute('DELETE FROM translations WHERE id = ?', (record_id,))
            db.commit()
            if cursor.rowcount == 0:
                return jsonify({'error': 'Record not found'}), 404
        logger.info(f"History record {record_id} deleted successfully")
        return jsonify({'message': 'History record deleted successfully'}), 200
    except Exception as e:
        logger.error(f"Error deleting history: {str(e)}")
        return jsonify({'error': f'Failed to delete history: {str(e)}'}), 500

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
    try:
        response = requests.get(f"{COLAB_URL.replace('/predict_colab', '/health')}", timeout=5)
        return 'online' if response.status_code == 200 else 'offline'
    except:
        return 'offline'

def check_database_status():
    try:
        with get_db() as db:
            cursor = db.execute('SELECT COUNT(*) FROM translations')
            count = cursor.fetchone()[0]
            return f'online ({count} records)'
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

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Database path: {DATABASE_PATH}")
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
