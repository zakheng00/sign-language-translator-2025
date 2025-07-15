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
CORS(app, resources={r"/*": {"origins": "*"}})
socketio = SocketIO(app, cors_allowed_origins=["https://sign-language-translator-2025.onrender.com"])
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# Colab API 端點 (替換為實際 NGROK URL)
COLAB_URL = "https://0ae5c8df1dae.ngrok-free.app/predict_colab"
COLAB_STT_URL = "https://0ae5c8df1dae.ngrok-free.app/speech_to_text"

# --- SQLite 設置 ---
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'translations.db')

@contextmanager
def get_db():
    """上下文管理器，確保數據庫連接正確關閉"""
    db = None
    try:
        db = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
        db.row_factory = sqlite3.Row  # 允許通過列名訪問
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

@app.route('/history-page')  # 改名避免冲突
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
        logger.info(f"Processing video file: {video_file.filename}")
        files = {'video': (video_file.filename, video_file, 'video/mp4')}
        
        # 添加錯誤重試機制
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = requests.post(COLAB_URL, files=files, timeout=60)
                response.raise_for_status()
                result = response.json()
                logger.info(f"Received prediction from Colab: {result}")
                return jsonify(result)
            except requests.exceptions.Timeout:
                if attempt < max_retries - 1:
                    logger.warning(f"Timeout attempt {attempt + 1}, retrying...")
                    time.sleep(2)
                    continue
                raise
                
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
        
        response = requests.post(COLAB_STT_URL, files=files, timeout=60)
        response.raise_for_status()
        result = response.json()
        logger.info(f"Received speech to text result from Colab: {result}")
        return jsonify(result)
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Failed to connect to Colab for speech to text: {e}")
        return jsonify({'error': 'Failed to process audio on Colab'}), 500

# --- 歷史記錄 API ---
@app.route('/api/history', methods=['GET', 'OPTIONS'])  # 改為 /api/history
def get_history():
    if request.method == 'OPTIONS':
        return '', 204
    
    logger.info(f"Received history request from origin: {request.headers.get('Origin')}")
    try:
        with get_db() as db:
            cursor = db.execute('SELECT * FROM translations ORDER BY timestamp DESC LIMIT 100')
            history = []
            for row in cursor.fetchall():
                history.append({
                    'id': row['id'],
                    'user': row['user'],
                    'text': row['text'],
                    'gesture': row['gesture'],
                    'timestamp': row['timestamp']
                })
        return jsonify(history)
    except Exception as e:
        logger.error(f"Error fetching history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/save_history', methods=['POST', 'OPTIONS'])  # 改為 /api/save_history
def save_history():
    if request.method == 'OPTIONS':
        return '', 204
    
    logger.info(f"Received save request from origin: {request.headers.get('Origin')}")
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # 數據驗證
        user = data.get('user', 'Unknown')
        text = data.get('text', '')
        gesture = data.get('gesture', 0)
        timestamp = data.get('timestamp', datetime.datetime.utcnow().isoformat())
        
        with get_db() as db:
            cursor = db.execute(
                'INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                (user, text, gesture, timestamp)
            )
            db.commit()
            
            # 返回插入的記錄 ID
            record_id = cursor.lastrowid
            
        return jsonify({
            'message': 'History saved successfully',
            'id': record_id,
            'data': {
                'user': user,
                'text': text,
                'gesture': gesture,
                'timestamp': timestamp
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}")
        return jsonify({'error': f'Failed to save history: {str(e)}'}), 500

@app.route('/api/delete_history/<int:record_id>', methods=['DELETE', 'OPTIONS'])
def delete_history(record_id):
    if request.method == 'OPTIONS':
        return '', 204
    
    try:
        with get_db() as db:
            cursor = db.execute('DELETE FROM translations WHERE id = ?', (record_id,))
            db.commit()
            
            if cursor.rowcount == 0:
                return jsonify({'error': 'Record not found'}), 404
            
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
        'colab_status': check_colab_status()
    })

def check_colab_status():
    """檢查 Colab 服務狀態"""
    try:
        response = requests.get(f"{COLAB_URL.replace('/predict_colab', '/health')}", timeout=5)
        return 'online' if response.status_code == 200 else 'offline'
    except:
        return 'offline'

# --- 錯誤處理 ---
@app.errorhandler(404)
def not_found(error):
    return jsonify({'error': 'Not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    logger.error(f"Internal server error: {str(error)}")
    return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)
