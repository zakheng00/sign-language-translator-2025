import os
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import requests
import sqlite3
import datetime
from flask import Flask, request, jsonify, send_from_directory, render_template  # Added render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit, disconnect
from contextlib import contextmanager
from typing import Optional, Dict, Any
import signal
import sys
import threading

# Flask 設置
app = Flask(__name__, static_folder='static', template_folder='templates')

# SocketIO 配置
socketio = SocketIO(
    app, 
    cors_allowed_origins="*",
    async_mode='threading',
    ping_timeout=60,
    ping_interval=25,
    logger=True,
    engineio_logger=True
)

executor = ThreadPoolExecutor(max_workers=2)

# CORS 配置
CORS(app, resources={
    r"/*": {
        "origins": "*",
        "methods": ["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization", "ngrok-skip-browser-warning"],
        "max_age": 86400
    }
}, supports_credentials=False)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger.setLevel(logging.INFO)

# 全局變量
active_connections = set()
shutdown_event = threading.Event()

# 預設翻譯內容和計數器
fixed_translations = [
    {"text": "How are you", "gesture": 18, "confidence": 0.95},
    {"text": "How are you", "gesture": 18, "confidence": 0.95},
    {"text": "I am fine thank you", "gesture": 7, "confidence": 0.95},
    {"text": "Today is Monday", "gesture": 17, "confidence": 0.95},
    {"text": "What is your name", "gesture": 4, "confidence": 0.95},
    {"text": "Can you help me.", "gesture": 3, "confidence": 0.95}
]
predict_count = 0
FIXED_TRANSLATION_LIMIT = 6

# Colab API 端點
def get_colab_base_url():
    try:
        headers = {'ngrok-skip-browser-warning': 'true'}
        response = requests.get("https://9b2523a90fde.ngrok-free.app", 
                              headers=headers, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data.get('colab_url', "https://9b2523a90fde.ngrok-free.app")
    except requests.RequestException as e:
        logger.warning(f"Failed to fetch COLAB_BASE_URL, using default: {e}")
        return "https://9b2523a90fde.ngrok-free.app"

COLAB_BASE_URL = os.environ.get('COLAB_BASE_URL', get_colab_base_url())
COLAB_PREDICT_URL = f"{COLAB_BASE_URL}/predict_colab"
COLAB_STT_URL = f"{COLAB_BASE_URL}/speech_to_text"
COLAB_STT_MALAY_URL = f"{COLAB_BASE_URL}/speech_to_text_malay"
COLAB_HEALTH_URL = f"{COLAB_BASE_URL}/health"

# SQLite 設置
DATABASE_PATH = os.path.join(os.path.dirname(__file__), 'translations.db')

@contextmanager
def get_db():
    db = None
    try:
        db = sqlite3.connect(DATABASE_PATH, check_same_thread=False, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA journal_mode=WAL')
        yield db
    except sqlite3.Error as e:
        logger.error(f"SQLite error: {str(e)}")
        if db:
            try:
                db.rollback()
            except:
                pass
        raise
    finally:
        if db:
            try:
                db.close()
            except:
                pass

def init_db():
    try:
        with get_db() as db:
            db.execute('''CREATE TABLE IF NOT EXISTS translations
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           user TEXT,
                           text TEXT,
                           gesture INTEGER,
                           timestamp TEXT)''')
            db.execute('''CREATE TABLE IF NOT EXISTS feedback
                          (id INTEGER PRIMARY KEY AUTOINCREMENT,
                           user TEXT,
                           feedback TEXT,
                           timestamp TEXT)''')
            db.execute('CREATE INDEX IF NOT EXISTS idx_timestamp ON translations (timestamp)')
            db.commit()
        logger.info("SQLite database initialized with translations and feedback tables.")
    except Exception as e:
        logger.error(f"Database initialization failed: {e}")

def save_feedback(data):
    try:
        with get_db() as db:
            db.execute('INSERT INTO feedback (user, feedback, timestamp) VALUES (?, ?, ?)',
                       (data.get('user', 'anonymous'), data.get('feedback', ''), 
                        datetime.datetime.utcnow().isoformat()))
            db.commit()
        logger.info(f"Feedback saved for user: {data.get('user', 'anonymous')}")
    except Exception as e:
        logger.error(f"Failed to save feedback: {str(e)}")
        raise

init_db()

# Admin page to display all database records
@app.route('/admin', methods=['GET', 'OPTIONS'])
def admin_page():
    if request.method == 'OPTIONS':
        return '', 204
    try:
        with get_db() as db:
            cursor = db.execute('SELECT * FROM translations ORDER BY timestamp DESC')
            rows = cursor.fetchall()
            records = [dict(row) for row in rows]
        logger.info(f"Admin page accessed, retrieved {len(records)} records")
        return render_template('admin.html', records=records)
    except Exception as e:
        logger.error(f"Failed to load admin page: {str(e)}")
        return jsonify({'error': str(e)}), 500

# ... 以下是原始程式碼的其他路由和函數，保持不變 ...

@app.route('/test')
def test():
    log_resource_usage()
    return jsonify({
        'message': 'Server is running',
        'timestamp': datetime.datetime.utcnow().isoformat(),
        'active_connections': len(active_connections),
        'endpoints': ['/health', '/predict', '/speech_to_text', '/speech_to_text_malay', 
                     '/api/history', '/api/settings', '/api/feedback', '/api/clear_history', 
                     '/api/set_fixed_translations', '/admin']
    })

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

@app.route('/speech-to-text-malay')
def speech_to_text_page_malay():
    return send_from_directory('templates', 'speech-to-text-malay.html')

@app.route('/speech-to-text')
def speech_to_text_page():
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/history')
def history():
    return send_from_directory('templates', 'history.html')

@app.route('/settings')
def settings():
    return send_from_directory('templates', 'settings.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

# ... 其他路由和函數（/api/set_fixed_translations, process_media_request, /predict, /speech_to_text, 
# /speech_to_text_malay, /api/history, /api/settings, /api/feedback, /api/clear_history, /health, 
# 錯誤處理等）保持不變，參考您提供的程式碼 ...

# SocketIO 事件處理
@socketio.on('connect')
def handle_connect():
    logger.info(f'Client connected: {request.sid}')
    active_connections.add(request.sid)
    
@socketio.on('disconnect')
def handle_disconnect():
    logger.info(f'Client disconnected: {request.sid}')
    active_connections.discard(request.sid)

@socketio.on('error')
def handle_error(e):
    logger.error(f'SocketIO error: {e}')

# 優雅關閉處理
def signal_handler(signum, frame):
    logger.info(f"Received signal {signum}, shutting down gracefully...")
    shutdown_event.set()
    socketio.stop()
    sys.exit(0)

signal.signal(signal.SIGTERM, signal_handler)
signal.signal(signal.SIGINT, signal_handler)

if __name__ == '__main__':
    logger.info("Starting Flask application...")
    logger.info(f"Database path: {DATABASE_PATH}")
    logger.info(f"Active worker threads: {executor._max_workers}")
    
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
