import os
import tempfile
import vosk
import logging
import numpy as np
from collections import Counter
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify, send_from_directory, render_template
import subprocess
import gc
from pyngrok import ngrok
import nest_asyncio
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from eventlet import Timeout
import json
from vosk import Model, KaldiRecognizer
import wave
import datetime
import sqlite3
from google.colab import drive
from flask_babel import Babel, _

# 掛載 Google Drive 持久化 SQLite
drive.mount('/content/drive')

# 設置 authtoken
AUTHTOKEN = "2zfFeLFJZwFYaOLB2ez0eCLLGM8_LRxYRjWbdFpRJvWZ7vhj"
ngrok.set_auth_token(AUTHTOKEN)
nest_asyncio.apply()

# 設置日誌
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型路徑
MODEL_PATH = "H.h5"
VOSK_MODEL_PATH = "/content/drive/MyDrive/x/vosk-model-en-us-0.22"

# 全局模型變量
model = None
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)

# 模型加載
def load_models():
    global model
    try:
        if not os.path.exists(MODEL_PATH):
            logger.error(f"Model file {MODEL_PATH} not found")
            return False
        model = load_model(MODEL_PATH)
        logger.info("Model loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Failed to load model from {MODEL_PATH}: {str(e)}")
        return False

if not load_models():
    logger.critical("Failed to load models, application may not function correctly")
    raise RuntimeError("Model loading failed")

# 手語預測函數（優化）
def extract_keypoints_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    kps = []
    if res.multi_hand_landmarks:
        for hm in res.multi_hand_landmarks[:2]:
            for lm in hm.landmark:
                kps.extend([lm.x, lm.y])
    return kps[:42] if kps else [0.0] * 42

def normalize_keypoints(kps):
    arr = np.array(kps).reshape(-1, 2)
    if arr.size == 0:
        return np.zeros(42, dtype=float)
    center = arr[0]
    arr -= center
    scale = np.linalg.norm(arr, axis=1).max()
    return arr.flatten() / (scale + 1e-6) if scale > 0 else arr.flatten()

def predict_30frames_middle20_similarity(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames in video: {total}")
    if total == 0:
        raise ValueError("Video file is empty or unreadable")

    idxs = np.linspace(0, total - 1, 30, dtype=int)
    frames = []
    i = 0
    while i < total and len(frames) < 30:
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
            frames.append(frame.copy())
        i += 1
    cap.release()

    if len(frames) < 30:
        logger.warning(f"Insufficient frames: got {len(frames)}, padding with zeros")
        while len(frames) < 30:
            frames.append(np.zeros_like(frames[0]) if frames else np.zeros((480, 640, 3), dtype=np.uint8))

    mid_frames = frames[5:25]
    predictions = []

    for f in mid_frames:
        kps = extract_keypoints_from_frame(f)
        kps = normalize_keypoints(kps)
        input_vector = np.tile(kps, 10).reshape(1, -1)
        pred = model.predict(input_vector, verbose=0)[0]
        pred_label = int(np.argmax(pred))
        predictions.append(pred_label)

    gc.collect()
    final_label = Counter(predictions).most_common(1)[0][0] if predictions else 0
    logger.info(f"Prediction result: {{'gesture': {final_label}, 'predictions': {predictions}}}")
    return {'gesture': final_label, 'predictions': predictions}

# Flask 應用
app = Flask(__name__, static_folder='static', template_folder='templates')
app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = 'translations'

# 初始化 Flask-Babel
babel = Babel(app)

@babel.localeselector
def get_locale():
    try:
        with get_db() as db:
            cursor = db.execute('SELECT language FROM settings ORDER BY id DESC LIMIT 1')
            row = cursor.fetchone()
            return row[0] if row else 'en'
    except Exception as e:
        logger.error(f"Error getting locale: {str(e)}")
        return 'en'

# 修復 CORS 配置
CORS(app, resources={
    r"/*": {
        "origins": [
            "https://sign-language-translator-2025.onrender.com",
            "https://*.ngrok-free.app",
            "https://*.ngrok.io",
            "http://localhost:*",
            "http://127.0.0.1:*"
        ],
        "methods": ["GET", "POST", "DELETE", "OPTIONS"],
        "allow_headers": ["Content-Type", "Cache-Control", "ngrok-skip-browser-warning"],
        "supports_credentials": True
    }
}, supports_credentials=True)

# 使用 eventlet 作為異步模式
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins=["https://sign-language-translator-2025.onrender.com"])

# 添加 ngrok 跳過瀏覽器警告的中間件
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

# --- SQLite 設置 ---
def get_db():
    try:
        db_path = '/content/drive/My Drive/translations.db'
        db = sqlite3.connect(db_path, check_same_thread=False)
        db.execute('''CREATE TABLE IF NOT EXISTS translations
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       user TEXT,
                       text TEXT,
                       gesture INTEGER,
                       timestamp TEXT)''')
        db.execute('''CREATE TABLE IF NOT EXISTS settings
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       language TEXT DEFAULT 'en')''')
        db.execute('''CREATE TABLE IF NOT EXISTS feedback
                      (id INTEGER PRIMARY KEY AUTOINCREMENT,
                       feedback TEXT,
                       timestamp TEXT)''')
        return db
    except sqlite3.OperationalError as e:
        logger.error(f"SQLite error: {str(e)}")
        raise

def init_db():
    with get_db() as db:
        db.commit()
    logger.info("SQLite database initialized with translations, settings, and feedback tables.")

# 確保數據庫初始化
init_db()

# 儲存翻譯記錄（使用 SQLite）
def save_translation(data, video_blob=None, audio_blob=None):
    try:
        with get_db() as db:
            db.execute('INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                       (data.get('user', 'Unknown'), data.get('text', ''), data.get('gesture', 0), str(datetime.datetime.utcnow())))
            db.commit()
        logger.info(f"Translation record saved for user: {data.get('user', 'Unknown')}")
    except Exception as e:
        logger.error(f"Failed to save translation: {str(e)}")
        raise

# 添加頁面路由
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/live-translation')
def live_translation():
    return render_template('live-translation.html')

@app.route('/room-mode')
def room_mode():
    return render_template('room-mode.html')

@app.route('/speech-to-text')
def speech_to_text_page():
    return render_template('speech-to-text.html')

@app.route('/history')
def history():
    return render_template('history.html')

@app.route('/settings')
def settings():
    return render_template('settings.html')

@app.route('/favicon.ico')
def favicon():
    return '', 204

# 添加測試端點
@app.route('/test', methods=['GET', 'OPTIONS'])
def test_endpoint():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        'message': 'Colab server is running',
        'timestamp': str(datetime.datetime.utcnow()),
        'status': 'healthy'
    })

# 添加健康檢查端點
@app.route('/health', methods=['GET', 'OPTIONS'])
def health_check():
    if request.method == 'OPTIONS':
        return '', 204
    return jsonify({
        'status': 'healthy',
        'timestamp': str(datetime.datetime.utcnow()),
        'server': 'colab'
    })

@app.route('/predict_colab', methods=['POST', 'OPTIONS'])
def predict_colab():
    logger.info(f"Received request from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'}), 400
    video_file = request.files['video']
    in_path = tempfile.mktemp(suffix='.mp4')
    converted_path = in_path + '.converted.mp4'
    try:
        video_file.save(in_path)
        subprocess.run(['ffmpeg', '-i', in_path, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-y', converted_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        with Timeout(300):
            result = predict_30frames_middle20_similarity(converted_path)
        with open(converted_path, 'rb') as f:
            video_data = f.read()
        if video_data:
            save_translation(result, video_data, None)
            logger.info(f"Video data saved, size: {len(video_data)} bytes")
        else:
            logger.warning("No video data to save")
        logger.info(f"Response sent: {result}")
        return jsonify(result)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return jsonify({'error': 'Video conversion failed'}), 500
    except Timeout as e:
        logger.error(f"Prediction timed out: {str(e)}")
        return jsonify({'error': 'Prediction timed out'}), 500
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        for path in [in_path, converted_path]:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except Exception as e:
                    logger.warning(f"Failed to clean up {path}: {str(e)}")

@app.route('/speech_to_text', methods=['POST', 'OPTIONS'])
def handle_speech_to_text():
    logger.info(f"Received request from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204
    if 'audio' not in request.files:
        logger.error('Missing audio file')
        return jsonify({'error': 'Missing audio file'}), 400
    audio_file = request.files['audio']
    in_path = tempfile.mktemp(suffix='.webm')
    wav_path = in_path + '.wav'
    try:
        audio_file.save(in_path)
        logger.info(f"Saved audio file to: {in_path}")
        subprocess.run(['ffmpeg', '-i', in_path, '-acodec', 'pcm_s16le', '-ac', '1', '-ar', '16000', '-y', wav_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        logger.info(f"Converted audio to: {wav_path}")
        with wave.open(wav_path, 'rb') as wf:
            logger.info(f"Audio format - channels: {wf.getnchannels()}, rate: {wf.getframerate()}")
            data = wf.readframes(wf.getnframes())
            logger.info(f"Audio data length: {len(data)} bytes")
        result = speech_to_text(wav_path)
        logger.info(f"Speech to text response: {result}")
        if 'text' in result or 'error' in result:
            save_translation(result)
        return jsonify(result)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return jsonify({'error': 'Audio conversion failed'}), 500
    except Exception as e:
        logger.error(f"Speech to text processing failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        for path in [in_path, wav_path]:
            if os.path.exists(path):
                os.remove(path)

# Socket.IO 事件
@socketio.on('connect')
def on_connect():
    logger.info(f"Client connected: {request.sid}")
    emit('message', {'msg': 'Connected to chat room', 'sid': request.sid})

@socketio.on('message')
def handle_message(data):
    logger.info(f"Received message from {data.get('sid', 'Unknown')}: {data['msg']}")
    emit('message', {'msg': data['msg'], 'sid': data.get('sid', 'Unknown')}, broadcast=True)

@socketio.on('send_message')
def handle_send_message(data):
    logger.info(f"Received send_message from {data.get('user', 'Unknown')}: {data.get('message', '')}")
    emit('new_message', {
        'user': data.get('user', 'Unknown'),
        'message': data.get('message', ''),
        'timestamp': str(datetime.datetime.utcnow())
    }, broadcast=True)

@socketio.on('translation')
def handle_translation(data):
    logger.info(f"Received translation request from {data.get('sid', 'Unknown')}: {data.get('gesture')}")
    try:
        save_translation(data)
        emit('translation', {
            'gesture': data.get('gesture', 'Unknown'),
            'text': data.get('text', 'Unknown'),
            'user': data.get('user', 'Unknown'),
            'error': data.get('error'),
            'sid': data.get('sid')
        }, broadcast=True)
    except Exception as e:
        logger.error(f"Failed to handle translation: {e}")
        emit('translation', {'error': str(e), 'sid': data.get('sid')}, broadcast=True)

@socketio.on('language_change')
def handle_language_change(data):
    logger.info(f"Received language change request: {data.get('language')}")
    emit('language_change', {'language': data.get('language')}, broadcast=True)

@socketio.on('disconnect')
def on_disconnect():
    logger.info(f"Client disconnected: {request.sid}")

# Speech-to-Text 函數
if not os.path.exists(VOSK_MODEL_PATH):
    logger.error(f"Vosk model not found at {VOSK_MODEL_PATH}. Please ensure it is manually placed there.")
    raise FileNotFoundError(f"Vosk model not found at {VOSK_MODEL_PATH}. Please ensure it is manually placed there.")
else:
    logger.info(f"Vosk model found at {VOSK_MODEL_PATH}")

try:
    vosk_model = Model(VOSK_MODEL_PATH)
    vosk_recognizer = KaldiRecognizer(vosk_model, 16000)
    logger.info("Vosk model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load Vosk model from {VOSK_MODEL_PATH}: {str(e)}")
    raise

def speech_to_text(audio_path):
    try:
        with wave.open(audio_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getframerate() != 16000:
                raise ValueError(f"Audio format not supported, expected mono 16kHz, got channels: {wf.getnchannels()}, rate: {wf.getframerate()}")
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if vosk_recognizer.AcceptWaveform(data):
                    logger.debug("Vosk accepted waveform data")
                else:
                    logger.debug("Vosk did not accept waveform data")
            result = json.loads(vosk_recognizer.FinalResult())
            text = result.get('text', None)
            if text is None:
                logger.warning("No text recognized by Vosk, full result: %s", result)
                return {'error': 'No transcription detected'}
            logger.info(f"Speech to text result: {text}")
            return {'text': text}
    except Exception as e:
        logger.error(f"Speech to text failed: {e}")
        return {'error': str(e)}

# 歷史記錄管理端點
@app.route('/api/history', methods=['GET'])
def get_history():
    with get_db() as db:
        cursor = db.execute('SELECT * FROM translations')
        rows = cursor.fetchall()
        history = [{"id": row[0], "user": row[1], "text": row[2], "gesture": row[3], "timestamp": row[4]} for row in rows]
        return jsonify(history)

@app.route('/api/save_history', methods=['POST', 'OPTIONS'])
def save_history():
    logger.info(f"Received save request from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        with get_db() as db:
            db.execute('INSERT INTO translations (user, text, gesture, timestamp) VALUES (?, ?, ?, ?)',
                       (data.get('user', 'Unknown'), data.get('text', ''), data.get('gesture', 0), str(datetime.datetime.utcnow())))
            db.commit()
        logger.info(f"History saved for user: {data.get('user', 'Unknown')}")
        return jsonify({'message': 'History saved successfully'})
    except Exception as e:
        logger.error(f"Error saving history: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/delete_history/<int:record_id>', methods=['DELETE', 'OPTIONS'])
def delete_history(record_id):
    logger.info(f"Received delete request for record {record_id} from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204
    try:
        with get_db() as db:
            cursor = db.execute('DELETE FROM translations WHERE id = ?', (record_id,))
            db.commit()
            if cursor.rowcount == 0:
                return jsonify({'error': 'Record not found'}), 404
        logger.info(f"History record {record_id} deleted successfully")
        return jsonify({'message': 'History record deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 新增語言設置端點
@app.route('/api/settings', methods=['GET', 'POST', 'OPTIONS'])
def handle_settings():
    logger.info(f"Received settings request from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204
    try:
        with get_db() as db:
            if request.method == 'GET':
                cursor = db.execute('SELECT language FROM settings ORDER BY id DESC LIMIT 1')
                row = cursor.fetchone()
                language = row[0] if row else 'en'
                return jsonify({'language': language})
            elif request.method == 'POST':
                data = request.get_json()
                if not data or 'language' not in data or data['language'] not in ['en', 'ms']:
                    return jsonify({'error': 'Invalid language selection'}), 400
                db.execute('INSERT INTO settings (language) VALUES (?)', (data['language'],))
                db.commit()
                logger.info(f"Language set to: {data['language']}")
                socketio.emit('language_change', {'language': data['language']}, broadcast=True)
                return jsonify({'message': 'Language updated successfully', 'language': data['language']})
    except Exception as e:
        logger.error(f"Error handling settings: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 新增反饋端點
@app.route('/api/feedback', methods=['POST', 'OPTIONS'])
def handle_feedback():
    logger.info(f"Received feedback request from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204
    try:
        data = request.get_json()
        if not data or 'feedback' not in data or not data['feedback'].strip():
            return jsonify({'error': 'Feedback cannot be empty'}), 400
        with get_db() as db:
            db.execute('INSERT INTO feedback (feedback, timestamp) VALUES (?, ?)',
                       (data['feedback'], str(datetime.datetime.utcnow())))
            db.commit()
        logger.info("Feedback saved successfully")
        return jsonify({'message': 'Feedback submitted successfully'})
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 新增清除所有歷史記錄端點
@app.route('/api/clear_history', methods=['DELETE', 'OPTIONS'])
def clear_history():
    logger.info(f"Received clear history request from origin: {request.headers.get('Origin')}")
    if request.method == 'OPTIONS':
        return '', 204
    try:
        with get_db() as db:
            db.execute('DELETE FROM translations')
            db.commit()
        logger.info("All history records cleared successfully")
        return jsonify({'message': 'All history cleared successfully'})
    except Exception as e:
        logger.error(f"Error clearing history: {str(e)}")
        return jsonify({'error': str(e)}), 500

# 啟動服務
print("Starting Colab server...")
ngrok.kill()
ngrok_tunnel = ngrok.connect(5000, bind_tls=True)
print(f"NGROK tunnel URL: {ngrok_tunnel.public_url}")

# 更新 COLAB_URL 以供前端使用
COLAB_URL = ngrok_tunnel.public_url
print(f"Updated COLAB_URL: {COLAB_URL}")

# 啟動服務器
try:
    socketio.run(app, host='0.0.0.0', port=5000)
except Exception as e:
    logger.error(f"Server failed to start: {str(e)}")
