import os
import base64
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import firebase_admin
from firebase_admin import credentials, db as firebase_db

import numpy as np
import tensorflow as tf
from vosk import Model, KaldiRecognizer
import wave
import subprocess
import json

# ─── 日志 ───────────────────────────────────
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# ─── Flask ──────────────────────────────────
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# ─── 全局变量 ────────────────────────────────
db_ref = None             # Firebase database reference
executor = ThreadPoolExecutor(max_workers=2)

# ─── 模型路径 ───────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

model = None
labels = None
vosk_model = None
recognizer = None
temp_file_path = None

# ─── Firebase 初始化 ─────────────────────────
def initialize_firebase():
    global db_ref, temp_file_path

    b64 = os.environ.get('FIREBASE_SERVICE_ACCOUNT_BASE64', '')
    url = os.environ.get('FIREBASE_DATABASE_URL', '')

    if not b64:
        logger.error("FIREBASE_SERVICE_ACCOUNT_BASE64 is not set.")
        return
    if not url:
        logger.error("FIREBASE_DATABASE_URL is not set.")
        return

    try:
        # 解码并写入临时 JSON 文件
        decoded = base64.b64decode(b64)
        with tempfile.NamedTemporaryFile(mode='wb', delete=False, suffix='.json') as tf:
            tf.write(decoded)
            temp_file_path = tf.name

        # 初始化 Admin SDK
        cred = credentials.Certificate(temp_file_path)
        app_firebase = firebase_admin.initialize_app(cred, {
            'databaseURL': url
        })
        db_ref = firebase_db.reference(app=app_firebase)
        logger.info("✅ Firebase initialized successfully")
    except Exception as e:
        logger.error(f"🔥 Firebase initialization failed: {e}")

# ─── 模型加载 ─────────────────────────────────
def load_models():
    global model, labels, vosk_model, recognizer
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        logger.info("TensorFlow model loaded")
    if vosk_model is None:
        vosk_model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(vosk_model, 16000)
        logger.info("Vosk model loaded")

# ─── 音频转录 ─────────────────────────────────
def transcribe_audio(audio_data):
    load_models()
    try:
        # 保存上传的 webm
        in_path = tempfile.mktemp(suffix='.webm')
        out_path = tempfile.mktemp(suffix='.wav')
        with open(in_path, 'wb') as f:
            f.write(audio_data)
        # 转换
        subprocess.run(
            ['ffmpeg', '-i', in_path, '-ac', '1', '-ar', '16000', '-y', out_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        # 识别
        with wave.open(out_path, 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                recognizer.AcceptWaveform(data)
        result = json.loads(recognizer.FinalResult())
        return result.get('text', '') or 'Unable to recognize speech'
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 'Transcription error'
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)

# ─── 手语预测（异步）──────────────────────────
def predict_gesture_async(frames, room_id):
    if not db_ref:
        return
    load_models()
    try:
        seq = np.array(frames, dtype=np.float32).reshape(1, 100, 74, 3)
        seq = (seq - seq.mean((0,1))) / (seq.std((0,1)) + 1e-8)
        seq = np.expand_dims(seq, -1)
        pred = model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(pred))
        gesture = labels.get(str(idx), 'Unknown')
        db_ref.child('rooms').child(room_id).child('messages').push({
            'type': 'gesture',
            'data': gesture,
            'probabilities': pred.tolist(),
            'timestamp': firebase_admin.db.ServerValue.TIMESTAMP
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

# ─── 路由 ─────────────────────────────────────
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    f = request.files.get('audio')
    if not f:
        return jsonify({'error': 'Missing audio'}), 400
    text = transcribe_audio(f.read())
    room_id = request.headers.get('X-Socket-ID')
    if db_ref and room_id:
        db_ref.child('rooms').child(room_id).child('messages').push({
            'type': 'transcription',
            'data': text,
            'timestamp': firebase_admin.db.ServerValue.TIMESTAMP
        })
    return jsonify({'transcription': text})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    frames = data.get('frames', [])
    room_id = request.headers.get('X-Socket-ID')
    if not frames or not room_id:
        return jsonify({'error': 'Missing frames or session'}), 400
    executor.submit(predict_gesture_async, frames, room_id)
    return jsonify({'status': 'processing'})

@app.route('/list_rooms', methods=['GET'])
def list_rooms():
    if not db_ref:
        return jsonify({'error': 'Firebase unavailable'}), 500
    rooms = db_ref.child('rooms').get() or {}
    return jsonify([
        {'room_id': rid, 'user_count': len(info.get('users', []))}
        for rid, info in rooms.items()
    ])

@app.route('/join_room', methods=['POST'])
def join_room():
    if not db_ref:
        return jsonify({'error': 'Firebase unavailable'}), 500
    rid = (request.get_json() or {}).get('room_id')
    if not rid:
        return jsonify({'error': 'Missing room_id'}), 400
    info = db_ref.child('rooms').child(rid).get()
    if not info:
        return jsonify({'error': 'Room not found'}), 404
    count = len(info.get('users', []))
    if count >= 2:
        return jsonify({'error': 'Room full'}), 403
    return jsonify({'status': 'success'})

# ─── 启动 ─────────────────────────────────────
if __name__ == '__main__':
    initialize_firebase()   # ← 一定要先调用
    load_models()
    # 预创建两个房间
    if db_ref:
        for _ in range(2):
            rid = uuid4().hex[:8]
            db_ref.child('rooms').child(rid).set({'users': [], 'messages': [], 'created_at': int(time.time()*1000)})
            logger.info(f"Pre-created room: {rid}")
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
