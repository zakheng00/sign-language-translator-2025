import os
import base64
import json
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room, emit
import numpy as np
import tensorflow as tf
from vosk import Model, KaldiRecognizer
import wave
import subprocess

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
rooms = {
    "room1": {"users": []},
    "room2": {"users": []},
}

executor = ThreadPoolExecutor(max_workers=2)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 模型路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

# 全局模型變量
model = None
labels = None
vosk_model = None
recognizer = None

# --- 模型加載（僅在啟動時執行一次） ---
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

# 應用啟動時加載模型
load_models()

# --- 音頻轉錄 ---
def transcribe_audio(audio_data):
    in_path = tempfile.mktemp(suffix='.webm')
    out_path = tempfile.mktemp(suffix='.wav')
    try:
        with open(in_path, 'wb') as f:
            f.write(audio_data)
        subprocess.run(
            ['ffmpeg', '-i', in_path, '-ac', '1', '-ar', '16000', '-y', out_path],
            check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=30
        )
        with wave.open(out_path, 'rb') as wf:
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                recognizer.AcceptWaveform(data)
        result = json.loads(recognizer.FinalResult())
        return result.get('text', '') or 'Unable to recognize speech'
    except subprocess.CalledProcessError as e:
        logger.error(f"Transcription failed: ffmpeg error - {e.stderr.decode()}")
        return 'Transcription error: ffmpeg failed'
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return 'Transcription error'
    finally:
        for p in (in_path, out_path):
            if os.path.exists(p):
                os.remove(p)

# --- 手語預測（異步） ---
def predict_gesture_async(frames, room_id, sid):
    try:
        seq = np.array(frames, dtype=np.float32).reshape(1, 100, 74, 3)
        seq = (seq - seq.mean((0, 1))) / (seq.std((0, 1)) + 1e-8)
        seq = np.expand_dims(seq, -1)
        pred = model.predict(seq, verbose=0)[0]
        idx = int(np.argmax(pred))
        gesture = labels.get(str(idx), 'Unknown')
        timestamp = time.time() * 1000
        socketio.emit('gesture', {
            'type': 'gesture',
            'data': gesture,
            'probabilities': pred.tolist(),
            'timestamp': timestamp,
            'sid': sid
        }, room=room_id)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

# --- 路由 ---
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    f = request.files.get('audio')
    room_id = request.headers.get('X-Socket-ID')
    if not f or not room_id:
        return jsonify({'error': 'Missing audio or room ID'}), 400
    text = transcribe_audio(f.read())
    sid = request.headers.get('X-Socket-ID')  # 使用客戶端提供的 SID
    socketio.emit('transcription', {
        'type': 'transcription',
        'data': text,
        'timestamp': time.time() * 1000,
        'sid': sid
    }, room=room_id)
    return jsonify({'transcription': text})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json() or {}
    frames = data.get('frames', [])
    room_id = request.headers.get('X-Socket-ID')
    if not frames or not room_id:
        return jsonify({'error': 'Missing frames or session'}), 400
    executor.submit(predict_gesture_async, frames, room_id, request.headers.get('X-Socket-ID'))
    return jsonify({'status': 'processing'})

@app.route('/list_rooms', methods=['GET'])
def list_rooms():
    return jsonify([
        {"room_id": rid, "user_count": len(info["users"])}
        for rid, info in rooms.items()
    ])

@app.route('/join_room', methods=['POST'])
def http_join_room():
    data = request.get_json() or {}
    rid = data.get("room_id")
    if rid not in rooms:
        return jsonify({"error": "Room not found"}), 404
    if len(rooms[rid]["users"]) >= 2:
        return jsonify({"error": "Room full"}), 403
    return jsonify({"status": "success"})

# --- Socket.IO 事件 ---
@socketio.on('join')
def on_join(data):
    rid = data.get("room_id")
    sid = request.sid
    if rid not in rooms or len(rooms[rid]["users"]) >= 2:
        emit('error', {'msg': 'Cannot join room'}, to=sid)
        return
    join_room(rid)
    rooms[rid]["users"].append(sid)
    emit('user_joined', {'sid': sid, 'timestamp': time.time() * 1000}, room=rid)
    logger.info(f"User {sid} joined room {rid}")

@socketio.on('message')
def handle_message(data):
    rid = data.get("room_id")
    msg = data.get("msg")
    if rid in rooms and msg:
        timestamp = time.time() * 1000
        emit('message', {"sid": request.sid, "msg": msg, "timestamp": timestamp}, room=rid)
    else:
        logger.warning(f"Invalid message data: {data}")

@socketio.on('leave')
def on_leave(data):
    rid = data.get("room_id")
    sid = request.sid
    if rid in rooms and sid in rooms[rid]["users"]:
        leave_room(rid)
        rooms[rid]["users"].remove(sid)
        emit('user_left', {'sid': sid, 'timestamp': time.time() * 1000}, room=rid)
    else:
        logger.warning(f"Invalid leave request from {sid} for room {rid}")

if __name__ == '__main__':
    print("Pre-created rooms:", list(rooms.keys()))
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), 
                 worker_class='eventlet', workers=4, timeout=300)