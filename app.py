import os
import base64
import json
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room, emit
import tensorflow.keras.models as keras_models
from mediapipe.python.solutions import hands

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
rooms = {"room1": {"users": []}, "room2": {"users": []}}
executor = ThreadPoolExecutor(max_workers=2)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# 模型路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'sign_classifier_v3_person1_11to20.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')

# 全局變量
model = None
labels = None
hands = hands.Hands(static_image_mode=True, max_num_hands=2)

def load_models():
    global model, labels
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent}% (total: {memory.total / 1024 / 1024:.2f}MB, available: {memory.available / 1024 / 1024:.2f}MB)")
    if model is None and memory.percent < 80:
        model = keras_models.load_model(MODEL_PATH)
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        logger.info("Model loaded")
    return model is not None

def extract_keypoints_from_frame(frame_data):
    # 假設 frame_data 是從前端傳來的關鍵點數據（暫時跳過 cv2 處理，直接使用前端提取）
    # 在後端直接使用前端傳來的 42 維關鍵點
    return frame_data[:42] if len(frame_data) >= 42 else [0.0] * 42

def normalize_keypoints(kps):
    arr = np.array(kps).reshape(-1, 2)
    center = arr[0]
    arr -= center
    scale = np.linalg.norm(arr, axis=1).max()
    if scale > 0:
        arr /= scale
    return arr.flatten()

def predict_gesture_async(frames, room_id, sid):
    try:
        if len(frames) < 30:
            raise ValueError("Insufficient frames: expected at least 30 frames")
        
        # 選取中間 20 幀（第 6 到 25 幀，索引 5 到 24）
        mid_frames = frames[5:25]
        predictions = []

        for frame_data in mid_frames:
            kps = extract_keypoints_from_frame(frame_data)
            kps = normalize_keypoints(kps)
            input_vector = np.tile(kps, 10).reshape(1, 420)  # 適配 (1, 420)
            pred = model.predict(input_vector, verbose=0)[0]
            pred_label = int(np.argmax(pred))
            predictions.append(pred_label)

        # 多數投票
        from collections import Counter
        final_label, _ = Counter(predictions).most_common(1)[0]
        gesture = labels.get(str(final_label), 'Unknown')
        timestamp = time.time() * 1000
        
        socketio.emit('gesture', {
            'type': 'gesture',
            'data': gesture,
            'probabilities': None,  # 可選：若需要概率，可返回平均概率
            'timestamp': timestamp,
            'sid': sid
        }, room=room_id)
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

# --- 路由與 Socket.IO 事件保持不變 ---
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
    # ... (保持不變)

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
    return jsonify([{"room_id": rid, "user_count": len(info["users"])} for rid, info in rooms.items()])

@app.route('/join_room', methods=['POST'])
def http_join_room():
    data = request.get_json() or {}
    rid = data.get("room_id")
    if rid not in rooms or len(rooms[rid]["users"]) >= 2:
        return jsonify({"error": "Room not found or full"}), 404
    return jsonify({"status": "success"})

@socketio.on('join')
def on_join(data):
    rid = data.get("room_id")
    sid = request.sid
    if rid in rooms and len(rooms[rid]["users"]) < 2:
        join_room(rid)
        rooms[rid]["users"].append(sid)
        emit('user_joined', {'sid': sid, 'timestamp': time.time() * 1000}, room=rid)
    else:
        emit('error', {'msg': 'Cannot join room'}, to=sid)

@socketio.on('message')
def handle_message(data):
    rid = data.get("room_id")
    msg = data.get("msg")
    if rid in rooms and msg:
        emit('message', {"sid": request.sid, "msg": msg, "timestamp": time.time() * 1000}, room=rid)

@socketio.on('leave')
def on_leave(data):
    rid = data.get("room_id")
    sid = request.sid
    if rid in rooms and sid in rooms[rid]["users"]:
        leave_room(rid)
        rooms[rid]["users"].remove(sid)
        emit('user_left', {'sid': sid, 'timestamp': time.time() * 1000}, room=rid)

@socketio.on('ping')
def on_ping(data):
    rid = data.get("room_id")
    sid = request.sid
    emit('pong', {'sid': sid}, room=rid)

if __name__ == '__main__':
    if load_models():
        print("Pre-created rooms:", list(rooms.keys()))
        socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), 
                     worker_class='eventlet', workers=4, timeout=300)