import eventlet
eventlet.monkey_patch()
import os
import base64
import json
import cv2
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import tflite_runtime.interpreter as tflite  # 使用 TFLite 運行時
from vosk import Model, KaldiRecognizer
import wave
import subprocess
import numpy as np
import mediapipe as mp        # ← 新增这一行

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, leave_room, emit


# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')
rooms = {"room1": {"users": []}, "room2": {"users": []}}
executor = ThreadPoolExecutor(max_workers=4)  # 增加工作進程數

socketio = SocketIO(
    app,
    cors_allowed_origins="*",
    async_mode='eventlet',
    ping_interval=25,     # 客户端心跳间隔（秒）
    ping_timeout=300      # 心跳超时（秒），要大于 Gunicorn timeout
)
# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 模型路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'H.tflite')  # 更新為 TFLite 模型
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

# 全局模型變量
interpreter = None
labels = None
vosk_model = None
recognizer = None

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# --- 模型加載（僅在啟動時執行一次） ---
def load_models():
    global interpreter, labels, vosk_model, recognizer
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent}% (total: {memory.total / 1024 / 1024:.2f}MB, available: {memory.available / 1024 / 1024:.2f}MB)")
    try:
        if interpreter is None and memory.percent < 70:
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            logger.info(f"TFLite model loaded. Input shape: {input_details[0]['shape']}, Output shape: {output_details[0]['shape']}")
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        if vosk_model is None and memory.percent < 70:
            vosk_model = Model(VOSK_MODEL_PATH)
            recognizer = KaldiRecognizer(vosk_model, 16000)
            logger.info("Vosk model loaded")
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False
    return True

# 應用啟動時加載模型
if not load_models():
    logger.critical("Failed to load models, application may not function correctly")

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
                if not recognizer.AcceptWaveform(data):
                    logger.warning("Partial recognition failure")
            result = json.loads(recognizer.Result() or recognizer.FinalResult())
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
    """
    frames: 前端传来的完整帧列表（任意帧数）
    本函数会等距抽取 30 帧，然后对中间 18 帧做形态分类，多数表决
    """
    start_time = time.time()
    try:
        total = len(frames)
        if total < 30:
            raise ValueError(f"Insufficient frames: got {total}, need >=30")

        # 1. 等距抽取 30 帧
        idxs = np.linspace(0, total-1, 30, dtype=int)
        sampled = [frames[i] for i in idxs]

        # 2. 取中间 18 帧（index 6~23）
        mid = sampled[6:24]

        preds = []
        for raw in mid:
            # --- 新增：把 JSON list 转成 numpy array ---
            frame = np.array(raw, dtype=np.uint8)
            # 如果前端传的是扁平数组，你还需要 reshape 成 (H, W, 3)，
            # 例如： frame = frame.reshape((height, width, 3))

            # 然后再调用提取函数
            kps = extract_keypoints_from_frame(frame)
            kps = normalize_keypoints(kps)
            inp = np.tile(kps, 10).reshape(1, 420).astype(np.float32)

            # TFLite 预测
            input_detail = interpreter.get_input_details()[0]
            output_detail = interpreter.get_output_details()[0]
            interpreter.set_tensor(input_detail['index'], inp)
            interpreter.invoke()
            out = interpreter.get_tensor(output_detail['index'])[0]
            preds.append(int(np.argmax(out)))

        # 多数表决
        final_label, count = Counter(preds).most_common(1)[0]


        # 返回并通过 SocketIO 广播
        if room_id and sid:
            socketio.emit('gesture', {
                'type': 'gesture',
                'data': final_label,
                'predictions': preds,
                'timestamp': time.time() * 1000,
                'sid': sid
            }, room=room_id)

        logger.info(f"Predicted {final_label} from {len(preds)} frames in {time.time()-start_time:.2f}s")
        return {'gesture': final_label, 'predictions': preds}

    except Exception as e:
        logger.error(f"Prediction failed for {sid}: {e}")
        return {'error': str(e)}

def extract_keypoints_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    kps = []
    if res.multi_hand_landmarks:
        for hl in res.multi_hand_landmarks[:2]:
            for lm in hl.landmark:
                kps += [lm.x, lm.y]
    while len(kps) < 42:
        kps += [0.0, 0.0]
    return kps[:42]

def normalize_keypoints(kps):
    arr = np.array(kps).reshape(-1,2)
    ctr = arr[0]
    arr -= ctr
    scale = np.linalg.norm(arr,axis=1).max()
    if scale>0:
        arr /= scale
    return arr.flatten()
# --- 路由 ---
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    f = request.files.get('audio')
    room_id = request.headers.get('X-Socket-ID')
    if not f:
        return jsonify({'error': 'Missing audio'}), 400
    text = transcribe_audio(f.read())
    sid = request.headers.get('X-Socket-ID') or str(uuid4())  # 默認生成 SID
    if room_id:
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
    if not frames:
        return jsonify({'error': 'Missing frames'}), 400
    future = executor.submit(predict_gesture_async, frames, room_id, request.headers.get('X-Socket-ID'))
    return jsonify({'status': 'processing', 'task_id': str(uuid4())})

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
                 worker_class='eventlet', workers=6, timeout=300)  # 增加 workers 和 timeout