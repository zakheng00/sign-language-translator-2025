import os
import tempfile
import logging
import time
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import psutil
import numpy as np
import tflite_runtime.interpreter as tflite
import json
from collections import Counter
import cv2
import mediapipe as mp
import subprocess

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO

# --- Flask 設置 ---
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet', ping_interval=25, ping_timeout=300)
executor = ThreadPoolExecutor(max_workers=4)

# 設置日誌
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

# --- 模型路徑 ---
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'H.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')

# 全局模型變量
interpreter = None
labels = None
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# --- 模型加載（僅在啟動時執行一次） ---
def load_models():
    global interpreter, labels
    memory = psutil.virtual_memory()
    logger.info(f"Memory usage: {memory.percent}% (total: {memory.total / 1024 / 1024:.2f}MB, available: {memory.available / 1024 / 1024:.2f}MB)")
    try:
        if memory.percent > 70:
            logger.warning("Memory usage exceeds 70%, consider optimizing or upgrading instance")
        if interpreter is None and memory.percent < 70:
            interpreter = tflite.Interpreter(model_path=MODEL_PATH)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            logger.info(f"TFLite model loaded. Input shape: {input_details[0]['shape']}, Output shape: {output_details[0]['shape']}")
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels = json.load(f)
    except Exception as e:
        logger.error(f"Model loading failed: {e}")
        return False
    return True

# 應用啟動時加載模型
if not load_models():
    logger.critical("Failed to load models, application may not function correctly")

# --- 手語預測（異步） ---
def extract_keypoints_from_frame(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    res = hands.process(rgb)
    kps = []
    if res.multi_hand_landmarks:
        for hm in res.multi_hand_landmarks[:2]:
            for lm in hm.landmark:
                kps.extend([lm.x, lm.y])
    while len(kps) < 42:
        kps += [0.0, 0.0]
    return kps[:42]

def normalize_keypoints(kps):
    arr = np.array(kps).reshape(-1, 2)
    center = arr[0]
    arr -= center
    scale = np.linalg.norm(arr, axis=1).max()
    if scale > 0:
        arr /= scale
    return arr.flatten()

def predict_100frames_middle20(video_path):
    memory = psutil.virtual_memory()
    if memory.percent > 80:
        raise ValueError("Insufficient memory available")

    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames in video: {total}")
    if total == 0:
        raise ValueError("Video file is empty or unreadable")

    # 優化：降低分辨率並限制幀數
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)
    step = max(1, total // 30)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i % step == 0 and len(frames) < 30:
            frames.append(frame.copy())
        i += 1
    cap.release()

    if len(frames) < 30:
        raise ValueError(f"Insufficient frames: got {len(frames)}, need >=30")

    # 中間 20 幀（索引 5 到 24）
    mid_frames = frames[5:25]
    predictions = []

    for f in mid_frames:
        kps = extract_keypoints_from_frame(f)
        kps = normalize_keypoints(kps)
        input_vector = np.tile(kps, 10).reshape(1, 420)  # (1, 420)

        input_detail = interpreter.get_input_details()[0]
        output_detail = interpreter.get_output_details()[0]
        interpreter.set_tensor(input_detail['index'], input_vector)
        interpreter.invoke()
        out = interpreter.get_tensor(output_detail['index'])[0]
        pred_label = int(np.argmax(out))
        predictions.append(pred_label)

    # 多数投票得出最终预测
    final_label = Counter(predictions).most_common(1)[0][0]

    logger.info(f"Predicted {final_label} from {len(predictions)} frames")
    return {'gesture': final_label, 'predictions': predictions}

# --- 路由 ---
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'}), 400
    video_file = request.files['video']
    in_path = tempfile.mktemp(suffix='.mp4')
    converted_path = in_path + '.converted.mp4'
    try:
        video_file.save(in_path)
        # 優化 ffmpeg 轉換，增加超時並降低品質
        subprocess.run(['ffmpeg', '-i', in_path, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-y', converted_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        logger.info(f"FFmpeg conversion completed for {in_path}")
        result = executor.submit(predict_100frames_middle20, converted_path).result(timeout=300)
        return jsonify(result)
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg conversion failed: {e.stderr.decode()}")
        return jsonify({'error': 'Video conversion failed'}), 500
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        for path in [in_path, converted_path]:
            if os.path.exists(path):
                os.remove(path)

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)), debug=True)