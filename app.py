# Colab 端代碼 (例如 Colab_notebook.ipynb)

!pip install flask gunicorn numpy opencv-python-headless mediapipe tensorflow pyngrok flask-cors

import os
import tempfile
import logging
import numpy as np
from collections import Counter
import cv2
import mediapipe as mp
from tensorflow.keras.models import load_model
from flask import Flask, request, jsonify
import subprocess
import gc
from pyngrok import ngrok
import nest_asyncio
from flask_cors import CORS

# 設置 authtoken
AUTHTOKEN = "2zfFeLFJZwFYaOLB2ez0eCLLGM8_LRxYRjWbdFpRJvWZ7vhj"  # 將此處替換為您的 ngrok authtoken
ngrok.set_auth_token(AUTHTOKEN)
nest_asyncio.apply()

# 設置日誌
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型路徑
MODEL_PATH = "H.h5"

# 全局模型變量
model = None
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

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

# 手語預測函數
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

def predict_30frames_middle20_similarity(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    logger.info(f"Total frames in video: {total}")
    if total == 0:
        raise ValueError("Video file is empty or unreadable")

    idxs = np.linspace(0, total - 1, 30, dtype=int)
    frames = []
    i = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if i in idxs:
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
        input_vector = np.tile(kps, 10).reshape(1, -1)  # (1, 420)

        pred = model.predict(input_vector, verbose=0)[0]
        pred_label = int(np.argmax(pred))
        predictions.append(pred_label)

    gc.collect()
    final_label = Counter(predictions).most_common(1)[0][0]
    logger.info(f"Prediction result: {{'gesture': {final_label}, 'predictions': {predictions}}}")
    return {'gesture': final_label, 'predictions': predictions}

# Flask 應用
app = Flask(__name__)
CORS(app, resources={r"/predict_colab": {"origins": "https://sign-language-translator-2025.onrender.com"}})  # 明確指定來源

@app.route('/predict_colab', methods=['POST'])
def predict_colab():
    if 'video' not in request.files:
        return jsonify({'error': 'Missing video file'}), 400
    video_file = request.files['video']
    in_path = tempfile.mktemp(suffix='.mp4')
    converted_path = in_path + '.converted.mp4'
    try:
        video_file.save(in_path)
        subprocess.run(['ffmpeg', '-i', in_path, '-c:v', 'libx264', '-crf', '23', '-c:a', 'aac', '-y', converted_path],
                       check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, timeout=300)
        result = predict_30frames_middle20_similarity(converted_path)
        logger.info(f"Response sent: {result}")
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

# 啟動服務
print("Starting Colab server...")
ngrok_tunnel = ngrok.connect(5000)
print(f"NGROK tunnel URL: {ngrok_tunnel.public_url}")

app.run(host='0.0.0.0', port=5000)