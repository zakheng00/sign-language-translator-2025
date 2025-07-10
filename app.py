from flask import Flask, request, jsonify, send_from_directory
import cv2
import numpy as np
import os
import uuid
from collections import Counter
from tensorflow.keras.models import load_model
import mediapipe as mp
import gc

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  # 限制上传大小为10MB

# 初始化 MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2)

# 模型懒加载
model = None
def get_model():
    global model
    if model is None:
        model = load_model("2.h5")  # 请确保 H.h5 与 app.py 同目录
    return model

# 关键点提取 + 归一化
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

# 预测接口
@app.route("/predict", methods=["POST"])
def predict_from_video():
    if 'video' not in request.files:
        return jsonify({"error": "未找到视频文件"}), 400

    video_file = request.files["video"]
    temp_path = f"/tmp/{uuid.uuid4()}.webm"
    video_file.save(temp_path)

    cap = cv2.VideoCapture(temp_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
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
        os.remove(temp_path)
        return jsonify({"error": "视频帧数不足 30 帧"}), 400

    mid_frames = frames[5:25]
    predictions = []

    for f in mid_frames:
        kps = extract_keypoints_from_frame(f)
        kps = normalize_keypoints(kps)
        input_vector = np.tile(kps, 10).reshape(1, -1)  # 420维输入
        pred = get_model().predict(input_vector, verbose=0)[0]
        label = int(np.argmax(pred))
        predictions.append(label)

    final_label, count = Counter(predictions).most_common(1)[0]

    # 清理内存
    del frames, predictions
    gc.collect()
    os.remove(temp_path)

    return jsonify({"prediction": final_label})

# 主页（可选）
@app.route("/")
def index():
    return send_from_directory(".", "index.html")

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

# 静态资源（JS/CSS）
@app.route("/<path:path>")
def static_proxy(path):
    return send_from_directory(".", path)

# 启动服务（Render 使用 gunicorn 启动）
if __name__ == "__main__":
    app.run(debug=True)
