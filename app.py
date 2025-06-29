from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import logging
import os
import wave
from vosk import Model, KaldiRecognizer
import ffmpeg
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import pyrebase
import tempfile
import atexit
import time

# 禁用 ONEDNN（避免某些平台下的 TensorFlow 問題）
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 初始化 Flask
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 日誌配置
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 模型與 Firebase 配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

model = None
labels = None
vosk_model = None
recognizer = None
executor = ThreadPoolExecutor(max_workers=2)
rooms = {}
db = None

# Firebase 服務賬戶配置
firebase_service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT', '{}')
temp_file_path = None

try:
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(firebase_service_account_json)
        temp_file_path = temp_file.name
except Exception as e:
    logger.error(f"Failed to create temp Firebase key: {e}")

# 初始化 Firebase
if temp_file_path:
    firebase_config = {
        "apiKey": os.environ.get("FIREBASE_API_KEY", "YOUR_API_KEY"),
        "authDomain": os.environ.get("FIREBASE_AUTH_DOMAIN", "YOUR_DOMAIN"),
        "databaseURL": os.environ.get("FIREBASE_DATABASE_URL", ""),
        "projectId": os.environ.get("FIREBASE_PROJECT_ID", ""),
        "storageBucket": os.environ.get("FIREBASE_STORAGE_BUCKET", ""),
        "messagingSenderId": os.environ.get("FIREBASE_MESSAGING_SENDER_ID", ""),
        "appId": os.environ.get("FIREBASE_APP_ID", ""),
        "serviceAccount": temp_file_path
    }

    if not firebase_config["databaseURL"]:
        logger.error("Missing databaseURL")
    else:
        try:
            firebase = pyrebase.initialize_app(firebase_config)
            db = firebase.database()
            db.child("test").push({"init": "ok"})
            logger.info("Firebase connected successfully.")
        except Exception as e:
            logger.error(f"Firebase init failed: {e}")
            db = None

# 清理臨時文件
@atexit.register
def cleanup():
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    for fname in ['input.webm', 'temp.wav']:
        if os.path.exists(fname):
            os.remove(fname)

# 加載模型
def load_models():
    global model, labels, vosk_model, recognizer
    if model is None:
        model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels = json.load(f)
    if vosk_model is None:
        vosk_model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(vosk_model, 16000)

# 音頻轉錄
def transcribe_audio(audio_data):
    load_models()
    try:
        with open("input.webm", "wb") as f:
            f.write(audio_data)
        ffmpeg.input("input.webm").output("temp.wav", ac=1, ar="16000").run(overwrite_output=True)
        with wave.open("temp.wav", "rb") as wf:
            result_text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)
            result = json.loads(recognizer.FinalResult())
            return result.get("text", "") or "Unable to recognize speech"
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return "Transcription error"

# 異步手語預測
def predict_gesture_async(frames, room_id):
    load_models()
    if db is None:
        return
    try:
        keypoints_sequence = np.array(frames, dtype=np.float32).reshape(1, 100, 74, 3)
        keypoints_sequence = (keypoints_sequence - keypoints_sequence.mean(axis=(0, 1))) / (keypoints_sequence.std(axis=(0, 1)) + 1e-8)
        keypoints_sequence = np.expand_dims(keypoints_sequence, axis=-1)
        prediction = model.predict(keypoints_sequence, verbose=0)
        pred_probs = prediction[0].tolist()
        pred_index = np.argmax(prediction, axis=-1)[0]
        gesture = labels.get(str(pred_index), 'Unknown')
        db.child("rooms").child(room_id).child("messages").push({
            "type": "gesture",
            "data": gesture,
            "probabilities": pred_probs,
            "timestamp": firebase.ServerValue.TIMESTAMP
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

# 路由定義
@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')

@app.route('/speech-to-text')
def speech_to_text():
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/room-mode')
def room_mode():
    return send_from_directory('templates', 'room-mode.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': 'Missing audio file'}), 400
        audio_data = request.files['audio'].read()
        room_id = request.headers.get('X-Socket-ID')
        transcription = transcribe_audio(audio_data)
        if db and room_id:
            db.child("rooms").child(room_id).child("messages").push({
                "type": "transcription",
                "data": transcription,
                "timestamp": firebase.ServerValue.TIMESTAMP
            })
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Transcribe error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        frames = data.get("frames", [])
        room_id = request.headers.get("X-Socket-ID")
        if not frames or not room_id:
            return jsonify({'error': 'Missing frames or session'}), 400
        executor.submit(predict_gesture_async, frames, room_id)
        return jsonify({'status': 'processing'})
    except Exception as e:
        logger.error(f"Prediction request error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/create_room', methods=['POST'])
def create_room():
    if db is None:
        return jsonify({'error': 'Firebase unavailable', 'status': 'failure'}), 500
    try:
        room_id = str(uuid4())
        logger.info(f"Creating room: {room_id}")
        db.child("rooms").child(room_id).set({
            "users": [],
            "messages": [],
            "created_at": int(time.time() * 1000)
        })

        # 增加 sleep 時間 + debug log
        for attempt in range(5):
            time.sleep(3)  # 延遲更久給 Firebase
            room_data = db.child("rooms").child(room_id).get().val()
            logger.debug(f"Attempt {attempt + 1}: Read room data = {room_data}")
            if isinstance(room_data, dict) and "users" in room_data and "messages" in room_data:
                logger.info(f"Room {room_id} verified successfully.")
                return jsonify({'room_id': room_id, 'status': 'success'})

        logger.warning(f"Room verification failed after retries: {room_id}")
        return jsonify({'error': 'Room verification failed', 'status': 'failure'}), 500

    except Exception as e:
        logger.error(f"Create room error: {e}")
        return jsonify({'error': str(e), 'status': 'failure'}), 500

@app.route('/join_room', methods=['POST'])
def join_room():
    if db is None:
        return jsonify({'error': 'Firebase unavailable', 'status': 'failure'}), 500
    try:
        data = request.get_json()
        room_id = data.get("room_id")
        room_data = db.child("rooms").child(room_id).get().val()
        if not room_data or "messages" not in room_data or "users" not in room_data:
            return jsonify({'error': 'Room does not exist or is incomplete', 'status': 'failure'}), 404
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Join room error: {e}")
        return jsonify({'error': str(e), 'status': 'failure'}), 500

# 啟動伺服器
if __name__ == '__main__':
    try:
        load_models()
        logger.info("Starting Flask server")
        app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    except Exception as e:
        logger.error(f"Server failed: {e}")
