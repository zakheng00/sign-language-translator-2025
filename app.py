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
from threading import Thread
from concurrent.futures import ThreadPoolExecutor
import pyrebase
import tempfile
import atexit

# 初始化 Flask 應用
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 禁用 ONEDNN 以避免 TensorFlow 兼容性問題
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 配置日誌
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定義路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

# 全局變量（懶加載）
model = None
labels = None
vosk_model = None
recognizer = None
executor = ThreadPoolExecutor(max_workers=2)
rooms = {}  # 全局房間存儲
db = None  # Firebase 數據庫，初始化可能失敗

# 從環境變量加載 Firebase 配置
firebase_service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT', '{}')
temp_file_path = None

try:
    with tempfile.NamedTemporaryFile(mode='w', delete=False) as temp_file:
        temp_file.write(firebase_service_account_json)
        temp_file_path = temp_file.name
    logger.info("Firebase service account file created successfully")
except json.JSONDecodeError:
    logger.error("Invalid JSON in FIREBASE_SERVICE_ACCOUNT environment variable")
    temp_file_path = None  # 設置為 None，後續處理
except Exception as e:
    logger.error(f"Failed to create temporary file for Firebase service account: {e}")
    temp_file_path = None  

if temp_file_path:
    firebase_config = {
    "apiKey": os.environ.get('FIREBASE_API_KEY', 'YOUR_API_KEY'),
    "authDomain": os.environ.get('FIREBASE_AUTH_DOMAIN', 'signlanguagetranslator-cce9e.firebaseapp.com'),
    "databaseURL": os.environ.get('FIREBASE_DATABASE_URL', 'https://signlanguagetranslator-cce9e-default-rtdb.asia-southeast1.firebasedatabase.app'),
    "projectId": os.environ.get('FIREBASE_PROJECT_ID', 'signlanguagetranslator-cce9e'),
    "storageBucket": os.environ.get('FIREBASE_STORAGE_BUCKET', 'signlanguagetranslator-cce9e.appspot.com'),
    "messagingSenderId": os.environ.get('FIREBASE_MESSAGING_SENDER_ID', 'YOUR_MESSAGING_SENDER_ID'),
    "appId": os.environ.get('FIREBASE_APP_ID', 'YOUR_APP_ID'),
    "serviceAccount": temp_file_path
}

    try:
        firebase = pyrebase.initialize_app(firebase_config)
        db = firebase.database()
        
        try:
            test_ref = db.child("test").push({"test": "ping"})
            logger.info("Firebase connection test succeeded")
        except Exception as e:
            logger.warning(f"Firebase connection test failed: {e} - Continuing with limited functionality")
    except Exception as e:
        logger.error(f"Failed to initialize Firebase: {e}")
        db = None  # 設置為 None，後續處理
else:
    logger.warning("Firebase service account not configured, running without Firebase support")
    db = None

# 確保臨時文件在應用退出時刪除
@atexit.register
def cleanup():
    if temp_file_path and os.path.exists(temp_file_path):
        try:
            os.unlink(temp_file_path)
            logger.info(f"Cleaned up temporary file: {temp_file_path}")
        except Exception as e:
            logger.error(f"Failed to clean up temporary file: {e}")
    # 清理其他臨時文件
    for fname in ['input.webm', 'temp.wav']:
        if os.path.exists(fname):
            try:
                os.remove(fname)
                logger.info(f"Cleaned up temporary file: {fname}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {fname}: {e}")

# 懶加載模型
def load_models():
    global model, labels, vosk_model, recognizer
    if model is None:
        try:
            logger.info("Loading model and labels...")
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels = json.load(f)
            logger.info("Sign language model and labels loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load model or labels: {e}")
            raise
    if vosk_model is None:
        try:
            if not os.path.exists(VOSK_MODEL_PATH):
                raise FileNotFoundError(f"Vosk model file {VOSK_MODEL_PATH} does not exist")
            vosk_model = Model(VOSK_MODEL_PATH)
            recognizer = KaldiRecognizer(vosk_model, 16000)
            logger.info("Vosk model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            raise

# 音頻轉錄
def transcribe_audio(audio_data):
    load_models()
    try:
        with open("input.webm", "wb") as f:
            f.write(audio_data)
        ffmpeg.input('input.webm').output('temp.wav', ac=1, ar='16000').run(overwrite_output=True)
        with wave.open("temp.wav", "rb") as wf:
            if wf.getframerate() != 16000:
                raise ValueError("Audio sample rate must be 16000 Hz")
            result_text = ""
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)
            result = recognizer.FinalResult()
            result_text = json.loads(result).get("text", "")
            return result_text if result_text.strip() else "Unable to recognize speech"
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return "Transcription error"

# 手語預測（異步）
def predict_gesture_async(frames, room_id):
    load_models()
    if db is None:
        logger.error("Firebase not initialized, skipping gesture prediction")
        return
    try:
        logger.info("Starting model inference in thread")
        keypoints_sequence = np.array(frames, dtype=np.float32).reshape(1, 100, 74, 3)
        keypoints_sequence = (keypoints_sequence - keypoints_sequence.mean(axis=(0, 1))) / (keypoints_sequence.std(axis=(0, 1)) + 1e-8)
        keypoints_sequence = np.expand_dims(keypoints_sequence, axis=-1)
        prediction = model.predict(keypoints_sequence, verbose=0)
        pred_probs = prediction[0].tolist()
        pred_index = np.argmax(prediction, axis=-1)[0]
        gesture = labels[str(pred_index)] if str(pred_index) in labels else 'Unknown'
        logger.info(f"Prediction result: {gesture}, Probabilities: {pred_probs}")
        db.child("rooms").child(room_id).child("messages").push({
            "type": "gesture",
            "data": gesture,
            "probabilities": pred_probs,
            "timestamp": firebase.ServerValue.TIMESTAMP
        })
    except Exception as e:
        logger.error(f"Prediction failed in thread: {e}")

# 路由
@app.route('/')
def index():
    logger.info("Accessed homepage")
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    logger.info("Accessed live sign language translation page")
    return send_from_directory('templates', 'live-translation.html')

@app.route('/speech-to-text')
def speech_to_text():
    logger.info("Accessed speech-to-text page")
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/room-mode')
def room_mode():
    logger.info("Accessed room-mode page")
    return send_from_directory('templates', 'room-mode.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logger.info("Received /transcribe request")
    try:
        if 'audio' not in request.files:
            logger.error("Missing audio file")
            return jsonify({'error': "Missing audio file"}), 400

        audio_file = request.files['audio']
        audio_data = audio_file.read()
        room_id = request.headers.get('X-Socket-ID')
        transcription = transcribe_audio(audio_data)
        logger.info(f"Transcription result: {transcription}")
        if db and room_id:
            db.child("rooms").child(room_id).child("messages").push({
                "type": "transcription",
                "data": transcription,
                "timestamp": firebase.ServerValue.TIMESTAMP
            })
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("Received /predict request")
    try:
        if not request.is_json:
            logger.error("Request is not JSON format")
            return jsonify({'error': 'Request must be in JSON format'}), 400

        data = request.get_json()
        if 'frames' not in data:
            logger.error("Missing 'frames' field")
            return jsonify({'error': "Request missing 'frames' field"}), 400

        frames = data['frames']
        logger.info(f"Received {len(frames)} frames")
        room_id = request.headers.get('X-Socket-ID')
        if room_id:
            executor.submit(predict_gesture_async, frames, room_id)
            return jsonify({'status': 'processing'})
        else:
            return jsonify({'error': 'No valid session'}), 400
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/create_room', methods=['POST'])
def create_room():
    logger.info("Received /create_room request")
    if db is None:
        logger.error("Firebase not initialized, cannot create room")
        return jsonify({'error': 'Firebase service unavailable', 'status': 'failure'}), 500
    try:
        room_id = str(uuid4())
        rooms[room_id] = {'users': []}  # 確保 rooms 已初始化
        db.child("rooms").child(room_id).set({"users": [], "messages": []})
        logger.info(f"Created room with ID: {room_id}")
        return jsonify({'room_id': room_id, 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to create room: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/create_room', methods=['POST'])
def create_room():
    logger.info("Received /create_room request")
    if db is None:
        logger.error("Firebase not initialized, cannot create room")
        return jsonify({'error': 'Firebase service unavailable', 'status': 'failure'}), 500
    try:
        room_id = str(uuid4())
        rooms[room_id] = {'users': []}
        logger.debug(f"Attempting to set room at: rooms/{room_id}")
        db.child("rooms").child(room_id).set({"users": [], "messages": []})
        # 驗證寫入
        room_data = db.child("rooms").child(room_id).get().val()
        if not room_data or room_data.get("users") is None or room_data.get("messages") is None:
            logger.warning(f"Room {room_id} data verification failed: {room_data}")
        else:
            logger.info(f"Room {room_id} data verified: {room_data}")
        logger.info(f"Created room with ID: {room_id}")
        return jsonify({'room_id': room_id, 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to create room: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        load_models()
        logger.info("Starting Flask server")
        app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    except Exception as e:
        logger.error(f"Server failed to start: {e}")