from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import logging
import os
import wave
from vosk import Model, KaldiRecognizer
import subprocess
from uuid import uuid4
from concurrent.futures import ThreadPoolExecutor
import firebase_admin
from firebase_admin import credentials, db as firebase_db
import tempfile
import atexit
import time

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

model = None
labels = None
vosk_model = None
recognizer = None
executor = ThreadPoolExecutor(max_workers=2)
db = None

# 環境變量和臨時文件處理
firebase_service_account_json = os.environ.get('FIREBASE_SERVICE_ACCOUNT', '{}')
temp_file_path = None
logger.debug(f"FIREBASE_SERVICE_ACCOUNT content length: {len(firebase_service_account_json)}")

if firebase_service_account_json and firebase_service_account_json != '{}':
    try:
        temp_dir = '/tmp'
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        with tempfile.NamedTemporaryFile(mode='w', delete=False, dir=temp_dir) as temp_file:
            temp_file.write(firebase_service_account_json)
            temp_file_path = temp_file.name
        logger.info(f"Temporary Firebase key file created at: {temp_file_path}")
    except Exception as e:
        logger.error(f"Failed to create temp Firebase key: {e}")
else:
    logger.error("FIREBASE_SERVICE_ACCOUNT environment variable is invalid or empty")

# Firebase 初始化
if temp_file_path:
    database_url = os.environ.get("FIREBASE_DATABASE_URL", "")
    logger.debug(f"Using database URL: {database_url}")
    if not database_url:
        logger.error("FIREBASE_DATABASE_URL is not set or empty")
    try:
        logger.info("Attempting to initialize Firebase app")
        cred = credentials.Certificate(temp_file_path)
        firebase_app = firebase_admin.initialize_app(cred, {
            'databaseURL': database_url
        })
        db = firebase_db.reference(app=firebase_app)
        logger.info("Firebase Admin connected successfully")
    except ValueError as ve:
        logger.error(f"Invalid Firebase configuration: {ve}")
    except Exception as e:
        logger.error(f"Firebase initialization failed: {e}")
else:
    logger.error("Skipping Firebase initialization due to missing service account file")

@atexit.register
def cleanup():
    if temp_file_path and os.path.exists(temp_file_path):
        os.remove(temp_file_path)
    for fname in ['input.webm', 'temp.wav']:
        if os.path.exists(fname):
            os.remove(fname)

def load_models():
    global model, labels, vosk_model, recognizer
    if model is None:
        try:
            model = tf.keras.models.load_model(MODEL_PATH, compile=False)
            with open(LABELS_PATH, 'r', encoding='utf-8') as f:
                labels = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load model or labels: {e}")
            raise
    if vosk_model is None:
        try:
            vosk_model = Model(VOSK_MODEL_PATH)
            recognizer = KaldiRecognizer(vosk_model, 16000)
        except Exception as e:
            logger.error(f"Failed to load Vosk model: {e}")
            raise

def transcribe_audio(audio_data):
    load_models()
    if db is None:
        return "Database not initialized"
    try:
        with open("input.webm", "wb") as f:
            f.write(audio_data)
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        subprocess.run(['ffmpeg', '-i', 'input.webm', 'temp.wav', '-ac', '1', '-ar', '16000', '-y'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        with wave.open("temp.wav", "rb") as wf:
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                recognizer.AcceptWaveform(data)
            result = json.loads(recognizer.FinalResult())
            return result.get("text", "") or "Unable to recognize speech"
    except subprocess.CalledProcessError as e:
        logger.error(f"FFmpeg error: {e.stderr.decode()}")
        return "Transcription error: FFmpeg failed"
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return "Transcription error"

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
            "timestamp": db.ServerValue.TIMESTAMP
        })
    except Exception as e:
        logger.error(f"Prediction failed: {e}")

@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')

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
                "timestamp": db.ServerValue.TIMESTAMP
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
        room_id = str(uuid4())[:8]
        logger.info(f"Creating room: {room_id}")
        db.child("rooms").child(room_id).set({
            "users": [],
            "messages": [],
            "created_at": db.ServerValue.TIMESTAMP
        })
        return jsonify({'room_id': room_id, 'status': 'success'})
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
        room_data = db.child("rooms").child(room_id).get()
        if room_data is None:
            return jsonify({'error': 'Room does not exist', 'status': 'failure'}), 404
        room_dict = room_data.val() if hasattr(room_data, 'val') else room_data
        user_count = len(room_dict.get("users", []))
        if user_count >= 2:
            return jsonify({'error': 'Room full', 'status': 'failure'}), 403
        return jsonify({'status': 'success'})
    except Exception as e:
        logger.error(f"Join room error: {e}")
        return jsonify({'error': str(e), 'status': 'failure'}), 500

@app.route('/list_rooms')
def list_rooms():
    if db is None:
        return jsonify({'error': 'Firebase unavailable', 'status': 'failure'}), 500
    try:
        rooms_data = db.child("rooms").get().val() or {}
        result = []
        for room_id, details in rooms_data.items():
            users = details.get("users", [])
            result.append({
                "room_id": room_id,
                "user_count": len(users)
            })
        return jsonify(result)
    except Exception as e:
        logger.error(f"List rooms error: {e}")
        return jsonify({'error': str(e), 'status': 'failure'}), 500

if __name__ == '__main__':
    try:
        load_models()
        if db:
            for _ in range(2):  # 創建 2 個房間
                room_id = str(uuid4())[:8]
                db.child("rooms").child(room_id).set({
                    "users": [],
                    "messages": [],
                    "created_at": db.ServerValue.TIMESTAMP
                })
                logger.info(f"Pre-created room with ID: {room_id}")
        logger.info("Starting Flask server")
        app.run(debug=False, host='0.0.0.0', port=int(os.environ.get('PORT', 5000)))
    except Exception as e:
        logger.error(f"Server failed: {e}")