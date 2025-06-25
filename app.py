from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import logging
import os
import wave
from vosk import Model, KaldiRecognizer
import ffmpeg  # ✅ Added: for format conversion
from uuid import uuid4

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

# In-memory room storage (for simplicity)
rooms = {}

try:
    logger.info("Loading model and labels...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model file {MODEL_PATH} does not exist")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"Labels file {LABELS_PATH} does not exist")

    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    logger.info("Sign language model and labels loaded successfully")

    if not os.path.exists(VOSK_MODEL_PATH):
        raise FileNotFoundError(f"Vosk model file {VOSK_MODEL_PATH} does not exist")
    logger.info(f"Verifying Vosk model directory: {os.path.isdir(VOSK_MODEL_PATH)} - Contents: {os.listdir(VOSK_MODEL_PATH) if os.path.isdir(VOSK_MODEL_PATH) else 'Invalid directory'}")
    vosk_model = Model(VOSK_MODEL_PATH)
    logger.info(f"Vosk model object created: {vosk_model is not None}")
    recognizer = KaldiRecognizer(vosk_model, 16000)
    if recognizer is None:
        raise RuntimeError("Failed to initialize Vosk recognizer")
    logger.info("Vosk model loaded successfully")
except Exception as e:
    logger.error(f"Loading failed: {type(e).__name__} - {str(e)}")
    exit(1)

def transcribe_audio(audio_data):
    global recognizer
    try:
        # ✅ Step 1: Save raw audio as input.webm
        with open("input.webm", "wb") as f:
            f.write(audio_data)

        # ✅ Step 2: Convert input.webm to temp.wav (16kHz, mono) using ffmpeg
        ffmpeg.input('input.webm').output('temp.wav', ac=1, ar='16000').run(overwrite_output=True)

        # ✅ Step 3: Use Vosk for speech recognition
        with wave.open("temp.wav", "rb") as wf:
            if wf.getframerate() != 16000:
                raise ValueError("Audio sample rate must be 16000 Hz")
            logger.info(f"Processing WAV file with sample rate: {wf.getframerate()} Hz")
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
    finally:
        # ✅ Step 4: Clean up temporary files
        for fname in ['input.webm', 'temp.wav']:
            if os.path.exists(fname):
                os.remove(fname)

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

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logger.info("Received /transcribe request")
    try:
        if 'audio' not in request.files:
            logger.error("Missing audio file")
            return jsonify({'error': "Missing audio file"}), 400

        audio_file = request.files['audio']
        audio_data = audio_file.read()

        transcription = transcribe_audio(audio_data)
        logger.info(f"Transcription result: {transcription}")
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({'error': str(e)}), 500)

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
        frames = np.array(frames, dtype=np.float32)

        if len(frames) != 100:
            logger.warning(f"Incorrect number of frames: expected 100, got {len(frames)}")
            if len(frames) < 100:
                padding = np.zeros((100 - len(frames), 74 * 3), dtype=np.float32)
                frames = np.concatenate([frames, padding], axis=0)
            else:
                frames = frames[:100]

        if frames.shape[1] != 74 * 3:
            logger.error(f"Keypoints shape error: expected {74 * 3}, got {frames.shape[1]}")
            return jsonify({'error': f"Keypoints shape error: expected {74 * 3}, got {frames.shape[1]}"}), 400

        keypoints_sequence = frames.reshape(1, 100, 74, 3)
        keypoints_sequence = (keypoints_sequence - keypoints_sequence.mean(axis=(0, 1))) / (keypoints_sequence.std(axis=(0, 1)) + 1e-8)
        keypoints_sequence = np.expand_dims(keypoints_sequence, axis=-1)
        logger.info(f"Keypoints sequence shape: {keypoints_sequence.shape}")

        logger.info("Starting model inference")
        prediction = model.predict(keypoints_sequence, verbose=0)
        pred_probs = prediction[0].tolist()
        pred_index = np.argmax(prediction, axis=-1)[0]
        gesture = labels[str(pred_index)] if str(pred_index) in labels else 'Unknown'
        logger.info(f"Probabilities: {pred_probs}")
        logger.info(f"Result: {gesture} (Index: {pred_index})")
        return jsonify({'gesture': gesture, 'probabilities': pred_probs})
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({'error': str(e)}), 500)

# Room management routes
@app.route('/create_room', methods=['POST'])
def create_room():
    logger.info("Received /create_room request")
    try:
        room_id = str(uuid4())
        rooms[room_id] = {'users': []}
        logger.info(f"Created room with ID: {room_id}")
        return jsonify({'room_id': room_id, 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to create room: {e}")
        return jsonify({'error': str(e), 'status': 'failure'}), 500

@app.route('/join_room', methods=['POST'])
def join_room():
    logger.info("Received /join_room request")
    try:
        data = request.get_json()
        if not data or 'room_id' not in data:
            logger.error("Missing room_id in request")
            return jsonify({'error': 'Missing room_id', 'status': 'failure'}), 400

        room_id = data['room_id']
        if room_id not in rooms:
            logger.error(f"Room {room_id} does not exist")
            return jsonify({'error': 'Room does not exist', 'status': 'failure'}), 404

        user_id = str(uuid4())  # Simple user identification
        rooms[room_id]['users'].append(user_id)
        logger.info(f"User {user_id} joined room {room_id}")
        return jsonify({'user_id': user_id, 'room_id': room_id, 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to join room: {e}")
        return jsonify({'error': str(e), 'status': 'failure'}), 500

@app.route('/rooms', methods=['GET'])
def list_rooms():
    logger.info("Received /rooms request")
    try:
        return jsonify({'rooms': list(rooms.keys()), 'status': 'success'})
    except Exception as e:
        logger.error(f"Failed to list rooms: {e}")
        return jsonify({'error': str(e), 'status': 'failure'}), 500

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")