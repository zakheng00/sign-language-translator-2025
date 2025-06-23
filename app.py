from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
from flask_socketio import SocketIO, join_room, emit
import numpy as np
import tensorflow.lite as tflite
import json
import logging
import os
import wave
import ffmpeg
import sys
import traceback
from vosk import Model, KaldiRecognizer
import eventlet

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Disable ONEDNN for better TensorFlow compatibility on Render
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_with_flex.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-ms-0.3')  # <- MALAY MODEL

# Global variables
interpreter = None
input_details = None
output_details = None
labels = None
recognizer = None


def load_models():
    global interpreter, input_details, output_details, labels, recognizer
    try:
        logger.info(f"Loading TFLite model from {MODEL_PATH} and labels from {LABELS_PATH}...")

        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} does not exist")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels file {LABELS_PATH} does not exist")

        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info("TensorFlow Lite model loaded successfully")

        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels.update(json.load(f)) if labels else labels := json.load(f)
        logger.info("Labels loaded successfully")

        logger.info(f"Checking if Vosk model path exists: {VOSK_MODEL_PATH}")
        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Vosk model folder does not exist: {VOSK_MODEL_PATH}")

        vosk_model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(vosk_model, 16000)
        if recognizer is None:
            raise RuntimeError("Recognizer is None after initialization")

        logger.info("Vosk recognizer initialized successfully")
    except Exception as e:
        logger.error(f"Model loading failed: {type(e).__name__} - {e}")
        logger.debug(traceback.format_exc())
        return False
    return True


def cleanup_files():
    for fname in ['input.webm', 'temp.wav']:
        if os.path.exists(fname):
            try:
                os.remove(fname)
                logger.info(f"Deleted temporary file: {fname}")
            except Exception as e:
                logger.warning(f"Failed to delete {fname}: {e}")


def transcribe_audio(audio_data):
    global recognizer
    if recognizer is None:
        logger.error("Recognizer is not initialized. Check Vosk model loading.")
        return "Recognizer not available"
    try:
        with open("input.webm", "wb") as f:
            f.write(audio_data)
        logger.info("Audio saved to input.webm")

        ffmpeg.input('input.webm').output('temp.wav', ac=1, ar='16000').run(overwrite_output=True)
        logger.info("Audio converted to temp.wav")

        with wave.open("temp.wav", "rb") as wf:
            if wf.getframerate() != 16000:
                raise ValueError("Sample rate must be 16000 Hz")

            result_text = ""
            while True:
                data = wf.readframes(4000)
                if not data:
                    break
                recognizer.AcceptWaveform(data)

            result = recognizer.FinalResult()
            text = json.loads(result).get("text", "")
            return text if text.strip() else "Unable to recognize speech"
    except Exception as e:
        logger.error(f"Transcription error: {e}")
        return "Transcription error"
    finally:
        cleanup_files()


@app.route('/')
def index():
    return send_from_directory('templates', 'index.html')


@app.route('/live-translation')
def live_translation():
    return send_from_directory('templates', 'live-translation.html')


@app.route('/speech-to-text')
def speech_to_text():
    return send_from_directory('templates', 'speech-to-text.html')


@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': "Missing audio file"}), 400
        audio_data = request.files['audio'].read()
        text = transcribe_audio(audio_data)
        return jsonify({'transcription': text})
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': 'Request must be JSON'}), 400

        data = request.get_json()
        frames = data.get('frames', [])
        if not frames:
            return jsonify({'error': 'Missing frames'}), 400

        frames = np.array(frames, dtype=np.float32)
        if frames.shape[1] != 222:
            return jsonify({'error': f"Expected 222 keypoints, got {frames.shape[1]}"}), 400

        frames = frames[:100]
        x = frames.reshape(1, len(frames), 74, 3)
        x = (x - x.mean(axis=(0, 1))) / (x.std(axis=(0, 1)) + 1e-8)
        x = np.expand_dims(x, axis=-1).astype(np.float32)

        interpreter.set_tensor(input_details[0]['index'], x)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details[0]['index'])

        probs = output[0].tolist()
        pred_idx = int(np.argmax(probs))
        gesture = labels.get(str(pred_idx), 'Unknown')

        room = request.args.get('room', 'default')
        socketio.emit('translation_result', {'gesture': gesture, 'probabilities': probs, 'type': 'sign'}, room=room)

        return jsonify({'gesture': gesture, 'probabilities': probs})
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 500


@socketio.on('join_room')
def join(data):
    room = data.get('room', 'default')
    join_room(room)
    emit('join_ack', {'room': room}, room=room)


@socketio.on('translation_result')
def handle_result(data):
    room = data.get('room', 'default')
    emit('translation_result', data, room=room)


if __name__ == '__main__':
    if load_models():
        port = int(os.environ.get('PORT', 5000))
        socketio.run(app, host='0.0.0.0', port=port)
    else:
        logger.error("Startup failed due to model loading error")
        sys.exit(1)
