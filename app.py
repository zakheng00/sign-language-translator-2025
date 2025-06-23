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
import eventlet

# Initialize Flask app and SocketIO
app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")

# Disable ONEDNN for better TensorFlow compatibility on Render
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# Configure logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Define paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model_with_flex.tflite')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

# Global variables for model and recognizer
interpreter = None
input_details = None
output_details = None
labels = None
recognizer = None

def load_models():
    global interpreter, input_details, output_details, labels, recognizer
    try:
        logger.info("Loading model and labels...")
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"Model file {MODEL_PATH} does not exist")
        if not os.path.exists(LABELS_PATH):
            raise FileNotFoundError(f"Labels file {LABELS_PATH} does not exist")

        interpreter = tflite.Interpreter(model_path=MODEL_PATH)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logger.info("TensorFlow Lite model with Flex support loaded successfully")

        with open(LABELS_PATH, 'r', encoding='utf-8') as f:
            labels = json.load(f)
        logger.info("Labels loaded successfully")

        if not os.path.exists(VOSK_MODEL_PATH):
            raise FileNotFoundError(f"Vosk model file {VOSK_MODEL_PATH} does not exist")
        vosk_model = Model(VOSK_MODEL_PATH)
        recognizer = KaldiRecognizer(vosk_model, 16000)
        logger.info("Vosk model loaded successfully")
    except Exception as e:
        logger.error(f"Loading failed: {e}")
        sys.exit(1)

def cleanup_files():
    """Clean up temporary files."""
    for fname in ['input.webm', 'temp.wav']:
        if os.path.exists(fname):
            try:
                os.remove(fname)
                logger.info(f"Cleaned up temporary file: {fname}")
            except Exception as e:
                logger.warning(f"Failed to clean up {fname}: {e}")

def transcribe_audio(audio_data):
    """Transcribe audio data using Vosk."""
    try:
        with open("input.webm", "wb") as f:
            f.write(audio_data)
        ffmpeg.input('input.webm').output('temp.wav', ac=1, ar='16000').run(overwrite_output=True)
        with wave.open("temp.wav", "rb") as wf:
            if wf.getframerate() != 16000:
                raise ValueError("Audio sample rate must be 16000 Hz")
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if not recognizer.AcceptWaveform(data):
                    logger.warning("Partial recognition failed")
            result = recognizer.FinalResult()
            result_text = json.loads(result).get("text", "")
            return result_text if result_text.strip() else "Unable to recognize speech"
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return "Transcription error"
    finally:
        cleanup_files()

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
    logger.info("Accessed room mode page")
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
        transcription = transcribe_audio(audio_data)
        logger.info(f"Transcription result: {transcription}")
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({'error': str(e)}), 500
    finally:
        cleanup_files()

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

        frames = frames[:100] if len(frames) >= 100 else frames
        if len(frames) == 0:
            return jsonify({'gesture': 'No frames received', 'probabilities': []})

        if frames.shape[1] != 74 * 3:
            logger.error(f"Keypoints shape error: expected {74 * 3}, got {frames.shape[1]}")
            return jsonify({'error': f"Keypoints shape error: expected {74 * 3}, got {frames.shape[1]}"}), 400

        keypoints_sequence = frames.reshape(1, len(frames), 74, 3)
        keypoints_sequence = (keypoints_sequence - keypoints_sequence.mean(axis=(0, 1))) / (keypoints_sequence.std(axis=(0, 1)) + 1e-8)
        keypoints_sequence = np.expand_dims(keypoints_sequence, axis=-1).astype(np.float32)
        logger.info(f"Keypoints sequence shape: {keypoints_sequence.shape}")

        logger.info("Starting model inference")
        interpreter.set_tensor(input_details[0]['index'], keypoints_sequence)
        interpreter.invoke()
        prediction = interpreter.get_tensor(output_details[0]['index'])
        pred_probs = prediction[0].tolist()
        pred_index = np.argmax(prediction[0])
        gesture = labels.get(str(pred_index), 'Unknown')
        logger.info(f"Probabilities: {pred_probs}")
        logger.info(f"Result: {gesture} (Index: {pred_index})")

        # Emit result to connected clients in the same room
        room = request.args.get('room', 'default_room')
        socketio.emit('translation_result', {'gesture': gesture, 'probabilities': pred_probs, 'type': 'sign'}, room=room)
        return jsonify({'gesture': gesture, 'probabilities': pred_probs})
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({'error': str(e)}), 500

@socketio.on('join_room')
def on_join(data):
    """Handle room join event."""
    room = data.get('room', 'default_room')
    join_room(room)
    emit('join_ack', {'room': room}, room=room)
    logger.info(f"User joined room: {room}")

@socketio.on('translation_result')
def on_translation_result(data):
    """Broadcast translation result to room."""
    room = data.get('room', 'default_room')
    emit('translation_result', data, room=room)
    logger.info(f"Broadcasted translation to room: {room}")

if __name__ == '__main__':
    try:
        logger.info("Starting Flask server with SocketIO")
        load_models()
        port = int(os.environ.get('PORT', 5000))
        socketio.run(app, debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"Server failed to start: {e}")
        sys.exit(1)