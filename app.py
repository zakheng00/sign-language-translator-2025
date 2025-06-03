from flask import Flask, request, jsonify, send_from_directory, render_template
from flask_cors import CORS
from flask_babel import Babel, _
import numpy as np
import tensorflow as tf
import json
import logging
import os
import wave
from vosk import Model, KaldiRecognizer
import ffmpeg

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

app.config['BABEL_DEFAULT_LOCALE'] = 'en'
app.config['BABEL_TRANSLATION_DIRECTORIES'] = './translations'
babel = Babel(app)

@babel.localeselector
def get_locale():
    return request.args.get('lang') or 'en'

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')
VOSK_MODEL_PATH = os.path.join(BASE_DIR, 'models', 'vosk-model-small-en-us-0.15')

try:
    logger.info("Loading model and labels...")
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    vosk_model = Model(VOSK_MODEL_PATH)
    recognizer = KaldiRecognizer(vosk_model, 16000)
    logger.info("Models loaded successfully")
except Exception as e:
    logger.error(f"Loading failed: {e}")
    exit(1)

def transcribe_audio(audio_data):
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
            return result_text if result_text.strip() else _("Unable to recognize speech")
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return _("Transcription error")
    finally:
        for fname in ['input.webm', 'temp.wav']:
            if os.path.exists(fname):
                os.remove(fname)

@app.route('/')
def index():
    return render_template('index.html',
        title=_("Sign Language Translator"),
        choose_feature=_("Choose a feature to begin"),
        live_sign=_("Live Sign Translation"),
        live_desc=_("Translate sign language in real-time using your webcam"),
        speech=_("Speech to Text"),
        speech_desc=_("Convert spoken words into text"),
        history=_("Translation History"),
        history_desc=_("View your past translation records"),
        settings=_("Settings"),
        settings_desc=_("Adjust language and model preferences"))

@app.route('/live-translation')
def live_translation():
    return render_template('live-translation.html')

@app.route('/speech-to-text')
def speech_to_text():
    return render_template('speech-to-text.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    try:
        if 'audio' not in request.files:
            return jsonify({'error': _("Missing audio file")}), 400
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        transcription = transcribe_audio(audio_data)
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"Transcription failed: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if not request.is_json:
            return jsonify({'error': _('Request must be in JSON format')}), 400
        data = request.get_json()
        if 'frames' not in data:
            return jsonify({'error': _("Request missing 'frames' field")}), 400

        frames = np.array(data['frames'], dtype=np.float32)
        if len(frames) != 100:
            if len(frames) < 100:
                padding = np.zeros((100 - len(frames), 74 * 3), dtype=np.float32)
                frames = np.concatenate([frames, padding], axis=0)
            else:
                frames = frames[:100]

        if frames.shape[1] != 74 * 3:
            return jsonify({'error': _('Keypoints shape error')}), 400

        keypoints_sequence = frames.reshape(1, 100, 74, 3)
        keypoints_sequence = (keypoints_sequence - keypoints_sequence.mean(axis=(0, 1))) / (keypoints_sequence.std(axis=(0, 1)) + 1e-8)
        keypoints_sequence = np.expand_dims(keypoints_sequence, axis=-1)

        prediction = model.predict(keypoints_sequence, verbose=0)
        pred_probs = prediction[0].tolist()
        pred_index = np.argmax(prediction, axis=-1)[0]
        gesture = labels.get(str(pred_index), _('Unknown'))
        return jsonify({'gesture': gesture, 'probabilities': pred_probs})
    except Exception as e:
        logger.error(f"Inference failed: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000)
