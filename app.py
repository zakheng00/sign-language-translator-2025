from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import json
import logging
import os
from google.cloud import speech

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')

# 設置 Google Cloud 憑證
if os.getenv('GOOGLE_APPLICATION_CREDENTIALS_CONTENT'):
    with open('google_credentials.json', 'w') as f:
        f.write(os.getenv('GOOGLE_APPLICATION_CREDENTIALS_CONTENT'))
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = 'google_credentials.json'
else:
    os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = os.path.join(BASE_DIR, 'credentials.json')

try:
    logger.info("正在加載模型和標籤...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"標籤文件 {LABELS_PATH} 不存在")
    
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    logger.info("手語模型和標籤加載成功")
    
    client = speech.SpeechClient()
    logger.info("Google Speech-to-Text 客戶端加載成功")
except Exception as e:
    logger.error(f"加載失敗: {e}")
    exit(1)

def transcribe_audio(audio_data):
    audio = speech.RecognitionAudio(content=audio_data)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=16000,
        language_code="en-US"
    )
    response = client.recognize(config=config, audio=audio)
    if response.results:
        return response.results[0].alternatives[0].transcript
    return "無法識別語音"

@app.route('/')
def index():
    logger.info("訪問主頁")
    return send_from_directory('templates', 'index.html')

@app.route('/live-translation')
def live_translation():
    logger.info("訪問實時手語翻譯頁面")
    return send_from_directory('templates', 'live-translation.html')

@app.route('/speech-to-text')
def speech_to_text():
    logger.info("訪問語音轉文字頁面")
    return send_from_directory('templates', 'speech-to-text.html')

@app.route('/transcribe', methods=['POST'])
def transcribe():
    logger.info("接收到 /transcribe 請求")
    try:
        if 'audio' not in request.files:
            logger.error("缺少 audio 文件")
            return jsonify({'error': "缺少 audio 文件"}), 400
        
        audio_file = request.files['audio']
        audio_data = audio_file.read()
        
        transcription = transcribe_audio(audio_data)
        logger.info(f"轉錄結果: {transcription}")
        return jsonify({'transcription': transcription})
    except Exception as e:
        logger.error(f"轉錄失敗: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/predict', methods=['POST'])
def predict():
    logger.info("接收到 /predict 請求")
    try:
        if not request.is_json:
            logger.error("請求不是 JSON 格式")
            return jsonify({'error': '請求必須是 JSON 格式'}), 400
        
        data = request.get_json()
        if 'frames' not in data:
            logger.error("缺少 'frames' 字段")
            return jsonify({'error': "請求中缺少 'frames' 字段"}), 400
        
        frames = data['frames']
        logger.info(f"收到 {len(frames)} 幀")
        frames = np.array(frames, dtype=np.float32)
        
        if len(frames) != 100:
            logger.warning(f"幀數不正確，預期 100，實際 {len(frames)}")
            if len(frames) < 100:
                padding = np.zeros((100 - len(frames), 74 * 3), dtype=np.float32)
                frames = np.concatenate([frames, padding], axis=0)
            else:
                frames = frames[:100]
        
        if frames.shape[1] != 74 * 3:
            logger.error(f"關鍵點形狀錯誤，預期 {74 * 3}，實際 {frames.shape[1]}")
            return jsonify({'error': f"關鍵點形狀錯誤，預期 {74 * 3}，實際 {frames.shape[1]}"}), 400
        
        keypoints_sequence = frames.reshape(1, 100, 74, 3)
        keypoints_sequence = (keypoints_sequence - keypoints_sequence.mean(axis=(0, 1))) / (keypoints_sequence.std(axis=(0, 1)) + 1e-8)
        keypoints_sequence = np.expand_dims(keypoints_sequence, axis=-1)
        logger.info(f"關鍵點序列形狀: {keypoints_sequence.shape}")
        
        logger.info("開始模型推斷")
        prediction = model.predict(keypoints_sequence, verbose=0)
        pred_probs = prediction[0].tolist()
        pred_index = np.argmax(prediction, axis=-1)[0]
        gesture = labels[str(pred_index)] if str(pred_index) in labels else 'Unknown'
        logger.info(f"機率: {pred_probs}")
        logger.info(f"結果: {gesture} (索引: {pred_index})")
        return jsonify({'gesture': gesture, 'probabilities': pred_probs})
    except Exception as e:
        logger.error(f"推斷失敗: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("啟動 Flask 伺服器")
        port = int(os.environ.get('PORT', 5000))
        app.run(debug=False, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"伺服器啟動失敗: {e}")