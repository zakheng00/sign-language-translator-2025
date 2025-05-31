from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pandas as pd
import logging
import os



app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 動態生成路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABEL_PATH = os.path.join(BASE_DIR, 'models', 'label.csv')

try:
    logger.info("正在載入模型和標籤...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在")
    if not os.path.exists(LABEL_PATH):
        raise FileNotFoundError(f"標籤文件 {LABEL_PATH} 不存在")
    model = tf.keras.models.load_model(MODEL_PATH)
    labels_df = pd.read_csv(LABEL_PATH)
    labels = {str(row['index']): row['label'] for _, row in labels_df.iterrows()}
    logger.info("模型和標籤載入成功")
except Exception as e:
    logger.error(f"載入模型或標籤失敗: {e}")
    exit(1)

@app.route('/')
def index():
    logger.info("訪問首頁")
    return send_from_directory('templates', 'index.html')

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
        gesture = labels.get(str(pred_index), '未知')
        logger.info(f"機率: {pred_probs}")
        logger.info(f"結果: {gesture} (索引: {pred_index})")
        return jsonify({'gesture': gesture, 'probabilities': pred_probs})
    except Exception as e:
        logger.error(f"推斷失敗: {e}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        logger.info("啟動 Flask 伺服器")
        port = int(os.environ.get('PORT', 5000))  # Heroku 動態分配端口
        app.run(debug=True, host='0.0.0.0', port=port)
    except Exception as e:
        logger.error(f"伺服器啟動失敗: {e}")