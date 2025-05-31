from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import numpy as np
import tensorflow as tf
import pandas as pd
import json
import logging
import os

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)

# 設置環境變量
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'

# 設置日誌
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 動態生成路徑
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, 'models', 'model.h5')
LABELS_PATH = os.path.join(BASE_DIR, 'models', 'labels.json')  # 改用 JSON 格式

# 載入模型和標籤
try:
    logger.info("正在加載模型和標籤...")
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"模型文件 {MODEL_PATH} 不存在")
    if not os.path.exists(LABELS_PATH):
        raise FileNotFoundError(f"標籤文件 {LABELS_PATH} 不存在")
    
    # 載入模型
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    
    # 載入標籤
    with open(LABELS_PATH, 'r', encoding='utf-8') as f:
        labels = json.load(f)
    logger.info("模型和標籤加載成功")
except Exception as e:
    logger.error(f"加載模型或標籤失敗: {e}")
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
        
        # 檢查幀數和形狀
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
        
        # 重塑為模型輸入形狀
        keypoints_sequence = frames.reshape(1, 100, 74, 3)
        
        # 標準化
        keypoints_sequence = (keypoints_sequence - keypoints_sequence.mean(axis=(0, 1))) / (keypoints_sequence.std(axis=(0, 1)) + 1e-8)
        
        # 增加通道維度
        keypoints_sequence = np.expand_dims(keypoints_sequence, axis=-1)  # 形狀: [1, 100, 74, 3, 1]
        logger.info(f"關鍵點序列形狀: {keypoints_sequence.shape}")
        
        # 運行推斷
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
        port = int(os.environ.get('PORT', 5000))  # 動態端口，適配 Render/Fly.io
        app.run(debug=False, host='0.0.0.0', port=port)  # 部署時禁用 debug
    except Exception as e:
        logger.error(f"伺服器啟動失敗: {e}")