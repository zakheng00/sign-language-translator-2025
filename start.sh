#!/bin/bash

# start.sh - 啟動腳本

set -e

# 設置環境變量
export PYTHONUNBUFFERED=1
export PYTHONPATH="/opt/render/project/src:$PYTHONPATH"

# 檢查必要的依賴
echo "Checking dependencies..."
python -c "import eventlet, flask_socketio, gunicorn" || {
    echo "Missing dependencies. Installing..."
    pip install -r requirements.txt
}

# 清理舊的 socket 文件
rm -f /tmp/gunicorn.sock

# 設置日志目錄
mkdir -p /tmp/logs

echo "Starting Flask SocketIO application with Gunicorn..."

# 方法1：使用 eventlet worker（推薦）
exec gunicorn \
    --config gunicorn.conf.py \
    --worker-class eventlet \
    --workers 1 \
    --timeout 0 \
    --keep-alive 30 \
    --max-requests 0 \
    --preload \
    --bind "0.0.0.0:${PORT:-5000}" \
    --access-logfile - \
    --error-logfile - \
    --log-level warning \
    "app:app"

# 方法2：如果 eventlet 有問題，使用 gevent（備選）
# exec gunicorn \
#     --config gunicorn.conf.py \
#     --worker-class gevent \
#     --worker-connections 1000 \
#     --workers 1 \
#     --timeout 0 \
#     --keep-alive 30 \
#     --bind "0.0.0.0:${PORT:-5000}" \
#     --access-logfile - \
#     --error-logfile - \
#     --log-level warning \
#     "app:app"

# 方法3：直接使用 Flask 開發服務器（僅用於測試）
# python app.py
