# Dockerfile
FROM python:3.11-slim

# 設置工作目錄
WORKDIR /app

# 設置環境變量
ENV PYTHONUNBUFFERED=1
ENV PYTHONPATH=/app
ENV PORT=5000

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# 複製requirements並安裝Python依賴
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 複製應用程序代碼
COPY . .

# 創建必要的目錄
RUN mkdir -p /app/static /app/templates /tmp/logs

# 設置權限
RUN chmod +x start.sh

# 創建非root用戶
RUN useradd --create-home --shell /bin/bash app && \
    chown -R app:app /app
USER app

# 健康檢查
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# 暴露端口
EXPOSE $PORT

# 啟動應用
CMD ["./start.sh"]
