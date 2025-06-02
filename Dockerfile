FROM python:3.12-slim
WORKDIR /app
COPY . .

# 安裝系統依賴
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    ffmpeg \
    build-essential \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 安裝 Python 依賴
RUN pip install --upgrade pip
RUN pip install -r requirements.txt

CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]