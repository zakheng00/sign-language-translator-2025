FROM python:3.12-slim
     WORKDIR /app
     COPY . .
     RUN apt-get update && apt-get install -y ffmpeg && apt-get clean && rm -rf /var/lib/apt/lists/*
     RUN pip install --upgrade pip
     RUN pip install -r requirements.txt
     CMD ["gunicorn", "--bind", "0.0.0.0:8080", "app:app"]