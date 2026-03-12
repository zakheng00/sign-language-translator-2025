# Sign Language Translator 2025

A real-time translation website built around Flask, providing sign language image recognition, speech-to-text conversion, history records, management pages, and multilingual interfaces. The backend receives video and audio data uploaded from the frontend, forwards it to the external Colab inference service for processing, and writes the results to SQLite.

## Feature Overview

- Real-time Sign Language Translation: Upload videos and obtain recognition results via `/predict`

- English Speech-to-Text: Upload audio via `/speech_to_text`

- Malay Speech-to-Text: Upload audio via `/speech_to_text_malay`

- Real-time Event Push: Push translation and language setting changes using Flask-SocketIO

- History: Translation results are written to the local SQLite database

- Feedback: Supports sending and managing feedback queries

- Management Page: Provides APIs for translation records, feedback, and statistics

- Multilingual Resources: Includes `en` and `ms` translation files

## Technical Components

- Backend: Flask, Flask-CORS, Flask-SocketIO

- Inference Integration: Requests connecting to external Colab API

- Database: SQLite

- Real-time Communication: Socket.IO

- Deployment: Gunicorn, Docker, Render configuration files

- Others: TensorFlow, MediaPipe, NumPy, psutil, Babel

## Project Structure

```text

.
|-- app.py # Flask main program and API routing

|-- requirements.txt # Python dependencies

|-- Dockerfile # Containerized deployment settings

|-- Procfile # Platform deployment start commands

|-- gunicorn.conf.py # Gunicorn settings

|-- templates/ # Frontend pages

|-- static/ # Static resources and i18n files

|-- translations/ # Babel translation files

|-- models/ # Tag and speech model resources

|-- model.h5 / 2.h5 # Model files

```

## System Requirements

- Python 3.12

- `ffmpeg`

- Connectable external Colab inference service

This project will automatically create `translations.db` upon startup; manual database initialization is not required.

## Local Installation

### 1. Setting up a Virtual Environment

Windows PowerShell:

```powershell
python -m venv .venv

.\.venv\Scripts\Activate.ps1

```

macOS / Linux:

```bash

python3 -m venv .venv

source .venv/bin/activate

```

### 2. Installing Dependent Packages

```bash

pip install --upgrade pip

pip install -r requirements.txt

```

### 3. Installing ffmpeg

Windows can use `winget`:

```powershell

winget install Gyan.FFmpeg

```

Ubuntu / Debian:

```bash

sudo apt-get update

sudo apt-get install -y ffmpeg

```

### 4. Configure environment variables

The application will prioritize using `COLAB_BASE_URL`; if not set, it will attempt to call the application's built-in default ngrok URL.

Windows PowerShell:

```powershell

$env:COLAB_BASE_URL = "https://your-colab-endpoint.example.com"

```

macOS / Linux:

```bash

export COLAB_BASE_URL="https://your-colab-endpoint.example.com"

```

### 5. Start the service

```bash
python app.py

```

By default, it will start at:

- `http://localhost:5000`

## Docker execution

### Image creation

```bash

docker build -t sign-language-translator-2025 .

```

### Start the container

```bash

docker run -p 8080:8080 -e COLAB_BASE_URL="https://your-colab-endpoint.example.com" sign-language-translator-2025

```

The container is started with Gunicorn by default:

```bash
gunicorn --bind 0.0.0.0:8080 app:app

```

## Main Page

- `/`: Homepage

- `/live-translation`: Live translation page

- `/room-mode`: Room mode

- `/speech-to-text`: English speech to text

- `/speech-to-text-malay`: Malay speech to text

- `/history`: History page

- `/settings`: Settings page

- `/admin`: Administration page

## API Overview

### System and Health Checks

- `GET `/test`: Basic testing and system information

- `GET /health`: Health check, Colab status, database status

### Inference Related

- `POST /predict`: Upload video and get sign language translation

- `POST /speech_to_text`: Upload audio and get English-to-text result

- `POST /speech_to_text_malay`: Upload audio and get Malay-to-text result

### User Data

- `GET /api/history`: Get the 100 most recent translation records

- `DELETE /api/clear_history`: Clear translation records

- `POST /api/feedback`: Submit feedback

- `POST /api/settings`: Save language settings

- `POST `/api/set_fixed_translations`: Sets the default translation content.

### Managing the API

- `GET /api/admin/translations`: Query translation records, supporting pagination and search.

- `DELETE /api/admin/translations/<id>`: Delete a single translation record.

- `GET /api/admin/feedback`: Query feedback data.

- `DELETE /api/admin/feedback/<id>`: Delete a single feedback record.

- `GET /api/admin/stats`: Retrieves statistical information.

## API Example

### Upload a video for sign language recognition

```bash
curl -X POST http://localhost:5000/predict \
-F "video=@sample.webm"

```

### Upload audio and convert it from English to text

```bash
curl -X POST http://localhost:5000/speech_to_text \
-F "audio=@sample.webm"

```

### Get History

```bash
curl http://localhost:5000/api/history

```

## Environment Variables

| Variable | Description | Default Value |

|---|---|---|

| `PORT` | Service startup port | `5000` |

| `COLAB_BASE_URL` | External Colab inference service base URL | Built-in ngrok URL |

## Data Storage

- Database file: `translations.db`

- Tables: `translations`, `feedback`

- Automatically creates tables and indexes upon startup.

## Deployment-Related Files

- `Dockerfile`: Creates the image using `python:3.12-slim`

- `Procfile`: Provides platform-based deployment startup instructions

- `gunicorn.conf.py`: Sets timeout, workers, and threads

- `render.yaml`: Draft settings for Render deployment

- `runtime.txt`: Specifies Python version 3.12

## Known Considerations

- Actual results for sign language and speech recognition depend on the availability of the external Colab API.

- `/predict` and the speech API require files to be uploaded as `multipart/form-data`, not JSON.

- The project includes model and speech resource files; the initial deployment image size may be larger than expected.

- Socket.IO functionality is executed directly on the local machine using `python app.py`. This is the easiest time to check and debug.

## Development Suggestions

- To adjust the default translation content, call `/api/set_fixed_translations`.

- To switch the front-end text, modify `static/i18n/` and `translations/`.

- To change the inference service, simply update `COLAB_BASE_URL`.
