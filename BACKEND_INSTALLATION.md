# Backend Installation Guide

Academic Research Assistant Backend - FastAPI server with local LLM support via Ollama, voice interaction, academic search, and intelligent summarization.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Server](#running-the-server)
- [Ollama Setup](#ollama-setup)
- [API Endpoints](#api-endpoints)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Python | 3.10+ | Runtime |
| Ollama | Latest | Local LLM inference |
| Git | Latest | Version control (optional) |

### Check Python Version

```bash
python --version
# or
python3 --version
```

---

## Installation

### 1. Navigate to Project Directory

```bash
cd D:\AI_Project\PycharmProjects\PycharmProjects\HomeWork9
```

### 2. Create Virtual Environment

**Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**Linux/Mac:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

#### Key Dependencies

| Category | Packages |
|----------|----------|
| **Web Framework** | fastapi, uvicorn, pydantic |
| **LLM/API** | openai, httpx |
| **Vector DB** | chromadb, sentence-transformers |
| **ASR (Speech)** | openai-whisper, faster-whisper |
| **TTS (Speech)** | edge-tts, pyttsx3, TTS |
| **ArXiv** | arxiv, feedparser |
| **Audio** | librosa, soundfile, pydub |
| **Utilities** | python-dotenv, aiohttp, aiofiles |

---

## Configuration

### Create/Edit `.env` File

Create a `.env` file in the project root with the following configuration:

```env
# ========== Local LLM Settings (Ollama) ==========
USE_LOCAL_LLM=true
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b

# ========== OpenAI API (Fallback/Optional) ==========
OPENAI_API_KEY=sk-proj-xxxxxxxxxxxxx
OPENAI_MODEL=gpt-4o

# ========== Notion Integration (Optional) ==========
NOTION_API_KEY=ntn_xxxxxxxxxxxxx
NOTION_DATABASE_ID=YourDatabaseID
ENABLE_NOTION_SYNC=true

# ========== Web Search (Optional) ==========
TAVILY_API_KEY=tvly-xxxxxxxxxxxx

# ========== API Server Settings ==========
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002

# ========== Whisper ASR Settings ==========
WHISPER_MODEL=base        # Options: tiny, base, small, medium, large
WHISPER_DEVICE=cpu        # Options: cpu, cuda
USE_FASTER_WHISPER=true

# ========== TTS Settings ==========
TTS_ENGINE=edge           # Options: edge, pyttsx3, openai
TTS_VOICE=en-US-AriaNeural
TTS_RATE=0                # Speech rate (+/- percentage)

# ========== Vector Database ==========
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# ========== Session Management ==========
MAX_CONTEXT_HISTORY=20
SESSION_TIMEOUT_MINUTES=60

# ========== Feature Flags ==========
ENABLE_VECTOR_SEARCH=true
ENABLE_WEB_SEARCH=true
```

---

## Ollama Setup

### 1. Install Ollama

Download and install from [ollama.com](https://ollama.com)

### 2. Pull the Model

```bash
# Pull qwen3:8b model
ollama pull qwen3:8b

# Verify model is available
ollama list
```

### 3. Start Ollama Service

Ollama typically starts automatically. If not:

**Windows:**
```bash
# Start Ollama from Start Menu or run:
ollama serve
```

**Linux/Mac:**
```bash
ollama serve
```

### 4. Test Ollama

```bash
ollama run qwen3:8b "Hello, can you hear me?"
```

---

## Running the Server

### Start Backend Server

**Option 1: Using Python module**
```bash
python -m backend.main
```

**Option 2: Using Uvicorn directly**
```bash
uvicorn backend.api.server:app --host 0.0.0.0 --port 8000 --reload
```

**Option 3: Using the main entry point**
```bash
python backend/main.py
```

### Verify Server is Running

Open your browser or use curl:

```bash
curl http://localhost:8000/health
```

Expected response:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "services": {
    "asr": true,
    "tts": true,
    "llm": true,
    "notion": true,
    "vector_db": true
  }
}
```

### Server Output

```
============================================================
  Academic Research Assistant - Backend API
============================================================
  Host: 0.0.0.0
  Port: 8000
  Model: qwen3:8b
  TTS Engine: edge
============================================================

INFO:     Started server process
INFO:     Uvicorn running on http://0.0.0.0:8000
```

---

## API Endpoints

### Health & Status

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/status` | GET | System status |

### Chat

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Main chat with tool calling |
| `/api/transcribe` | POST | Transcribe audio file |
| `/api/tts` | POST | Generate speech from text |

### Sessions

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/sessions` | POST | Create new session |
| `/api/sessions/{id}` | GET | Get session details |
| `/api/sessions` | GET | List all sessions |
| `/api/sessions/{id}` | DELETE | Delete session |

### Tools

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/tools/arxiv` | POST | Search ArXiv papers |
| `/api/tools/web-search` | POST | Web search |
| `/api/tools/summarize` | POST | Summarize papers |

### Vector Store

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/vector/search` | POST | Semantic search |
| `/api/vector/index` | POST | Index papers |

---

## Troubleshooting

### Port Already in Use

**Error:** `Address already in use`

**Solution:**
```bash
# Find process using port 8000 (Windows)
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

### Ollama Connection Failed

**Error:** `Connection refused` to `localhost:11434`

**Solution:**
1. Ensure Ollama is installed and running
2. Start Ollama service: `ollama serve`
3. Verify model is pulled: `ollama list`

### Model Not Found

**Error:** `model 'qwen3:8b' not found`

**Solution:**
```bash
ollama pull qwen3:8b
```

### Import Errors

**Error:** `ModuleNotFoundError`

**Solution:**
```bash
# Ensure virtual environment is activated
# Windows
venv\Scripts\activate

# Linux/Mac
source venv/bin/activate

# Reinstall dependencies
pip install -r requirements.txt
```

### CUDA/GPU Issues

**Error:** CUDA not available for Whisper

**Solution:**
Set `WHISPER_DEVICE=cpu` in `.env` file

### Vector Store Errors

**Error:** ChromaDB initialization failed

**Solution:**
```bash
# Ensure data directory exists
mkdir -p data/vector_db/chroma
```

### CORS Errors

**Error:** CORS policy blocked request

**Solution:**
Add your frontend URL to `API_CORS_ORIGINS` in `.env`:
```env
API_CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002
```

---

## Production Deployment

### Using Gunicorn (Linux/Mac)

```bash
pip install gunicorn
gunicorn backend.api.server:app --workers 4 --bind 0.0.0.0:8000
```

### Using Docker (Optional)

Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .
CMD ["uvicorn", "backend.api.server:app", "--host", "0.0.0.0", "--port", "8000"]
```

Build and run:
```bash
docker build -t research-backend .
docker run -p 8000:8000 --env-file .env research-backend
```

---

## Support

For issues or questions:
1. Check the [troubleshooting section](#troubleshooting)
2. Review backend logs in console
3. Verify all configurations in `.env`

---

## License

This project is for educational and portfolio purposes.
