# Academic Research Assistant

A full-stack AI-powered voice-enabled research assistant for academic paper search, summarization, and intelligent conversation. Built with Next.js 14, FastAPI, and local LLM support via Ollama.

![Version](https://img.shields.io/badge/version-0.1.0-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Python](https://img.shields.io/badge/python-3.10+-blue)
![Node](https://img.shields.io/badge/node-18+-green)

## Features

- **Voice Interaction** - Record voice queries with automatic transcription using Whisper ASR
- **Local LLM** - Run `qwen3:8b` locally via Ollama for privacy and cost efficiency
- **Academic Search** - Search ArXiv and web for research papers
- **Intelligent Summarization** - Synthesize multiple papers with citation preservation
- **Vector Database** - ChromaDB for semantic context retrieval (RAG)
- **Session Management** - Persistent conversation history
- **Text-to-Speech** - Voice responses with playback controls
- **Modern UI** - Beautiful dark theme interface with Tailwind CSS

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                         Frontend (Next.js)                      │
│                     http://localhost:3000                       │
├─────────────────────────────────────────────────────────────────┤
│  Voice Input → Chat Interface → API Client → Backend API       │
│       ↓              ↓                ↓            ↓            │
│   Recorder     Messages Display    Fetch     FastAPI Server    │
│       ↓              ↓                ↓            ↓            │
│   Audio Blob    Markdown Render  Response    LLM Router        │
│                                           ↓                    │
│                                    ┌──────┴──────┐            │
│                                    │             │            │
│                         ┌──────────▼─┐   ┌──────▼────┐       │
│                         │   Ollama    │   │ ChromaDB  │       │
│                         │  (qwen3:8b) │   │ Vector DB │       │
│                         └────────────┘   └───────────┘       │
└─────────────────────────────────────────────────────────────────┘
```

## Tech Stack

### Frontend

| Technology | Purpose |
|------------|---------|
| Next.js 14 | React framework with App Router |
| React 18 | UI library |
| TypeScript | Type safety |
| Tailwind CSS | Styling |
| Lucide React | Icons |
| Marked | Markdown rendering |
| DOMPurify | XSS protection |

### Backend

| Technology | Purpose |
|------------|---------|
| FastAPI | REST API server |
| Python 3.10+ | Runtime |
| Ollama | Local LLM inference |
| ChromaDB | Vector database |
| Whisper | Speech-to-text |
| Edge-TTS | Text-to-speech |
| sentence-transformers | Embeddings |

## Quick Start

### Prerequisites

- **Python 3.10+**
- **Node.js 18+**
- **Ollama** (for local LLM)

### Installation

1. **Clone the repository**
   ```bash
   cd D:\AI_Project\PycharmProjects\PycharmProjects\HomeWork9
   ```

2. **Install Backend**
   ```bash
   # Create virtual environment
   python -m venv venv

   # Activate (Windows)
   venv\Scripts\activate
   # Activate (Linux/Mac)
   source venv/bin/activate

   # Install dependencies
   pip install -r requirements.txt
   ```

3. **Install Frontend**
   ```bash
   cd frontend
   npm install
   ```

4. **Configure Ollama**
   ```bash
   # Pull the model
   ollama pull qwen3:8b

   # Start Ollama (usually starts automatically)
   ollama serve
   ```

5. **Configure Environment**

   **Backend `.env`:**
   ```env
   USE_LOCAL_LLM=true
   LOCAL_LLM_BASE_URL=http://localhost:11434
   LOCAL_LLM_MODEL=qwen3:8b

   API_HOST=0.0.0.0
   API_PORT=8000
   API_CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002

   ENABLE_VECTOR_SEARCH=true
   ```

   **Frontend `.env.local`:**
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

6. **Start the Application**

   **Terminal 1 - Backend:**
   ```bash
   python -m backend.main
   # Runs on http://localhost:8000
   ```

   **Terminal 2 - Frontend:**
   ```bash
   cd frontend
   npm run dev
   # Runs on http://localhost:3000
   ```

7. **Open in Browser**
   ```
   http://localhost:3000
   ```

## Usage

### Voice Search

1. Click the microphone button
2. Speak your query (e.g., "Search for papers on transformers")
3. Click again to stop recording
4. View the transcription and results

### Text Search

1. Type your query in the input field
2. Press Enter or click Send
3. View results in the chat interface

### Research Papers

- Papers appear in the right sidebar when found
- Click "View Paper" to open on ArXiv
- View authors, publication date, and abstract

### Voice Responses

- Toggle voice responses with the speaker icon
- Audio player appears at the bottom when enabled
- Use seek bar to navigate through responses

## Project Structure

```
HomeWork9/
├── backend/                    # Python FastAPI backend
│   ├── api/                   # API endpoints
│   │   ├── server.py          # Main FastAPI app
│   │   └── models.py          # Pydantic models
│   ├── core/                  # Configuration, sessions
│   │   ├── config.py          # Settings management
│   │   └── session_manager.py # Conversation state
│   ├── services/              # Business logic
│   │   ├── llm_router.py      # LLM with tool calling
│   │   ├── asr.py             # Whisper ASR
│   │   ├── tts.py             # Text-to-speech
│   │   └── vector_store.py    # ChromaDB wrapper
│   ├── tools/                 # API tools
│   │   ├── arxiv_tool.py      # ArXiv search
│   │   ├── web_search_tool.py # Web search
│   │   └── summarizer.py      # Paper summarization
│   └── main.py                # Entry point
│
├── frontend/                   # Next.js frontend
│   ├── src/
│   │   ├── app/               # Next.js App Router
│   │   │   ├── layout.tsx     # Root layout
│   │   │   ├── page.tsx       # Main page
│   │   │   └── globals.css    # Global styles
│   │   ├── components/        # React components
│   │   │   ├── ChatInterface.tsx
│   │   │   ├── VoiceRecorder.tsx
│   │   │   ├── AudioPlayer.tsx
│   │   │   └── ResultsDisplay.tsx
│   │   ├── services/          # API client
│   │   │   └── api.ts
│   │   └── types.ts           # TypeScript types
│   └── package.json
│
├── data/                       # Data directory
│   ├── sessions/              # Conversation history
│   ├── audio/                 # Generated TTS audio
│   └── vector_db/             # ChromaDB storage
│
├── .env                        # Backend environment
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── BACKEND_INSTALLATION.md     # Backend setup guide
├── FRONTEND_INSTALLATION.md    # Frontend setup guide
├── BACKEND_DESIGN.md           # Backend design document
└── FRONTEND_DESIGN.md          # Frontend design document
```

## API Documentation

Interactive API docs available at:
- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

### Main Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/chat` | POST | Send message, get response |
| `/api/transcribe` | POST | Transcribe audio |
| `/api/tts` | POST | Generate speech |
| `/api/sessions` | GET/POST | Manage sessions |
| `/api/tools/arxiv` | POST | Search ArXiv |
| `/api/vector/search` | POST | Semantic search |
| `/health` | GET | Health check |

## Configuration

### Backend Environment Variables

```env
# Local LLM (Ollama)
USE_LOCAL_LLM=true
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=http://localhost:3000,http://localhost:3001

# Vector Database
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENABLE_VECTOR_SEARCH=true

# ASR (Whisper)
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
USE_FASTER_WHISPER=true

# TTS
TTS_ENGINE=edge
TTS_VOICE=en-US-AriaNeural

# Session Management
MAX_CONTEXT_HISTORY=20
SESSION_TIMEOUT_MINUTES=60
```

### Frontend Environment Variables

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Development

### Backend Development

```bash
# Activate virtual environment
venv\Scripts\activate  # Windows
source venv/bin/activate  # Linux/Mac

# Run with auto-reload
uvicorn backend.api.server:app --reload --host 0.0.0.0 --port 8000
```

### Frontend Development

```bash
cd frontend

# Development server
npm run dev

# Production build
npm run build

# Start production
npm start

# Lint
npm run lint
```

## Screenshots

### Main Interface

```
┌─────────────────────────────────────────────────────────────┐
│  🎓 Academic Research Assistant          [🌙] [🔊]          │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────┐ ┌─────────────────────┐│
│  │                                 │ │ 📄 Found Papers    ││
│  │ 💬 Welcome!                     │ │                     ││
│  │                                 │ │ Attention Is All... ││
│  │ Try asking about:               │ │                     ││
│  │ • Recent papers on LLMs         │ │ BERT: Pre-training...││
│  │ • Climate change research       │ │                     ││
│  │ • Quantum computing              │ └─────────────────────┘│
│  │                                 │                         │
│  └─────────────────────────────────┘                         │
├─────────────────────────────────────────────────────────────┤
│  [Type your query...]                    [🎤] [Send]        │
└─────────────────────────────────────────────────────────────┘
```

## Troubleshooting

### Backend Issues

**Problem:** `Address already in use` (port 8000)
```bash
# Windows
netstat -ano | findstr :8000
taskkill /PID <PID> /F

# Linux/Mac
lsof -ti:8000 | xargs kill -9
```

**Problem:** Ollama connection failed
```bash
# Check Ollama is running
ollama list

# Restart Ollama
ollama serve
```

**Problem:** Model not found
```bash
ollama pull qwen3:8b
```

### Frontend Issues

**Problem:** Port 3000 in use
- Next.js will automatically try port 3001, 3002, etc.

**Problem:** API connection refused
- Verify backend is running on port 8000
- Check `NEXT_PUBLIC_API_URL` in `.env.local`

**Problem:** Microphone not working
- Check browser permissions
- Ensure using HTTPS or localhost
- Test at https://webcammictest.com

## Documentation

| Document | Description |
|----------|-------------|
| [BACKEND_INSTALLATION.md](BACKEND_INSTALLATION.md) | Backend setup guide |
| [FRONTEND_INSTALLATION.md](FRONTEND_INSTALLATION.md) | Frontend setup guide |
| [BACKEND_DESIGN.md](BACKEND_DESIGN.md) | Backend architecture |
| [FRONTEND_DESIGN.md](FRONTEND_DESIGN.md) | Frontend architecture |

## License

This project is for educational and portfolio purposes.

## Credits

Built with:
- [Next.js](https://nextjs.org/) - React framework
- [FastAPI](https://fastapi.tiangolo.com/) - Python web framework
- [Ollama](https://ollama.ai/) - Local LLM inference
- [ChromaDB](https://www.trychroma.com/) - Vector database
- [Tailwind CSS](https://tailwindcss.com/) - Styling

---

**Version:** 0.1.0
**Last Updated:** 2026-01-21
