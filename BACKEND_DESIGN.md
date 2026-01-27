# Backend Software Design Document

## Academic Research Assistant - Backend Architecture

**Version:** 0.1.0
**Last Updated:** 2026-01-21

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [System Components](#system-components)
5. [Database Design](#database-design)
6. [RAG (Retrieval-Augmented Generation)](#rag-retrieval-augmented-generation)
7. [API Design](#api-design)
8. [LLM Integration](#llm-integration)
9. [Session Management](#session-management)
10. [Tools & Services](#tools--services)
11. [Data Flow](#data-flow)
12. [Security](#security)
13. [Scalability Considerations](#scalability-considerations)

---

## Overview

The backend is a FastAPI-based REST API server that provides intelligent academic research assistance through voice interaction, AI-powered search, and multi-paper summarization. The system integrates local LLM (Ollama), vector database for semantic search, and various external APIs (ArXiv, Notion).

### Key Features

- **Local LLM Support**: Ollama integration with qwen3:8b
- **Voice I/O**: Whisper ASR for transcription, TTS for speech synthesis
- **RAG System**: ChromaDB vector store for semantic context retrieval
- **Session Management**: Conversation history with persistence
- **Tool Calling**: Dynamic tool routing for ArXiv, web search, summarization
- **Notion Integration**: Automatic session synchronization

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                           Frontend (Next.js)                        │
│                       http://localhost:3000                         │
└────────────────────────────┬────────────────────────────────────────┘
                             │ HTTPS/WSS
                             ▼
┌─────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend                             │
│                       http://localhost:8000                         │
├─────────────────────────────────────────────────────────────────────┤
│  ┌───────────────┐  ┌───────────────┐  ┌─────────────────────────┐ │
│  │   API Routes  │  │   Middleware  │  │   CORS & Security       │ │
│  └───────┬───────┘  └───────┬───────┘  └─────────────────────────┘ │
│          │                  │                                            │
│  ┌───────▼──────────────────▼─────────────────────────────────────┐ │
│  │                    Service Layer                               │ │
│  │  ┌──────────┐  ┌──────────┐  ┌────────────┐  ┌────────────┐   │ │
│  │  │   LLM    │  │   ASR    │  │    TTS     │  │Vector Store│   │ │
│  │  │  Router  │  │ Whisper  │  │ Edge-TTS   │  │  ChromaDB  │   │ │
│  │  └────┬─────┘  └────┬─────┘  └─────┬──────┘  └─────┬──────┘   │ │
│  └───────┼─────────────┼───────────────┼───────────────┼──────────┘ │
│          │             │               │               │              │
│  ┌───────▼─────────────▼───────────────▼───────────────▼──────────┐ │
│  │                    Core Services                               │ │
│  │  ┌──────────────┐  ┌──────────────┐  ┌─────────────────────────┐│ │
│  │  │    Tools     │  │     Sessions │  │      Notion Sync        ││ │
│  │  │ ArXiv, Web   │  │   Manager    │  │                         ││ │
│  │  └──────────────┘  └──────────────┘  └─────────────────────────┘│ │
│  └──────────────────────────────────────────────────────────────────┘ │
└───────────────────────────┬──────────────────────────────────────────────┘
                            │
    ┌───────────────────────┼───────────────────────┐
    │                       │                       │
    ▼                       ▼                       ▼
┌─────────┐          ┌──────────┐          ┌──────────────┐
│ Ollama  │          │ChromaDB  │          │  File System │
│ qwen3:8b│          │Vector DB │          │  Sessions    │
│ :11434  │          │          │          │  Audio       │
└─────────┘          └──────────┘          └──────────────┘
```

### Request Flow

```
User Request → API → Session Manager → LLM Router → Tools → RAG → Response
                      │                        │
                      ▼                        ▼
                 Context                Ollama/ChromaDB
                 History
```

---

## Technology Stack

### Core Framework

| Component | Technology | Version | Purpose |
|-----------|-----------|---------|---------|
| **Web Framework** | FastAPI | 0.104.0+ | REST API server |
| **ASGI Server** | Uvicorn | 0.24.0+ | Async server |
| **Data Validation** | Pydantic | 2.5.0+ | Request/response models |

### AI/ML

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **LLM** | Ollama (qwen3:8b) | Local inference |
| **Embeddings** | sentence-transformers | Vector embeddings |
| **Vector DB** | ChromaDB | Semantic search |
| **ASR** | Whisper/faster-whisper | Speech-to-text |
| **TTS** | edge-tts, pyttsx3 | Text-to-speech |

### External APIs

| Service | Library | Purpose |
|---------|---------|---------|
| **ArXiv** | arxiv | Academic paper search |
| **Notion** | notion-client | Session sync |
| **Web Search** | tavily-python | Web search |

### Utilities

| Library | Purpose |
|---------|---------|
| python-dotenv | Configuration |
| httpx | Async HTTP client |
| aiohttp | Async requests |
| aiofiles | Async file I/O |
| feedparser | RSS/Atom parsing |

---

## System Components

### Directory Structure

```
backend/
├── api/
│   ├── server.py           # FastAPI app & endpoints
│   ├── models.py           # Pydantic models
│   └── routes/
│       └── status.py       # Status endpoint
├── core/
│   ├── config.py           # Settings management
│   ├── session_manager.py  # Conversation sessions
│   └── notion_sync.py      # Notion integration
├── services/
│   ├── llm_router.py       # LLM with tool calling
│   ├── asr.py              # Whisper ASR service
│   ├── tts.py              # TTS service
│   └── vector_store.py     # ChromaDB wrapper
├── tools/
│   ├── base.py             # Base tool class
│   ├── arxiv_tool.py       # ArXiv search
│   ├── web_search_tool.py  # Web search
│   └── summarizer.py       # Paper summarization
└── main.py                 # Entry point
```

### Component Descriptions

#### API Server (`api/server.py`)
- FastAPI application
- Route definitions
- Middleware configuration (CORS)
- Request/response handling

#### LLM Router (`services/llm_router.py`)
- Tool registration and routing
- Ollama client wrapper
- Message history management
- Streaming response support

#### Session Manager (`core/session_manager.py`)
- Conversation state tracking
- Message persistence (JSON files)
- Context window management
- Session lifecycle (create, get, delete)

#### Vector Store (`services/vector_store.py`)
- ChromaDB integration
- Document chunking
- Embedding generation
- Semantic search

---

## Database Design

### Storage Strategy

The backend uses **file-based storage** for sessions and **ChromaDB** for vector data.

```
data/
├── sessions/              # Conversation history
│   ├── {session_id}.json  # Individual session files
│   └── ...
├── audio/                 # Generated TTS audio
│   ├── {uuid}.mp3
│   └── ...
└── vector_db/
    └── chroma/            # ChromaDB persistent storage
        ├── chroma.sqlite3
        └── ...
```

### Session Storage Schema

**File:** `data/sessions/{session_id}.json`

```json
{
  "session_id": "ba834bd0-301e-4d93-8b15-06d44295d769",
  "created_at": "2026-01-20T21:54:56.170000",
  "updated_at": "2026-01-20T21:55:08.604000",
  "title": "Hello, what is 2+2?",
  "messages": [
    {
      "role": "user",
      "content": "Hello, what is 2+2?",
      "timestamp": "2026-01-20T21:54:58.040000",
      "metadata": {}
    },
    {
      "role": "assistant",
      "content": "The answer to 2 + 2 is 4...",
      "timestamp": "2026-01-20T21:55:08.604000",
      "metadata": {}
    }
  ],
  "metadata": {}
}
```

### Data Models

#### ChatMessage
```python
{
    "role": str,          # user, assistant, system, tool
    "content": str,       # Message text
    "timestamp": datetime,
    "metadata": dict      # Additional data
}
```

#### Session
```python
{
    "session_id": str,
    "created_at": datetime,
    "updated_at": datetime,
    "messages": List[ChatMessage],
    "title": Optional[str],
    "metadata": dict
}
```

### Vector Database Schema (ChromaDB)

**Collection:** `research_papers`

| Field | Type | Description |
|-------|------|-------------|
| `id` | str | Unique chunk ID |
| `documents` | str | Text chunk |
| `metadatas` | dict | Metadata (title, arxiv_id, authors) |
| `embeddings` | List[float] | Vector embeddings (384-dim) |

**Metadata Schema:**
```python
{
    "title": str,
    "arxiv_id": str,
    "authors": List[str],
    "published": str,
    "url": str,
    "chunk_index": int
}
```

---

## RAG (Retrieval-Augmented Generation)

### Architecture

```
Query → Embed → Search ChromaDB → Retrieve Context → Augment Prompt → LLM → Response
```

### Vector Store Service

**File:** `services/vector_store.py`

#### Key Methods

| Method | Description |
|--------|-------------|
| `add_paper_chunks()` | Chunk and index paper |
| `search()` | Semantic search |
| `get_context_for_query()` | Retrieve RAG context |

#### Chunking Strategy

```python
def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks.

    Args:
        text: Input text
        chunk_size: Max tokens per chunk
        overlap: Token overlap between chunks

    Returns:
        List of text chunks
    """
```

#### Embedding Model

- **Model:** `sentence-transformers/all-MiniLM-L6-v2`
- **Dimensions:** 384
- **Device:** CPU (configurable to CUDA)

#### RAG Integration

**In Chat Request:**

```python
# 1. Get RAG context from vector store
rag_context = vector_store.get_context_for_query(
    query=user_message,
    n_results=3,
    max_tokens=1500
)

# 2. Augment prompt
full_message = f"{user_message}\n\n---\n{rag_context}"

# 3. Send to LLM
response = await llm_router.chat(full_message)
```

### Indexing Pipeline

```python
# Index ArXiv paper
paper_data = {
    "title": "Attention Is All You Need",
    "arxiv_id": "1706.03762",
    "authors": ["Vaswani et al."],
    "summary": "...",
    "full_text": "Title\n\nAbstract..."
}

# Add to vector store
chunks = vector_store.add_paper_chunks(paper_data)
# Returns: number of chunks indexed
```

---

## API Design

### Base URL

```
http://localhost:8000
```

### Endpoints Overview

| Category | Endpoint | Method | Description |
|----------|----------|--------|-------------|
| **Health** | `/health` | GET | System health check |
| **Status** | `/api/status` | GET | Detailed status |
| **Chat** | `/api/chat` | POST | Main chat endpoint |
| **Audio** | `/api/transcribe` | POST | Transcribe audio |
| **Audio** | `/api/tts` | POST | Generate speech |
| **Audio** | `/api/audio/{filename}` | GET | Serve audio file |
| **Sessions** | `/api/sessions` | POST | Create session |
| **Sessions** | `/api/sessions/{id}` | GET | Get session |
| **Sessions** | `/api/sessions` | GET | List sessions |
| **Sessions** | `/api/sessions/{id}` | DELETE | Delete session |
| **Tools** | `/api/tools/arxiv` | POST | ArXiv search |
| **Tools** | `/api/tools/web-search` | POST | Web search |
| **Tools** | `/api/tools/summarize` | POST | Summarize papers |
| **Vector** | `/api/vector/search` | POST | Semantic search |
| **Vector** | `/api/vector/index` | POST | Index data |

### Request/Response Models

#### Chat Request

```python
POST /api/chat
Content-Type: application/json

{
    "message": "Search for papers on transformers",
    "session_id": "optional-uuid",
    "voice_input": false,
    "tools_to_use": ["search_arxiv"],
    "stream": false
}
```

#### Chat Response

```python
{
    "response": "I found 5 papers on transformers...",
    "session_id": "ba834bd0-301e-4d93-8b15-06d44295d769",
    "tool_calls": [
        {
            "name": "search_arxiv",
            "arguments": {"query": "transformers", "max_results": 5},
            "result": "Found 5 papers..."
        }
    ],
    "papers": [
        {
            "title": "Attention Is All You Need",
            "authors": ["Vaswani et al."],
            "arxiv_id": "1706.03762",
            "url": "https://arxiv.org/abs/1706.03762"
        }
    ],
    "audio_url": "/api/audio/uuid.mp3",
    "timestamp": "2026-01-21T03:00:00.000000"
}
```

---

## LLM Integration

### Ollama Client

**File:** `services/llm_router.py`

#### Configuration

```python
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b
```

#### OllamaClient Class

```python
class OllamaClient:
    def __init__(self, base_url: str, model: str):
        self.base_url = base_url
        self.model = model
        self.client = httpx.AsyncClient(timeout=120.0)

    async def chat(
        self,
        messages: List[Dict],
        tools: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        stream: bool = False
    ) -> Union[str, AsyncGenerator[str, None]]:
        """Send chat request to Ollama."""
```

### LLM Router

#### Tool Registration

```python
llm_router.register_tools([
    create_tool_definition(
        name="search_arxiv",
        description="Search academic papers on arXiv",
        parameters={...},
        async_function=search_arxiv
    ),
    create_tool_definition(
        name="search_web",
        description="Search the web for academic information",
        parameters={...},
        async_function=search_web
    ),
    create_tool_definition(
        name="summarize_papers",
        description="Summarize and synthesize research papers",
        parameters={...},
        async_function=summarize_papers
    )
])
```

#### System Prompt

```
You are an Academic Research Assistant, an AI that helps users search,
understand, and summarize academic content.

Capabilities:
1. ArXiv Search: Search academic papers across various fields
2. Web Search: Search for academic information online
3. Context Retrieval: Access knowledge base of indexed papers
4. Summarization: Synthesize and summarize research findings

Always provide:
- Clear, direct answers
- Proper citations (titles, authors, years)
- Links to sources when available
- Follow-up suggestions for deeper exploration
```

### Tool Calling Flow

```
User Query → Keyword Detection → Tool Selection → Tool Execution →
Result Formatting → Context Augmentation → LLM Response
```

---

## Session Management

### SessionManager Class

**File:** `core/session_manager.py`

#### Key Methods

| Method | Description |
|--------|-------------|
| `create_session()` | Create new session with UUID |
| `get_session()` | Retrieve session by ID |
| `add_message()` | Add message to session |
| `get_context_for_llm()` | Format context for LLM |
| `list_sessions()` | List recent sessions |
| `delete_session()` | Delete session |

#### Context Window Management

```python
def get_context_for_llm(
    session_id: str,
    max_messages: int = 10
) -> List[Dict[str, str]]:
    """
    Get conversation context formatted for LLM.
    Returns messages in OpenAI format:
    [{"role": "user", "content": "..."}, ...]
    """
```

#### Auto-Save

Sessions are automatically saved to disk after each message:

```python
# File: data/sessions/{session_id}.json
# Auto-saved on: add_message()
```

---

## Tools & Services

### ASR Service (Whisper)

**File:** `services/asr.py`

#### Configuration

```env
WHISPER_MODEL=base        # tiny, base, small, medium, large
WHISPER_DEVICE=cpu        # cpu, cuda
USE_FASTER_WHISPER=true
```

#### Supported Models

| Model | Size | Speed | Accuracy |
|-------|------|-------|----------|
| tiny | ~39MB | Fastest | Lower |
| base | ~74MB | Fast | Good |
| small | ~244MB | Medium | Better |
| medium | ~769MB | Slow | Very Good |
| large | ~1550MB | Slowest | Best |

### TTS Service

**File:** `services/tts.py`

#### Engines

| Engine | Offline | Voice Options |
|--------|---------|---------------|
| edge-tts | No | 100+ voices |
| pyttsx3 | Yes | System voices |
| openai | No | Premium voices |

#### Configuration

```env
TTS_ENGINE=edge
TTS_VOICE=en-US-AriaNeural
TTS_RATE=0
```

### ArXiv Tool

**File:** `tools/arxiv_tool.py`

#### Search Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| query | string | Search query |
| category | string | ArXiv category filter |
| max_results | int | 1-20 results |
| sort_by | string | relevance, lastUpdatedDate, submittedDate |

### Summarizer Tool

**File:** `tools/summarizer.py`

#### Features

- Multi-paper synthesis
- Comparison mode
- Focus area specification
- Citation preservation

---

## Data Flow

### Chat Request Flow

```
┌──────────────────────────────────────────────────────────────────┐
│ 1. Client sends POST /api/chat with message                     │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 2. API: Get or create session                                    │
│    - Retrieve existing session from session_id                   │
│    - OR create new session with UUID                             │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 3. API: Add user message to session history                      │
│    - session_manager.add_message(role="user", content=message)  │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 4. API: Get RAG context (if enabled)                             │
│    - vector_store.get_context_for_query(message)                 │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 5. API: Build augmented message                                  │
│    - Original message + RAG context + conversation history       │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 6. LLM Router: Process message                                   │
│    - Keyword detection for tool selection                        │
│    - OR pattern matching for USE_TOOL format                     │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 7. Tools: Execute if triggered                                   │
│    - ArXiv search, web search, summarization                     │
│    - Return results to LLM Router                                │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 8. LLM Router: Generate final response                           │
│    - Send to Ollama: ollama_client.chat(messages)                │
│    - Stream or wait for completion                               │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 9. API: Handle TTS (if requested)                                │
│    - synthesize_speech(response_text)                            │
│    - Save audio file to data/audio/                              │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 10. API: Add assistant response to session                        │
│     - session_manager.add_message(role="assistant", ...)         │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 11. API: Sync to Notion (if enabled)                             │
│     - Background task: notion_sync.sync_session()                │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
┌──────────────────────────────────────────────────────────────────┐
│ 12. API: Return ChatResponse                                     │
│     - response, session_id, papers, audio_url, tool_calls       │
└────────────────────────┬─────────────────────────────────────────┘
                         ▼
                    Client Response
```

### TTS Request Flow

```
Client Request → API → TTS Service → Edge-TTS API →
Audio File → Save to Disk → Return URL → Client Playback
```

---

## Security

### CORS Configuration

```python
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.API_CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

### Environment Variables

Sensitive data stored in `.env`:
- API keys (OpenAI, Notion, Tavily)
- Database paths
- Service configuration

### Input Validation

- Pydantic models for request validation
- File type validation for audio uploads
- Query parameter sanitization

### Rate Limiting

**Not implemented** - Consider adding for production:
```python
from slowapi import Limiter
limiter = Limiter(key_func=get_remote_address)

@app.post("/api/chat")
@limiter.limit("10/minute")
async def chat_completion(...):
    ...
```

---

## Scalability Considerations

### Current Limitations

| Component | Limitation | Solution |
|-----------|------------|----------|
| Session Storage | File-based | Migrate to PostgreSQL/MongoDB |
| Vector DB | Local ChromaDB | ChromaDB Server / Qdrant |
| LLM | Local Ollama | Distributed Ollama / OpenAI API |
| Audio Storage | Local disk | S3 / Cloud Storage |

### Recommended Improvements

1. **Database Migration**
   ```python
   # Replace JSON files with SQL database
   # SQLAlchemy models for Sessions, Messages
   ```

2. **Caching Layer**
   ```python
   from redis import Redis
   cache = Redis(host='localhost', port=6379)
   ```

3. **Queue System**
   ```python
   from celery import Celery
   # For background tasks: TTS, Notion sync
   ```

4. **Load Balancing**
   ```
   Nginx → Multiple Uvicorn workers
   ```

---

## Configuration

### Environment Variables

```env
# Local LLM (Ollama)
USE_LOCAL_LLM=true
LOCAL_LLM_BASE_URL=http://localhost:11434
LOCAL_LLM_MODEL=qwen3:8b

# OpenAI Fallback
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-4o

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002

# Vector DB
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
ENABLE_VECTOR_SEARCH=true

# ASR
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
USE_FASTER_WHISPER=true

# TTS
TTS_ENGINE=edge
TTS_VOICE=en-US-AriaNeural

# Sessions
MAX_CONTEXT_HISTORY=20
SESSION_TIMEOUT_MINUTES=60

# Features
ENABLE_WEB_SEARCH=true
ENABLE_NOTION_SYNC=true
```

---

## Appendix

### Dependencies Summary

```
# Core
fastapi>=0.104.0
uvicorn[standard]>=0.24.0
pydantic>=2.5.0
python-dotenv>=1.0.0

# AI/ML
transformers>=4.35.0
sentence-transformers>=2.2.0
torch>=2.1.0
chromadb>=0.4.18

# ASR/TTS
openai-whisper>=20231117
faster-whisper>=0.10.0
edge-tts>=6.1.9

# APIs
openai>=1.6.0
arxiv>=1.4.0
notion-client>=2.2.0
tavily-python>=0.3.0

# Async
httpx>=0.25.0
aiohttp>=3.9.0
aiofiles>=23.2.0
```

### API Documentation

Interactive API docs available at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

---

**Document Version:** 0.1.0
**Author:** Auto-generated from codebase analysis
**Date:** 2026-01-21
