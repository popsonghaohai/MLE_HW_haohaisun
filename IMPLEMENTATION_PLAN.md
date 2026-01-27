# Voice-Enabled Academic Research Assistant - Implementation Plan

## Project Overview
Build a demo-ready full-stack voice assistant that can:
- Take voice queries via web interface
- Transcribe speech using Whisper ASR
- Route queries using GPT-4 function calling
- Search academic content (ArXiv, web)
- Summarize findings with context
- Return voice responses via TTS
- Sync all sessions to Notion

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                              FRONTEND (Web UI)                              │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐   │
│  │ Voice Input  │  │ Chat Display │  │ Results View │  │ Audio Player │   │
│  │  (Recorder)  │  │  (Messages)  │  │  (Papers)    │  │   (TTS)      │   │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘   │
│         └──────────────────┴──────────────────┴──────────────────┘           │
│                                  │                                          │
│                         HTTP/WebSocket API                                   │
└──────────────────────────────────┼──────────────────────────────────────────┘
                                   │
┌──────────────────────────────────┼──────────────────────────────────────────┐
│                     BACKEND API SERVER (FastAPI)                             │
│                                  │                                           │
│  ┌───────────────────────────────┼───────────────────────────────────────┐ │
│  │                       Main API Endpoints                               │ │
│  │  POST /api/chat           - Main chat endpoint with streaming          │ │
│  │  POST /api/transcribe     - Upload audio for transcription             │ │
│  │  GET  /api/sessions/:id   - Get session history                        │ │
│  │  POST /api/sessions       - Create new session                         │ │
│  └───────────────────────────────┼───────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────┴───────────────────────────────────────┐ │
│  │                    Agent Orchestrator Module                          │ │
│  │  - Session management with context history                            │ │
│  │  - LLM function calling orchestration                                 │ │
│  │  - Tool execution and response aggregation                            │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
│                                  │                                           │
│  ┌───────────────────────────────┴───────────────────────────────────────┐ │
│  │                        Tool Modules                                   │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │ │
│  │  │    ASR      │ │  LLM Router │ │    TTS      │ │   Notion    │     │ │
│  │  │  (Whisper)  │ │ (GPT-4 FC)  │ │  (OpenAI)   │ │   Sync      │     │ │
│  │  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘ └──────┬──────┘     │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌─────────────┐     │ │
│  │  │ ArXiv Tool  │ │ Web Search  │ │ Summarizer  │ │Vector Store │     │ │
│  │  │             │ │   (Tavily)  │ │             │ │ (ChromaDB)  │     │ │
│  │  └─────────────┘ └─────────────┘ └─────────────┘ └─────────────┘     │ │
│  └──────────────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## Implementation Phases

### Phase 1: Backend Foundation & Tool Integration
**Priority: HIGH - Core functionality**

#### 1.1 Project Setup
- [ ] Update requirements.txt with all dependencies
- [ ] Create modular project structure
- [ ] Set up environment configuration

#### 1.2 Core AI Services
- [ ] **Whisper ASR Module** (`asr.py`)
  - Audio file handling (webm, mp3, wav)
  - Transcription with OpenAI Whisper API
  - Fallback to local Whisper model

- [ ] **LLM Router with Function Calling** (`llm_router.py`)
  - OpenAI GPT-4 integration
  - Function calling schema definition
  - Tool decision logic
  - Response streaming support

- [ ] **TTS Module** (`tts.py`)
  - OpenAI TTS API integration
  - Audio generation for responses
  - Voice selection options

#### 1.3 Academic Tools
- [ ] **ArXiv Search Tool** (`tools/arxiv_tool.py`)
  - ArXiv API queries
  - Paper metadata extraction
  - Abstract/full-text retrieval
  - Integration with existing RAG pipeline

- [ ] **Web Search Tool** (`tools/web_search.py`)
  - Tavily API or similar academic search
  - Result parsing and ranking
  - Source citation handling

- [ ] **Summarization Tool** (`tools/summarizer.py`)
  - Multi-paper synthesis
  - Key point extraction
  - Citation preservation

#### 1.4 Vector Database & Context
- [ ] **ChromaDB Integration** (`vector_store.py`)
  - Embedding generation (OpenAI or local)
  - Document indexing from RAG data
  - Semantic search/retrieval
  - Context window management

### Phase 2: Session Management & Persistence
**Priority: HIGH - User experience**

#### 2.1 Session Management (`session_manager.py`)
- [ ] Session ID generation and tracking
- [ ] Conversation history storage
- [ ] Context window management (sliding window)
- [ ] Multi-turn conversation support

#### 2.2 Notion Integration (`notion_sync.py`)
- [ ] Notion client setup
- [ ] Database page creation
- [ ] Conversation logging
- [ ] Summary and citation storage
- [ ] Session metadata tracking

### Phase 3: Backend API Server
**Priority: HIGH - Full-stack connectivity**

#### 3.1 FastAPI Server (`server.py`)
- [ ] FastAPI app setup
- [ ] CORS configuration
- [ ] Request/response models

#### 3.2 API Endpoints
- [ ] `POST /api/chat` - Main chat with streaming
- [ ] `POST /api/transcribe` - Audio transcription
- [ ] `POST /api/sessions` - Create session
- [ ] `GET /api/sessions/{id}` - Get session history
- [ ] `GET /api/audio/{id}` - Get generated audio

#### 3.3 Background Tasks
- [ ] Async tool execution
- [ ] Notion sync queue
- [ ] Vector DB indexing

### Phase 4: Frontend Web Interface
**Priority: MEDIUM - Demo readiness**

#### 4.1 Project Setup
- [ ] Create React/Next.js project or vanilla HTML/JS
- [ ] Set up build pipeline
- [ ] Configure API base URL

#### 4.2 Core Components
- [ ] **VoiceRecorder Component**
  - MediaRecorder API integration
  - Audio format selection (webm/mp3)
  - Recording controls (start/stop/pause)
  - Visual feedback (waveform)

- [ ] **ChatInterface Component**
  - Message list with user/assistant bubbles
  - Streaming response rendering
  - Markdown rendering for responses
  - Citation rendering

- [ ] **ResultsDisplay Component**
  - Paper cards with metadata
  - Abstract expansion
  - Link to full paper
  - Save to Notion indicator

- [ ] **AudioPlayer Component**
  - TTS response playback
  - Playback controls
  - Auto-play on response

#### 4.3 Styling & UX
- [ ] Responsive design
- [ ] Dark/light theme
- [ ] Loading states
- [ ] Error handling UI

### Phase 5: Integration & Testing
**Priority: MEDIUM - Demo polish**

#### 5.1 End-to-End Integration
- [ ] Full voice-to-voice flow testing
- [ ] Session continuity testing
- [ ] Notion sync verification

#### 5.2 Error Handling
- [ ] Graceful degradation (no mic, API failures)
- [ ] Retry logic
- [ ] User-friendly error messages

#### 5.3 Performance Optimization
- [ ] Response caching
- [ ] Audio compression
- [ ] Lazy loading

---

## Project Structure (Proposed)

```
HomeWork9/
├── backend/
│   ├── api/
│   │   ├── __init__.py
│   │   ├── server.py              # FastAPI application
│   │   ├── models.py              # Pydantic models
│   │   └── routes/
│   │       ├── chat.py            # Chat endpoints
│   │       ├── sessions.py        # Session endpoints
│   │       └── audio.py           # Audio endpoints
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py              # Configuration management
│   │   ├── session_manager.py     # Session & context management
│   │   └── notion_sync.py         # Notion integration
│   ├── services/
│   │   ├── __init__.py
│   │   ├── asr.py                 # Whisper ASR service
│   │   ├── tts.py                 # TTS service
│   │   ├── llm_router.py          # LLM with function calling
│   │   └── vector_store.py        # ChromaDB integration
│   ├── tools/
│   │   ├── __init__.py
│   │   ├── base.py                # Base tool class
│   │   ├── arxiv_tool.py          # ArXiv search
│   │   ├── web_search_tool.py     # Web search
│   │   └── summarizer.py          # Summarization
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── audio_utils.py         # Audio processing utilities
│   │   └── text_utils.py          # Text processing utilities
│   └── main.py                    # Entry point
├── frontend/
│   ├── src/
│   │   ├── components/
│   │   │   ├── VoiceRecorder.jsx  # Voice recording component
│   │   │   ├── ChatInterface.jsx  # Chat display
│   │   │   ├── ResultsDisplay.jsx # Search results
│   │   │   └── AudioPlayer.jsx    # TTS playback
│   │   ├── services/
│   │   │   └── api.js             # API client
│   │   ├── styles/
│   │   │   └── main.css
│   │   └── App.jsx
│   ├── public/
│   └── package.json
├── static/                         # For vanilla JS option
├── arxiv_data_rag/                 # Existing RAG data
├── .env                            # Environment variables
├── requirements.txt                # Python dependencies
└── README.md
```

---

## Function Calling Schema

```json
{
  "tools": [
    {
      "type": "function",
      "function": {
        "name": "search_arxiv",
        "description": "Search academic papers on arXiv",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"},
            "category": {"type": "string", "description": "ArXiv category (e.g., cs.CL, cs.AI)"},
            "max_results": {"type": "integer", "default": 10}
          },
          "required": ["query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "search_web",
        "description": "Search the web for academic information",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Search query"},
            "max_results": {"type": "integer", "default": 5}
          },
          "required": ["query"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "summarize_papers",
        "description": "Summarize multiple research papers",
        "parameters": {
          "type": "object",
          "properties": {
            "papers": {
              "type": "array",
              "items": {"type": "object"},
              "description": "List of papers to summarize"
            },
            "focus": {
              "type": "string",
              "description": "Specific aspect to focus on"
            }
          },
          "required": ["papers"]
        }
      }
    },
    {
      "type": "function",
      "function": {
        "name": "retrieve_context",
        "description": "Retrieve relevant context from vector database",
        "parameters": {
          "type": "object",
          "properties": {
            "query": {"type": "string", "description": "Query for retrieval"},
            "n_results": {"type": "integer", "default": 5}
          },
          "required": ["query"]
        }
      }
    }
  ]
}
```

---

## Key Technologies

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend Framework | FastAPI | High-performance async API |
| ASR | OpenAI Whisper API | Speech-to-text transcription |
| LLM | OpenAI GPT-4o | Intent routing & function calling |
| TTS | OpenAI TTS API | Text-to-speech synthesis |
| Vector DB | ChromaDB | Semantic search & context retrieval |
| Academic Search | ArXiv API | Paper metadata & abstracts |
| Web Search | Tavily API | Academic web search |
| Persistence | Notion API | Session storage & summaries |
| Frontend | React/Next.js | Modern web UI |
| Audio | MediaRecorder API | Browser-based recording |

---

## Success Criteria

1. **Voice Pipeline**: User speaks → transcription → tool execution → TTS response
2. **Academic Search**: Returns relevant papers with proper citations
3. **Context Awareness**: Maintains conversation context across turns
4. **Notion Sync**: Every session automatically logged to Notion
5. **Demo Ready**: Polished UI that works in browser

---

## Implementation Order (Recommended)

1. **Week 1**: Backend foundation (ASR, LLM Router, TTS)
2. **Week 2**: Tools implementation (ArXiv, Web Search, Summarizer)
3. **Week 3**: Vector DB & Session Management + Notion
4. **Week 4**: FastAPI server & API endpoints
5. **Week 5**: Frontend development
6. **Week 6**: Integration, testing, demo polish

---

## Notes

- This project demonstrates full-stack AI orchestration
- Function calling showcases production-ready AI agent patterns
- Notion integration adds enterprise-grade persistence
- Voice interface makes it accessible and impressive for demos
