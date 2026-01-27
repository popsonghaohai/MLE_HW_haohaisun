# Frontend Software Design Document

## Academic Research Assistant - Frontend Architecture

**Version:** 0.1.0
**Last Updated:** 2026-01-21

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Technology Stack](#technology-stack)
4. [Project Structure](#project-structure)
5. [Components](#components)
6. [State Management](#state-management)
7. [API Integration](#api-integration)
8. [UI/UX Design](#uiux-design)
9. [Routing](#routing)
10. [Theming](#theming)
11. [TypeScript Types](#typescript-types)
12. [Performance Optimization](#performance-optimization)
13. [Browser Compatibility](#browser-compatibility)
14. [Deployment](#deployment)

---

## Overview

The frontend is a modern Next.js 14 application built with React 18, TypeScript, and Tailwind CSS. It provides an intuitive interface for voice-enabled academic research assistance with real-time chat, voice recording, and research paper display.

### Key Features

- **Real-time Chat**: Streaming responses with markdown rendering
- **Voice Recording**: Browser-based audio recording with MediaRecorder API
- **Audio Playback**: TTS response playback with seek controls
- **Paper Display**: Sidebar for research papers with metadata
- **Dark Theme**: Modern dark mode with system preference detection
- **Responsive Design**: Mobile-friendly layout

---

## Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                         Browser (Client)                            │
│                                                                    │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                      Next.js 14 App                          │  │
│  │  ┌─────────────────────────────────────────────────────────┐│  │
│  │  │                    React 18 Components                  ││  │
│  │  │  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌─────────┐ ││  │
│  │  │  │   Page   │  │  Layout  │  │Components│  │  Types  │ ││  │
│  │  │  └────┬─────┘  └────┬─────┘  └────┬─────┘  └────┬────┘ ││  │
│  │  └───────┼─────────────┼─────────────┼─────────────┼──────┘│  │
│  │          │             │             │             │         │  │
│  │  ┌───────▼─────────────▼─────────────▼─────────────▼──────┐ │  │
│  │  │                   Services Layer                        │ │  │
│  │  │  ┌────────────────────────────────────────────────────┐│ │  │
│  │  │  │              API Client (api.ts)                   ││ │  │
│  │  │  └────────────────────────────────────────────────────┘│ │  │
│  │  └──────────────────────────────────────────────────────────┘│  │
│  │                                                             │  │
│  │  ┌─────────────────────────────────────────────────────────┐│  │
│  │  │                   Browser APIs                          ││  │
│  │  │  ┌───────────┐  ┌───────────┐  ┌─────────────────────┐││  │
│  │  │  │ MediaRecorder│  │Audio      │  │LocalStorage        │││  │
│  │  │  │ (Voice Input)│  │ (Playback) │  │(Session Persist)   │││  │
│  │  │  └─────────────┘  └───────────┘  └─────────────────────┘││  │
│  │  └──────────────────────────────────────────────────────────┘│  │
│  └──────────────────────────────────────────────────────────────┘  │
│                              │                                      │
│                              ▼                                      │
│  ┌───────────────────────────────────────────────────────────────┐  │
│  │                    HTTP/JSON API                             │  │
│  │                   http://localhost:8000                      │  │
│  └───────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow

```
User Input → Component State → API Client → Backend API → Response
                                    │
                                    ▼
                              Update State
                                    │
                                    ▼
                              Re-render UI
```

---

## Technology Stack

### Core Framework

| Technology | Version | Purpose |
|------------|---------|---------|
| **Next.js** | 14.0.4 | React framework with App Router |
| **React** | 18.2.0 | UI library |
| **TypeScript** | 5.3.3 | Type safety |
| **Node.js** | 18+ | Runtime |

### UI & Styling

| Technology | Version | Purpose |
|------------|---------|---------|
| **Tailwind CSS** | 3.3.6 | Utility-first CSS |
| **PostCSS** | 8.4.32 | CSS processing |
| **Autoprefixer** | 10.4.16 | CSS vendor prefixes |

### Icons & Utilities

| Technology | Version | Purpose |
|------------|---------|---------|
| **Lucide React** | 0.294.0 | Icon library |
| **clsx** | 2.0.0 | Conditional className |
| **tailwind-merge** | 2.1.0 | Merge Tailwind classes |

### Content Rendering

| Technology | Version | Purpose |
|------------|---------|---------|
| **Marked** | 11.0.0 | Markdown parser |
| **DOMPurify** | 3.0.6 | XSS sanitization |

### HTTP Client

| Technology | Version | Purpose |
|------------|---------|---------|
| **Axios** | 1.6.2 | HTTP requests (optional, uses fetch) |

### Development Tools

| Technology | Version | Purpose |
|------------|---------|---------|
| **ESLint** | 8.56.0 | Code linting |
| **@types/*** | Various | TypeScript definitions |

---

## Project Structure

### Directory Layout

```
frontend/
├── src/
│   ├── app/                    # Next.js App Router
│   │   ├── layout.tsx          # Root layout
│   │   ├── page.tsx            # Home page (main chat)
│   │   └── globals.css         # Global styles
│   ├── components/             # React components
│   │   ├── AudioPlayer.tsx     # TTS audio playback
│   │   ├── ChatInterface.tsx   # Chat message display
│   │   ├── ResultsDisplay.tsx  # Papers sidebar
│   │   ├── ThemeSelector.tsx   # Dark/light toggle
│   │   └── VoiceRecorder.tsx   # Voice recording
│   ├── services/               # API client
│   │   └── api.ts              # Backend API functions
│   └── types.ts                # TypeScript definitions
├── public/                     # Static assets
├── .env.local                  # Environment variables
├── package.json                # Dependencies
├── tsconfig.json               # TypeScript config
├── tailwind.config.ts          # Tailwind config
├── next.config.js              # Next.js config
└── postcss.config.js           # PostCSS config
```

### File Purposes

| File | Purpose |
|------|---------|
| `app/layout.tsx` | Root layout with metadata and fonts |
| `app/page.tsx` | Main application page with chat interface |
| `app/globals.css` | Global styles, CSS variables, animations |
| `components/ChatInterface.tsx` | Message display with markdown |
| `components/VoiceRecorder.tsx` | Audio recording controls |
| `components/AudioPlayer.tsx` | TTS playback with seek bar |
| `components/ResultsDisplay.tsx` | Research papers sidebar |
| `services/api.ts` | API client with typed functions |
| `types.ts` | Shared TypeScript interfaces |

---

## Components

### Component Hierarchy

```
Page (app/page.tsx)
├── Header
│   ├── Logo & Title
│   ├── ThemeSelector
│   └── Mute Toggle
├── ChatArea
│   ├── ChatInterface
│   │   ├── Welcome Screen (when empty)
│   │   └── Message List
│   │       ├── User Messages
│   │       └── Assistant Messages (Markdown)
│   └── Loading Indicator
├── InputArea
│   ├── Text Input
│   ├── Send Button
│   └── VoiceRecorder
└── ResultsDisplay (when papers exist)
    └── Paper Cards
```

### Page Component (`app/page.tsx`)

**Main container** for the application. Manages:

| State | Type | Purpose |
|-------|------|---------|
| `messages` | `Message[]` | Chat history |
| `input` | `string` | Text input value |
| `isLoading` | `boolean` | Loading state |
| `isRecording` | `boolean` | Recording state |
| `sessionId` | `string \| null` | Session persistence |
| `papers` | `PaperReference[]` | Found papers |
| `audioUrl` | `string \| null` | TTS audio URL |
| `muted` | `boolean` | Voice response toggle |

### ChatInterface Component

**Renders chat messages** with markdown rendering and syntax highlighting.

**Features:**
- User/assistant message styling
- Markdown content parsing
- Code block highlighting
- Auto-scroll to bottom
- Welcome screen when empty

**Props:**
```typescript
interface ChatInterfaceProps {
  messages: Message[];
  isLoading: boolean;
}
```

### VoiceRecorder Component

**Records audio** from microphone using MediaRecorder API.

**Features:**
- Permission state detection
- Recording timer display
- Visual pulse animation
- Auto-cleanup on stop

**Props:**
```typescript
interface VoiceRecorderProps {
  isRecording: boolean;
  onStartRecording: () => void;
  onStopRecording: (audioBlob: Blob) => void;
  onFinish?: () => void;
}
```

**Browser API:**
```typescript
const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
const mediaRecorder = new MediaRecorder(stream, {
  mimeType: 'audio/webm;codecs=opus',
});
```

### AudioPlayer Component

**Plays TTS responses** with seek controls.

**Features:**
- Play/pause toggle
- Progress bar with seek
- Time display
- Auto-play attempt (with fallback)
- Fixed bottom positioning

**Props:**
```typescript
interface AudioPlayerProps {
  audioUrl: string;
  onClose: () => void;
}
```

### ResultsDisplay Component

**Shows research papers** in a sidebar.

**Features:**
- Collapsible sidebar
- Paper metadata display
- External link to ArXiv
- Truncated author lists
- Publication dates

**Props:**
```typescript
interface ResultsDisplayProps {
  papers: PaperReference[];
  onClose: () => void;
}
```

---

## State Management

### Approach: Component State (useState)

The application uses **local component state** rather than a global state management library (Redux, Zustand). This is appropriate for the current scope.

### State Flow

```
Parent (Page)           Child Components
    │
    ├─ messages ──────→ ChatInterface
    │
    ├─ isLoading ─────→ ChatInterface (loading dots)
    │
    ├─ isRecording ────→ VoiceRecorder
    │
    ├─ papers ─────────→ ResultsDisplay
    │
    └─ audioUrl ───────→ AudioPlayer
```

### Session Persistence

**LocalStorage** is used to persist the session ID:

```typescript
// Save session
localStorage.setItem('research_session_id', sessionId);

// Restore session
const savedSession = localStorage.getItem('research_session_id');
```

---

## API Integration

### API Client (`services/api.ts`)

**Base URL Configuration:**
```typescript
const API_BASE_URL = process.env.NEXT_PUBLIC_API_URL || 'http://localhost:8000';
```

### Request Function

Generic API request handler with error handling:

```typescript
async function apiRequest<T>(
  endpoint: string,
  options: RequestInit = {}
): Promise<T> {
  const url = `${API_BASE_URL}${endpoint}`;

  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options
  });

  if (!response.ok) {
    const error = await response.json();
    throw new Error(error.detail || `HTTP ${response.status}`);
  }

  return await response.json();
}
```

### API Functions

| Function | Method | Endpoint |
|----------|--------|----------|
| `chatCompletion()` | POST | `/api/chat` |
| `transcribeAudio()` | POST | `/api/transcribe` |
| `textToSpeech()` | POST | `/api/tts` |
| `createSession()` | POST | `/api/sessions` |
| `getSession()` | GET | `/api/sessions/{id}` |
| `listSessions()` | GET | `/api/sessions` |
| `deleteSession()` | DELETE | `/api/sessions/{id}` |
| `searchArxiv()` | POST | `/api/tools/arxiv` |
| `webSearch()` | POST | `/api/tools/web-search` |
| `summarizePapers()` | POST | `/api/tools/summarize` |
| `vectorSearch()` | POST | `/api/vector/search` |
| `healthCheck()` | GET | `/health` |

### Chat Completion Example

```typescript
const response = await chatCompletion({
  message: "Search for papers on transformers",
  session_id: sessionId || undefined,
  voice_input: !muted,
});

// Response
{
  response: string;
  session_id: string;
  tool_calls: ToolCall[];
  papers: PaperReference[];
  audio_url?: string;
}
```

### Next.js Rewrites

API rewrites configured in `next.config.js`:

```javascript
async rewrites() {
  return [
    {
      source: '/api/:path*',
      destination: `${process.env.NEXT_PUBLIC_API_URL}/api/:path*`,
    },
  ];
}
```

This allows API calls to use relative URLs.

---

## UI/UX Design

### Design Principles

1. **Clean Interface**: Minimal distractions, focus on content
2. **Dark Theme Default**: Easy on eyes for long sessions
3. **Responsive**: Works on desktop and mobile
4. **Visual Feedback**: Loading states, animations, hover effects

### Color Palette

| Role | Variable | Value |
|------|----------|-------|
| Background | `--background` | `#0a0a0a` |
| Foreground | `--foreground` | `#ededed` |
| Card | `--card` | `#141414` |
| Primary | `--primary` | `#3b82f6` (blue) |
| Secondary | `--secondary` | `#262626` |
| Muted | `--muted-foreground` | `#a3a3a3` |
| Border | `--border` | `#262626` |

### Typography

- **Font Family**: Inter (Google Fonts)
- **Fallback**: System UI fonts
- **Line Height**: 1.7 for markdown content

### Animations

| Animation | Duration | Purpose |
|-----------|----------|---------|
| `slide-in` | 0.3s | Message appearance |
| `pulse-ring` | 1.5s | Recording indicator |
| `blink` | 1.4s | Loading dots |

### Layout

```
┌─────────────────────────────────────────────────────────────┐
│  Header (Logo | Title | Theme | Mute)                      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────┬─────────────────────────┐ │
│  │                             │                         │ │
│  │      Chat Interface         │   Results Display       │ │
│  │   - Messages                │   (when papers)         │ │
│  │   - Loading Indicator       │   - Paper Cards         │ │
│  │                             │                         │ │
│  └─────────────────────────────┴─────────────────────────┘ │
├─────────────────────────────────────────────────────────────┤
│  Input Area (Text | Mic | Send)                              │
└─────────────────────────────────────────────────────────────┘
│  Audio Player (when TTS enabled)                             │
└─────────────────────────────────────────────────────────────┘
```

---

## Routing

### App Router (Next.js 14)

The application uses **Next.js App Router** with the `app/` directory.

### Route Structure

```
app/
├── layout.tsx       → Root layout (/)
├── page.tsx         → Home page (/)
└── globals.css      → Global styles
```

### Root Layout (`layout.tsx`)

```typescript
export const metadata: Metadata = {
  title: 'Academic Research Assistant',
  description: 'Voice-enabled AI assistant for academic research',
};

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en">
      <body className={inter.className}>{children}</body>
    </html>
  );
}
```

### Single Page Application

Currently, the app is a **single-page application** with no additional routes. Future expansion could include:

- `/settings` - Settings page
- `/sessions` - Session history
- `/papers/{id}` - Paper details

---

## Theming

### Dark Theme (Default)

The application uses a **dark-first** design with CSS variables for theming.

### CSS Variables

```css
:root {
  --background: #0a0a0a;
  --foreground: #ededed;
  --primary: #3b82f6;
  --secondary: #262626;
  --muted-foreground: #a3a3a3;
  /* ... */
}
```

### Tailwind Integration

```css
@tailwind base;
@tailwind components;
@tailwind utilities;
```

Custom classes use CSS variables:
```css
.bg-background { background-color: var(--background); }
.text-foreground { color: var(--foreground); }
```

### Future: Light Theme

To add a light theme, use a class-based approach:

```css
:root {
  --background: #ffffff;
  --foreground: #0a0a0a;
}

.dark {
  --background: #0a0a0a;
  --foreground: #ededed;
}
```

### Theme Detection

System preference detection (future implementation):

```typescript
const [theme, setTheme] = useState<'light' | 'dark'>(
  window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light'
);
```

---

## TypeScript Types

### Type Definitions (`types.ts`)

```typescript
// Chat message
export interface Message {
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: Date;
}

// Research paper
export interface PaperReference {
  title: string;
  authors: string[];
  arxiv_id?: string;
  url?: string;
  published?: string;
  abstract?: string;
}

// Chat request
export interface ChatRequest {
  message: string;
  session_id?: string;
  voice_input?: boolean;
  tools_to_use?: string[];
  stream?: boolean;
}

// Chat response
export interface ChatResponse {
  response: string;
  session_id: string;
  tool_calls: ToolCall[];
  papers: PaperReference[];
  audio_url?: string;
  timestamp: string;
}

// Tool call
export interface ToolCall {
  name: string;
  arguments: Record<string, unknown>;
  result?: string;
}

// Transcription
export interface TranscriptionResponse {
  text: string;
  language: string;
  confidence: number;
  model: string;
}

// Session
export interface Session {
  session_id: string;
  title?: string;
  created_at: string;
  updated_at: string;
  messages: Message[];
  message_count: number;
  metadata?: Record<string, unknown>;
}

// API wrapper
export interface ApiResponse<T> {
  data?: T;
  error?: string;
  detail?: string;
}
```

---

## Performance Optimization

### Next.js Optimizations

1. **Automatic Code Splitting**: Each page is a separate bundle
2. **Image Optimization**: Use `next/image` for images (future)
3. **Font Optimization**: Google Fonts with `next/font`
4. **Static Generation**: Pages pre-rendered at build time

### Best Practices

| Practice | Implementation |
|----------|----------------|
| **Lazy Loading** | Components load on demand |
| **Memoization** | `useMemo` for expensive calculations |
| **Callback Optimization** | `useCallback` for event handlers |
| **Bundle Size** | Tree-shaking, minification |

### Build Output

```
Route (app)                              Size
┌ ○ /                                   192 kB
├ ○ /_not-found                         192 kB
└ ○ /favicon.ico                        0 B
```

---

## Browser Compatibility

### Supported Browsers

| Browser | Version | Notes |
|---------|---------|-------|
| Chrome | 90+ | Full support |
| Firefox | 88+ | Full support |
| Safari | 14+ | Full support |
| Edge | 90+ | Full support |

### Required Browser APIs

| API | Purpose | Fallback |
|-----|---------|----------|
| MediaRecorder | Voice recording | None (required) |
| Audio | TTS playback | None (required) |
| LocalStorage | Session persistence | None (required) |
| Fetch | API requests | Polyfill available |

### Feature Detection

```typescript
// MediaRecorder support
const mediaRecorderSupported = 'MediaRecorder' in window;

// LocalStorage support
const storageSupported = typeof localStorage !== 'undefined';
```

---

## Deployment

### Development

```bash
npm run dev
# http://localhost:3000
```

### Production Build

```bash
npm run build
npm start
# http://localhost:3000
```

### Environment Variables

**File:** `.env.local`

```env
NEXT_PUBLIC_API_URL=http://localhost:8000
```

### Deployment Platforms

#### Vercel (Recommended)

1. Connect GitHub repository
2. Configure environment variable: `NEXT_PUBLIC_API_URL`
3. Deploy automatically on push

#### Docker

```dockerfile
FROM node:18-alpine AS build
WORKDIR /app
COPY package*.json ./
RUN npm ci
COPY . .
RUN npm run build

FROM node:18-alpine AS run
WORKDIR /app
COPY --from=build /app/.next ./.next
COPY --from=build /app/node_modules ./node_modules
COPY --from=build /app/package.json ./package.json
EXPOSE 3000
CMD ["npm", "start"]
```

#### Static Export

For static hosting (requires configuration):

```javascript
// next.config.js
module.exports = {
  output: 'export',
  images: { unoptimized: true }
};
```

### CORS Configuration

**Backend must allow frontend origin:**

```env
# Backend .env
API_CORS_ORIGINS=http://localhost:3000,https://yourdomain.com
```

---

## Appendix

### Package.json Scripts

```json
{
  "scripts": {
    "dev": "next dev",
    "build": "next build",
    "start": "next start",
    "lint": "next lint"
  }
}
```

### Tailwind Configuration

```typescript
// tailwind.config.ts
import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './src/pages/**/*.{js,ts,jsx,tsx,mdx}',
    './src/components/**/*.{js,ts,jsx,tsx,mdx}',
    './src/app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {},
  },
  plugins: [],
}
export default config
```

### TypeScript Configuration

```json
// tsconfig.json
{
  "compilerOptions": {
    "target": "ES2017",
    "lib": ["dom", "dom.iterable", "esnext"],
    "allowJs": true,
    "skipLibCheck": true,
    "strict": true,
    "noEmit": true,
    "esModuleInterop": true,
    "module": "esnext",
    "moduleResolution": "bundler",
    "resolveJsonModule": true,
    "isolatedModules": true,
    "jsx": "preserve",
    "incremental": true,
    "plugins": [{ "name": "next" }],
    "paths": {
      "@/*": ["./src/*"]
    }
  }
}
```

---

## Future Enhancements

### Potential Features

1. **Multi-Session Support**
   - Session list sidebar
   - Session switching
   - Session renaming

2. **Advanced Voice Features**
   - Wake word detection
   - Continuous listening
   - Voice command shortcuts

3. **Enhanced Paper View**
   - PDF preview
   - Highlight extraction
   - Citation management

4. **User Settings**
   - Theme customization
   - Voice preferences
   - API key management

5. **Mobile App**
   - React Native version
   - Offline support
   - Push notifications

---

**Document Version:** 0.1.0
**Author:** Auto-generated from codebase analysis
**Date:** 2026-01-21
