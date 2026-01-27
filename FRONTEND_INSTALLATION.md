# Frontend Installation Guide

Academic Research Assistant Frontend - Next.js 14 application with React 18, TypeScript, and Tailwind CSS. Features voice recording, real-time chat, and research paper display.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the Development Server](#running-the-development-server)
- [Building for Production](#building-for-production)
- [Project Structure](#project-structure)
- [Components](#components)
- [Troubleshooting](#troubleshooting)

---

## Prerequisites

### Required Software

| Software | Version | Purpose |
|----------|---------|---------|
| Node.js | 18+ | Runtime |
| npm | 9+ | Package manager (comes with Node.js) |

### Check Node.js Version

```bash
node --version
npm --version
```

**Note:** If Node.js is not installed, download from [nodejs.org](https://nodejs.org)

---

## Installation

### 1. Navigate to Frontend Directory

```bash
cd D:\AI_Project\PycharmProjects\PycharmProjects\HomeWork9\frontend
```

### 2. Install Dependencies

```bash
npm install
```

#### Key Dependencies

| Category | Packages |
|----------|----------|
| **Framework** | next@14.0.4, react@18.2.0 |
| **Language** | typescript@5.3.3 |
| **HTTP Client** | axios@1.6.2 |
| **Icons** | lucide-react@0.294.0 |
| **Markdown** | marked@11.0.0 |
| **Sanitization** | dompurify@3.0.6 |
| **Styling** | tailwindcss@3.3.6 |

---

## Configuration

### Create Environment File

Create `.env.local` in the frontend directory:

```bash
cd frontend
echo "NEXT_PUBLIC_API_URL=http://localhost:8000" > .env.local
```

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `NEXT_PUBLIC_API_URL` | Backend API URL | `http://localhost:8000` |

**Important:** The frontend connects to the backend at this URL. Ensure the backend is running on the configured port.

### Complete `.env.local` Example

```env
# Backend API URL
NEXT_PUBLIC_API_URL=http://localhost:8000
```

---

## Running the Development Server

### Start Development Server

```bash
cd frontend
npm run dev
```

### Expected Output

```
   ▲ Next.js 14.0.4
   - Local:        http://localhost:3000
   - Environments: .env.local

 ✓ Ready in 2.7s
 ○ Compiling / ...
 ✓ Compiled / in 1774ms (501 modules)
```

### Access the Application

Open your browser to:

```
http://localhost:3000
```

**Note:** If port 3000 is in use, Next.js will automatically try the next available port (3001, 3002, etc.).

---

## Building for Production

### 1. Build the Application

```bash
cd frontend
npm run build
```

### Expected Build Output

```
   Linting and checking validity of types
   ✓ Passed
   Creating an optimized production build
   ✓ Compiled successfully
   ✓ Collecting page data
   ✓ Generating static pages (3/3)
   ✓ Finalizing page optimization

Route (app)                              Size
┌ ○ /                                   192 kB
├ ○ /_not-found                         192 kB
└ ○ /favicon.ico                        0 B

○ (Static)  prerendered as static content
```

### 2. Start Production Server

```bash
npm start
```

The production server will run on:

```
http://localhost:3000
```

---

## Project Structure

```
frontend/
├── src/
│   ├── app/                    # Next.js app directory
│   │   ├── layout.tsx          # Root layout with theme provider
│   │   ├── page.tsx            # Main page (chat interface)
│   │   └── globals.css         # Global styles
│   ├── components/             # React components
│   │   ├── AudioPlayer.tsx     # Audio playback component
│   │   ├── ChatInterface.tsx   # Chat message display
│   │   ├── ResultsDisplay.tsx  # Research papers sidebar
│   │   ├── ThemeSelector.tsx   # Dark/light mode toggle
│   │   └── VoiceRecorder.tsx   # Voice recording component
│   ├── services/               # API client
│   │   └── api.ts              # Backend API functions
│   ├── types.ts                # TypeScript type definitions
│   └── utils.ts                # Utility functions (optional)
├── public/                     # Static assets
├── .env.local                  # Environment variables (create this)
├── package.json                # Dependencies and scripts
├── tsconfig.json               # TypeScript configuration
├── tailwind.config.ts          # Tailwind CSS configuration
├── next.config.js              # Next.js configuration
└── postcss.config.js           # PostCSS configuration
```

---

## Components

### AudioPlayer
Displays audio controls for TTS responses with play/pause functionality.

### ChatInterface
Renders chat messages with markdown rendering and syntax highlighting. Supports user and assistant message types.

### ResultsDisplay
Sidebar component showing research papers with:
- Paper titles and authors
- ArXiv links
- Publication dates

### ThemeSelector
Toggle between dark and light themes with system preference detection.

### VoiceRecorder
Records audio from the microphone with:
- Visual recording indicator
- Start/stop controls
- Automatic audio format selection

---

## Troubleshooting

### Port Already in Use

**Error:** `Port 3000 is already in use`

**Solution:**
Next.js will automatically try the next available port (3001, 3002, etc.). Or manually stop the process:

**Windows:**
```bash
netstat -ano | findstr :3000
taskkill /PID <PID> /F
```

**Linux/Mac:**
```bash
lsof -ti:3000 | xargs kill -9
```

### Module Not Found Errors

**Error:** `Module not found: Can't resolve '...'`

**Solution:**
```bash
# Clear node_modules and reinstall
rm -rf node_modules package-lock.json
npm install
```

### API Connection Errors

**Error:** `Failed to fetch` or `ERR_CONNECTION_REFUSED`

**Solutions:**

1. **Verify backend is running:**
   ```bash
   curl http://localhost:8000/health
   ```

2. **Check `.env.local` API URL:**
   ```env
   NEXT_PUBLIC_API_URL=http://localhost:8000
   ```

3. **Verify CORS configuration** in backend `.env`:
   ```env
   API_CORS_ORIGINS=http://localhost:3000,http://localhost:3001,http://localhost:3002
   ```

### TypeScript Errors

**Error:** Type errors in editor or build

**Solution:**
```bash
# Rebuild TypeScript
rm -rf .next
npm run dev
```

### Build Failures

**Error:** Build fails with errors

**Solution:**
```bash
# Clear Next.js cache and rebuild
rm -rf .next
npm run build
```

### Hot Reload Not Working

**Issue:** Changes not reflecting in browser

**Solution:**
```bash
# Restart dev server
# Press Ctrl+C to stop, then:
npm run dev
```

### Voice Recording Not Working

**Issue:** Microphone permission denied

**Solution:**
1. Check browser permissions for microphone access
2. Ensure using HTTPS or localhost
3. Test microphone in browser settings

---

## Available Scripts

| Command | Description |
|---------|-------------|
| `npm run dev` | Start development server with hot reload |
| `npm run build` | Build for production |
| `npm start` | Start production server |
| `npm run lint` | Run ESLint linter |

---

## Browser Compatibility

| Browser | Version |
|---------|---------|
| Chrome | 90+ |
| Firefox | 88+ |
| Safari | 14+ |
| Edge | 90+ |

**Features requiring modern APIs:**
- Voice Recording (MediaRecorder API)
- Audio Playback (HTML5 Audio)

---

## Performance Tips

1. **Enable Production Mode**
   - Always build and use `npm start` for production

2. **Code Splitting**
   - Next.js automatically splits code by route

3. **Image Optimization**
   - Use Next.js Image component for images

4. **Bundle Analysis**
   ```bash
   npm install @next/bundle-analyzer
   npx next build --analyze
   ```

---

## Development Tips

### Hot Reload
Next.js provides fast refresh by default. Changes to components will reflect immediately.

### Debugging
Use browser DevTools (F12):
- **Console**: View JavaScript errors
- **Network**: Inspect API requests
- **React DevTools**: Inspect component tree

### TypeScript
The project uses strict TypeScript. Ensure all types are properly defined.

### Tailwind CSS
Utility classes are available throughout. Use the [Tailwind documentation](https://tailwindcss.com/docs) for reference.

---

## Deployment

### Vercel (Recommended)

1. Push code to GitHub
2. Import project in [Vercel](https://vercel.com)
3. Set environment variable: `NEXT_PUBLIC_API_URL`
4. Deploy

### Docker

Create `Dockerfile`:
```dockerfile
FROM node:18-alpine

WORKDIR /app
COPY package*.json ./
RUN npm ci --only=production

COPY . .
RUN npm run build

EXPOSE 3000
CMD ["npm", "start"]
```

Build and run:
```bash
docker build -t research-frontend .
docker run -p 3000:3000 research-frontend
```

### Static Export

For static hosting (Netlify, S3):
```bash
# In next.config.js, add:
# output: 'export'

npm run build
# Deploy 'out' directory
```

---

## Support

For issues:
1. Check [troubleshooting section](#troubleshooting)
2. Verify backend is running and accessible
3. Check browser console for errors
4. Ensure all dependencies are installed

---

## License

This project is for educational and portfolio purposes.
