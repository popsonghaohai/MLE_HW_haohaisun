"""
Main entry point for the Academic Research Assistant.
Run this script to start both backend and frontend servers.
"""
import os
import sys
import subprocess
import threading
import time
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))


def check_env_file():
    """Check if .env file exists and has required variables."""
    env_file = PROJECT_ROOT / ".env"
    if not env_file.exists():
        print("⚠️  Warning: .env file not found!")
        print("   Creating a template .env file...")
        template = """# OpenAI API (Required)
OPENAI_API_KEY=

# Notion (Optional - for session persistence)
NOTION_API_KEY=
NOTION_DATABASE_ID=
ENABLE_NOTION_SYNC=true

# Tavily (Optional - for web search)
TAVILY_API_KEY=

# API Server
API_HOST=0.0.0.0
API_PORT=8000
API_CORS_ORIGINS=http://localhost:3000

# Whisper ASR
WHISPER_MODEL=base
WHISPER_DEVICE=cpu
USE_FASTER_WHISPER=true

# TTS
TTS_ENGINE=edge
TTS_VOICE=en-US-AriaNeural
TTS_RATE=0

# Vector DB
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2

# Session Management
MAX_CONTEXT_HISTORY=20
SESSION_TIMEOUT_MINUTES=60

# Feature Flags
ENABLE_VECTOR_SEARCH=true
ENABLE_WEB_SEARCH=true
"""
        with open(env_file, 'w') as f:
            f.write(template)
        print("   ✓ Created .env file with defaults")
        print("   ⚠️  Please add your OPENAI_API_KEY to .env before running!")
        return False

    # Check for required API key
    with open(env_file) as f:
        content = f.read()
        if "OPENAI_API_KEY=" not in content or not content.split("OPENAI_API_KEY=")[1].strip().replace('"', '').replace("'", ""):
            print("⚠️  Warning: OPENAI_API_KEY not set in .env!")
            print("   Please add your OpenAI API key to .env before running.")
            return False

    return True


def install_backend_dependencies():
    """Install Python dependencies if needed."""
    try:
        import fastapi
        import openai
        import chromadb
        print("✓ Backend dependencies installed")
        return True
    except ImportError as e:
        print(f"⚠️  Missing dependency: {e}")
        print("   Installing dependencies...")
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install",
                "-r", str(PROJECT_ROOT / "requirements.txt")
            ])
            print("✓ Dependencies installed successfully")
            return True
        except subprocess.CalledProcessError:
            print("✗ Failed to install dependencies")
            print("  Please run: pip install -r requirements.txt")
            return False


def start_backend():
    """Start the FastAPI backend server."""
    print("\n🚀 Starting backend server...")
    os.chdir(PROJECT_ROOT)

    # Set environment variables
    os.environ["PYTHONPATH"] = str(PROJECT_ROOT)

    try:
        subprocess.run([
            sys.executable, "-m", "uvicorn",
            "backend.api.server:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload"
        ])
    except KeyboardInterrupt:
        print("\n⏹️  Backend server stopped")


def start_frontend():
    """Start the Next.js frontend development server."""
    frontend_dir = PROJECT_ROOT / "frontend"

    if not frontend_dir.exists():
        print("⚠️  Frontend directory not found!")
        print("   Skipping frontend startup...")
        return

    print("\n🎨 Starting frontend server...")
    os.chdir(frontend_dir)

    try:
        # Check if node_modules exists
        if not (frontend_dir / "node_modules").exists():
            print("   Installing frontend dependencies...")
            subprocess.check_call(["npm", "install"])

        subprocess.run(["npm", "run", "dev"])
    except KeyboardInterrupt:
        print("\n⏹️  Frontend server stopped")


def main():
    """Main entry point."""
    print("=" * 60)
    print("  Academic Research Assistant")
    print("  Voice-enabled AI for academic research")
    print("=" * 60)

    # Check environment
    if not check_env_file():
        print("\n⚠️  Please configure your .env file first!")
        print("   Then run this script again.")
        return

    # Install dependencies
    if not install_backend_dependencies():
        print("\n⚠️  Please install dependencies first:")
        print("   pip install -r requirements.txt")
        return

    print("\n✓ Configuration check passed!")

    # Ask what to start
    print("\nWhat would you like to start?")
    print("  1. Backend only")
    print("  2. Frontend only")
    print("  3. Both (default)")
    print("  4. Exit")

    choice = input("\nEnter choice [1-4]: ").strip() or "3"

    if choice == "1":
        start_backend()
    elif choice == "2":
        start_frontend()
    elif choice == "3":
        # Start both in separate threads
        backend_thread = threading.Thread(target=start_backend, daemon=True)
        backend_thread.start()

        time.sleep(2)  # Give backend time to start

        frontend_thread = threading.Thread(target=start_frontend, daemon=True)
        frontend_thread.start()

        print("\n" + "=" * 60)
        print("  Both servers started!")
        print("  Backend:  http://localhost:8000")
        print("  Frontend: http://localhost:3000")
        print("  API Docs: http://localhost:8000/docs")
        print("=" * 60)
        print("\nPress Ctrl+C to stop both servers\n")

        # Keep main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("\n⏹️  Shutting down...")
    else:
        print("Exiting...")


if __name__ == "__main__":
    main()
