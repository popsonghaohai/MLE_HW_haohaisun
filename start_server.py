"""
Simple startup script for the Academic Research Assistant API server.
"""
import sys
from pathlib import Path

# Add project root to path FIRST, before any imports
PROJECT_ROOT = Path(__file__).parent.resolve()
sys.path.insert(0, str(PROJECT_ROOT))

print(f"Project root: {PROJECT_ROOT}")
print(f"Python path includes: {PROJECT_ROOT}")

# Now import uvicorn and run
import uvicorn

if __name__ == "__main__":
    # Import the app directly (not as string)
    from backend.api.server import app

    print("=" * 60)
    print("  Academic Research Assistant - Backend API")
    print("=" * 60)
    print("  Starting server...")
    print("  API: http://0.0.0.0:8000")
    print("  Docs: http://0.0.0.0:8000/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )
