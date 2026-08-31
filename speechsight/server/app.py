"""
SpeechSight AI Server Application.
FastAPI web application exposing AI endpoints and serving the interactive preview UI.
"""

import os
import logging
from pathlib import Path
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

from speechsight.server.api_routes import router as api_router
from speechsight.core.config import DEFAULT_CONFIG

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] [%(name)s] %(message)s"
)
logger = logging.getLogger("speechsight.server")

app = FastAPI(
    title="SpeechSight AI API",
    description="Visual Intelligence for Human Speech - Audio-Visual Speech Recognition Pipeline",
    version=DEFAULT_CONFIG.app_version,
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS for browser preview & external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include API Router
app.include_router(api_router)

# Mount static directory for UI and media assets
static_dir = Path(__file__).resolve().parent / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
def serve_index():
    """Serves the main application UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return {
        "app": "SpeechSight AI",
        "status": "online",
        "docs": "/docs",
        "api": "/api/health"
    }


def main():
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    host = os.getenv("HOST", "0.0.0.0")
    logger.info(f"Starting SpeechSight AI server on {host}:{port}")
    uvicorn.run("speechsight.server.app:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
