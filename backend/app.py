"""MedGraph AI production server (FastAPI + Uvicorn).

Local:
    cd backend && uvicorn app:app --reload --host 127.0.0.1 --port 8000

Render/Docker:
    uvicorn app:app --host 0.0.0.0 --port ${PORT:-8000}
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

BACKEND_DIR = Path(__file__).resolve().parent
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from medgraph.config import settings  # noqa: E402
from routes.api import router  # noqa: E402


def create_app() -> FastAPI:
    app = FastAPI(
        title=settings.app_name,
        version="1.0.0",
        description="Real LLM, FAISS RAG, and TigerGraph GraphRAG medical benchmark API.",
    )

    origins = [
        origin.strip()
        for origin in os.getenv("MEDGRAPH_CORS_ORIGINS", "*").split(",")
        if origin.strip()
    ]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins if origins != ["*"] else ["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(router)

    frontend_dir = settings.frontend_dir
    if frontend_dir.exists():
        static_names = {"styles.css", "app.js", "config.js"}

        @app.get("/")
        def index() -> FileResponse:
            return FileResponse(frontend_dir / "index.html")

        @app.get("/{filename}")
        def static_file(filename: str) -> FileResponse:
            if filename in static_names and (frontend_dir / filename).exists():
                media = "text/css" if filename.endswith(".css") else "application/javascript"
                return FileResponse(frontend_dir / filename, media_type=media)
            return FileResponse(frontend_dir / "index.html")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", "7860"))
    host = "0.0.0.0"
    uvicorn.run(app, host=host, port=port)
