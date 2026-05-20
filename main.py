"""ASGI application entry point."""

import os

import uvicorn

from app import create_app
from app.config import Config

app = create_app()

if __name__ == "__main__":
    port = int(os.environ.get("PORT", Config.PORT))
    reload = os.environ.get("FASTAPI_RELOAD", "false").lower() in ("1", "true", "yes")
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=reload)
