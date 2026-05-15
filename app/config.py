"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
    SQLALCHEMY_DATABASE_URI = os.environ.get(
        "DATABASE_URL", f"sqlite:///{BASE_DIR / 'debate_platform.db'}"
    )
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # OpenRouter / LLM settings
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "openai/gpt-4o-mini")

    # ModernBERT pipeline thresholds
    EVIDENCE_STRENGTH_THRESHOLD = float(os.environ.get("EVIDENCE_STRENGTH_THRESHOLD", "0.65"))
    PIPELINE_EVAL_INTERVAL = int(os.environ.get("PIPELINE_EVAL_INTERVAL", "2"))

    # Debate settings
    MAX_DEBATE_ROUNDS = int(os.environ.get("MAX_DEBATE_ROUNDS", "5"))

    # Server / infrastructure
    PORT = int(os.environ.get("PORT", "5000"))

    # Redis (for multi-pod SSE)
    REDIS_URL = os.environ.get("REDIS_URL", "")
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
