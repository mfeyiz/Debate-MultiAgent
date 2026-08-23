"""Application configuration."""

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent


class Config:
    """Base configuration."""

    SECRET_KEY = os.environ.get("SECRET_KEY", "dev-secret-key-change-in-prod")
    SQLALCHEMY_DATABASE_URI = os.environ.get("DATABASE_URL") or f"sqlite:///{BASE_DIR / 'debate_platform.db'}"
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # OpenRouter / LLM settings
    OPENROUTER_API_KEY = os.environ.get("OPENROUTER_API_KEY", "")
    OPENROUTER_BASE_URL = os.environ.get(
        "OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"
    )
    DEFAULT_MODEL = os.environ.get("DEFAULT_MODEL", "openai/gpt-4o-mini")

    # Fact-check search settings
    TAVILY_API_KEY = os.environ.get("TAVILY_API_KEY", "")
    TAVILY_BASE_URL = os.environ.get("TAVILY_BASE_URL", "https://api.tavily.com")
    GOOGLE_FACTCHECK_API_KEY = os.environ.get("GOOGLE_FACTCHECK_API_KEY", "")
    TCMB_EVDS_API_KEY = os.environ.get("TCMB_EVDS_API_KEY", "")
    TUIK_API_KEY = os.environ.get("TUIK_API_KEY", "")
    FACT_CHECK_MAX_CLAIMS = int(os.environ.get("FACT_CHECK_MAX_CLAIMS", "3"))
    FACT_CHECK_SEARCH_RESULTS = int(os.environ.get("FACT_CHECK_SEARCH_RESULTS", "2"))
    FACT_CHECK_MAX_QUERIES_PER_CLAIM = int(os.environ.get("FACT_CHECK_MAX_QUERIES_PER_CLAIM", "1"))
    FACT_CHECK_ARCHIVE_DOMAINS = [
        domain.strip()
        for domain in os.environ.get(
            "FACT_CHECK_ARCHIVE_DOMAINS",
            "teyit.org,dogrulukpayi.com,factcheck.org,politifact.com,snopes.com",
        ).split(",")
        if domain.strip()
    ]
    FACT_CHECK_TRUSTED_DOMAINS = [
        domain.strip()
        for domain in os.environ.get(
            "FACT_CHECK_TRUSTED_DOMAINS",
            "tuik.gov.tr,tcmb.gov.tr,resmigazete.gov.tr,who.int,worldbank.org,oecd.org",
        ).split(",")
        if domain.strip()
    ]
    # Turkish fact-check archive endpoints
    TURKISH_ARCHIVE_URLS = {
        "teyit": "https://teyit.org",
        "dogrulukpayi": "https://dogrulukpayi.com",
    }
    # Public data source endpoints
    TUIK_API_BASE = os.environ.get("TUIK_API_BASE", "https://data.tuik.gov.tr")
    TCMB_API_BASE = os.environ.get("TCMB_API_BASE", "https://evds2.tcmb.gov.tr")
    RESMI_GAZETE_BASE = os.environ.get("RESMI_GAZETE_BASE", "https://www.resmigazete.gov.tr")
    # Claim cache settings
    CLAIM_CACHE_TTL_DAYS = int(os.environ.get("CLAIM_CACHE_TTL_DAYS", "7"))
    CLAIM_SIMILARITY_THRESHOLD = float(os.environ.get("CLAIM_SIMILARITY_THRESHOLD", "0.92"))
    # Source credibility
    ENABLE_CREDIBILITY_SCORING = os.environ.get("ENABLE_CREDIBILITY_SCORING", "true").lower() == "true"
    CORS_ALLOW_ORIGIN_REGEX = os.environ.get(
        "CORS_ALLOW_ORIGIN_REGEX",
        r"^(chrome-extension://.*|http://localhost(:\d+)?|http://127\.0\.0\.1(:\d+)?)$",
    )

    # ModernBERT pipeline thresholds
    MODEL_BASE_DIR = os.environ.get("MODEL_BASE_DIR", str(BASE_DIR / "models"))
    COMPONENT_MODEL_DIR = os.environ.get(
        "COMPONENT_MODEL_DIR",
        str(Path(MODEL_BASE_DIR) / "component_classifier/final"),
    )
    RELATION_MODEL_DIR = os.environ.get(
        "RELATION_MODEL_DIR",
        str(Path(MODEL_BASE_DIR) / "relation_classifier/final"),
    )
    MODEL_GCS_BUCKET = os.environ.get("MODEL_GCS_BUCKET", "")
    MODEL_GCS_PREFIX = os.environ.get("MODEL_GCS_PREFIX", "modernbert")
    COMPONENT_MODEL_GCS_PREFIX = os.environ.get(
        "COMPONENT_MODEL_GCS_PREFIX",
        f"{MODEL_GCS_PREFIX.rstrip('/')}/component_classifier/final",
    )
    RELATION_MODEL_GCS_PREFIX = os.environ.get(
        "RELATION_MODEL_GCS_PREFIX",
        f"{MODEL_GCS_PREFIX.rstrip('/')}/relation_classifier/final",
    )
    MODEL_REQUIRED_FILES = tuple(
        file_name.strip()
        for file_name in os.environ.get(
            "MODEL_REQUIRED_FILES",
            "config.json,model.safetensors,tokenizer.json,tokenizer_config.json",
        ).split(",")
        if file_name.strip()
    )
    EVIDENCE_STRENGTH_THRESHOLD = float(os.environ.get("EVIDENCE_STRENGTH_THRESHOLD", "0.65"))
    PIPELINE_EVAL_INTERVAL = int(os.environ.get("PIPELINE_EVAL_INTERVAL", "2"))

    # Debate settings
    MAX_DEBATE_ROUNDS = int(os.environ.get("MAX_DEBATE_ROUNDS", "5"))

    # Server / infrastructure
    PORT = int(os.environ.get("PORT", "8080"))
    GOOGLE_CLOUD_PROJECT = os.environ.get("GOOGLE_CLOUD_PROJECT") or os.environ.get("GCP_PROJECT", "")
    SQLITE_BUSY_TIMEOUT_MS = int(os.environ.get("SQLITE_BUSY_TIMEOUT_MS", "30000"))
    # Cap PyTorch intra-op threads so a single BERT inference cannot peg every
    # CPU core and starve the async event loop (causes slow pages + pool timeouts).
    TORCH_NUM_THREADS = int(os.environ.get("TORCH_NUM_THREADS", "2"))
    # Async DB connection pool sizing.
    DB_POOL_SIZE = int(os.environ.get("DB_POOL_SIZE", "20"))
    DB_MAX_OVERFLOW = int(os.environ.get("DB_MAX_OVERFLOW", "40"))

    # Redis (for multi-pod SSE)
    REDIS_URL = os.environ.get("REDIS_URL", "")
    REDIS_HOST = os.environ.get("REDIS_HOST", "localhost")
    REDIS_PORT = int(os.environ.get("REDIS_PORT", "6379"))
    REDIS_DB = int(os.environ.get("REDIS_DB", "0"))
