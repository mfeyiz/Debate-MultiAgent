"""FastAPI application factory for pure async operations."""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from app.config import Config
from app.database import async_session_factory, init_db
from app.models import Agent
from app.services.debate_service import DebateService
from app.services.fact_check_service import FactCheckService


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifecycle manager for the FastAPI application.

    Initializes the async database and seeds default agents on startup.
    """
    await init_db()

    # Seed default agents if the database is empty
    async with async_session_factory() as db:
        from sqlalchemy import select
        stmt = select(Agent)
        result = await db.execute(stmt)
        if not result.scalars().first():
            defaults = [
                Agent(
                    name="Agent Alpha",
                    model_name="openai/gpt-4o-mini",
                    role="proponent",
                    system_prompt=(
                        "You are a rigorous proponent. Your job is to construct well-supported "
                        "claims using evidence, data, and logical reasoning. Always ground your "
                        "arguments in facts. Be concise and specific."
                    ),
                    temperature=0.7,
                ),
                Agent(
                    name="Agent Beta",
                    model_name="openai/gpt-4o-mini",
                    role="opponent",
                    system_prompt=(
                        "You are a critical opponent. Your job is to identify weaknesses, "
                        "contradictions, and logical gaps in the opposing view. Attack claims "
                        "with precision and cite counter-evidence when possible. Be concise and specific."
                    ),
                    temperature=0.7,
                ),
            ]
            db.add_all(defaults)
            await db.commit()

    yield


def create_app() -> FastAPI:
    """Create and configure the FastAPI application instance."""
    app = FastAPI(title="Logos", version="0.1.0", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=Config.CORS_ALLOW_ORIGIN_REGEX,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )

    app.state.debate_svc = DebateService()
    app.state.fact_svc = FactCheckService()

    @app.get("/health", include_in_schema=False)
    async def health() -> dict[str, str]:
        """Lightweight load balancer health check."""
        return {"status": "ok"}

    # Register modular routes
    from app.routers.api import router as api_router
    from app.routers.pages import router as pages_router
    from app.routers.stream import router as stream_router

    app.include_router(pages_router)
    app.include_router(api_router)
    app.include_router(stream_router)

    return app
