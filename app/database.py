"""Asynchronous database configuration using SQLAlchemy 2.0."""

import os
from collections.abc import AsyncGenerator

from sqlalchemy.ext.asyncio import (
    AsyncAttrs,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy import text
from sqlalchemy.orm import DeclarativeBase

from app.config import Config

# Convert sync sqlite/postgres URIs to async variations
database_url = Config.SQLALCHEMY_DATABASE_URI
if database_url.startswith("sqlite:///"):
    database_url = database_url.replace("sqlite:///", "sqlite+aiosqlite:///")
elif database_url.startswith("sqlite://"):
    database_url = database_url.replace("sqlite://", "sqlite+aiosqlite://")
elif database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql+asyncpg://")
elif database_url.startswith("postgresql://"):
    database_url = database_url.replace("postgresql://", "postgresql+asyncpg://")

# Configuration tuning for SQLite concurrency
connect_args = {}
if "sqlite" in database_url:
    connect_args["timeout"] = 30

engine = create_async_engine(
    database_url,
    connect_args=connect_args,
    echo=False,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    expire_on_commit=False,
)


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy database models."""

    pass


async def get_db() -> AsyncGenerator[AsyncSession, None]:
    """FastAPI Dependency for request-scoped database sessions."""
    async with async_session_factory() as session:
        try:
            yield session
        finally:
            await session.close()


async def init_db() -> None:
    """Initialize database and create tables if they do not exist."""
    async with engine.begin() as conn:
        from app import models  # noqa: F401
        await conn.run_sync(Base.metadata.create_all)
        if "sqlite" in database_url:
            result = await conn.execute(text("PRAGMA table_info(argument_components)"))
            columns = {row[1] for row in result.fetchall()}
            if "topic_relation_type" not in columns:
                await conn.execute(text("ALTER TABLE argument_components ADD COLUMN topic_relation_type VARCHAR(20)"))
            if "topic_relation_confidence" not in columns:
                await conn.execute(text("ALTER TABLE argument_components ADD COLUMN topic_relation_confidence FLOAT"))
            if "topic_relation_probabilities_json" not in columns:
                await conn.execute(
                    text("ALTER TABLE argument_components ADD COLUMN topic_relation_probabilities_json TEXT DEFAULT '{}'")
                )
