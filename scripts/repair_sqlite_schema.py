"""Repair local SQLite schema drift for development databases.

This is intentionally small and idempotent. It reuses the app's configured
database URL, so it repairs the same SQLite file the app will open on startup.
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.database import init_db


async def main() -> None:
    """Create missing tables and add compatible SQLite columns."""
    await init_db()


if __name__ == "__main__":
    asyncio.run(main())
