"""Test page rendering endpoints to make sure they do not 500."""

from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from tests.test_debate_lifecycle import test_env as test_env  # noqa: F401


@pytest.mark.asyncio
async def test_dashboard_renders(test_env):
    app_instance = test_env
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        res = await ac.get("/")
        assert res.status_code == 200
        assert "Kontrol Paneli" in res.text

@pytest.mark.asyncio
async def test_live_arena_renders(test_env):
    app_instance = test_env
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        # Create a debate first
        debate_res = await ac.post(
            "/api/debates",
            json={
                "topic": "Test Debate",
                "agent_ids": [1, 2],
                "max_rounds": 2,
            },
        )
        debate_id = debate_res.json()["id"]

        # View the page
        res = await ac.get(f"/debate/{debate_id}")
        assert res.status_code == 200
        assert "Akademik Münazara Analiz Kokpiti" in res.text


