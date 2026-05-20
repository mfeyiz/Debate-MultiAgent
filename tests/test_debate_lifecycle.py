"""Comprehensive debate lifecycle tests covering creation through resolution."""

from __future__ import annotations

import tempfile
from pathlib import Path
import pytest
from httpx import ASGITransport, AsyncClient

from app import create_app
import app.database
from app.config import Config
from app.models import Agent, Debate, DebateParticipant, Message, MessageVersion
from app.services.bert_service import ArgumentComponent
from app.services.debate_service import DebateService


class FakeBert:
    """Deterministic ModernBERT stand-in for lifecycle tests."""

    def extract_components(self, text: str, default_type: str) -> list[ArgumentComponent]:
        return [
            ArgumentComponent(
                text=text[:80] if len(text) > 80 else text,
                component_type=default_type,
                start_idx=0,
                end_idx=min(len(text), 80),
                confidence=0.85,
            )
        ]

    def classify_relation(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> tuple[str, float, dict[str, float]]:
        return "support", 0.80, {"support": 0.80, "attack": 0.10, "neutral": 0.10}

    def analyze(self, source_text: str, target_text: str, source_type: str, target_type: str):
        from app.services.bert_service import PipelineResult, Relation
        comp = ArgumentComponent(
            text=source_text[:80] if len(source_text) > 80 else source_text,
            component_type="evidence",
            start_idx=0,
            end_idx=min(len(source_text), 80),
            confidence=0.80,
        )
        target_comp = ArgumentComponent(
            text=target_text[:80] if len(target_text) > 80 else target_text,
            component_type="claim",
            start_idx=0,
            end_idx=min(len(target_text), 80),
            confidence=0.80,
        )
        return PipelineResult(
            components=[comp, target_comp],
            relations=[
                Relation(
                    source_component=comp,
                    target_component=target_comp,
                    relation_type="support",
                    confidence=0.80,
                    probabilities={"support": 0.80, "attack": 0.10, "neutral": 0.10},
                )
            ],
            overall_strength=0.75,
            feedback="Test feedback: güçlü destekleme kurulmuş.",
        )


@pytest.fixture(scope="function")
async def test_env():
    """Fixture to set up temporary database, mock BERT, and create test app."""
    # Mock BERT
    old_bert = DebateService._bert_instance
    DebateService._bert_instance = FakeBert()

    # Store old config
    old_db_uri = Config.SQLALCHEMY_DATABASE_URI
    old_openrouter_key = Config.OPENROUTER_API_KEY
    Config.OPENROUTER_API_KEY = ""  # Force mock mode

    # Create temporary database
    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / "test.db"
    test_db_url = f"sqlite+aiosqlite:///{db_path}"
    Config.SQLALCHEMY_DATABASE_URI = test_db_url

    # Swap database engine and session factory globally
    old_engine = app.database.engine
    old_session_factory = app.database.async_session_factory

    from sqlalchemy.ext.asyncio import create_async_engine, async_sessionmaker, AsyncSession
    test_engine = create_async_engine(test_db_url, connect_args={"timeout": 30})
    app.database.engine = test_engine
    app.database.async_session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )

    # Initialize schema
    await app.database.init_db()

    # Create FastAPI app
    app_instance = create_app()

    # Seed agents
    async with app.database.async_session_factory() as session:
        agent_pro = Agent(
            name="Test Savunucu",
            model_name="openai/gpt-4o-mini",
            role="proponent",
            system_prompt="Savun",
            temperature=0.7,
        )
        agent_opp = Agent(
            name="Test Karşıt",
            model_name="openai/gpt-4o-mini",
            role="opponent",
            system_prompt="Çürüt",
            temperature=0.7,
        )
        session.add_all([agent_pro, agent_opp])
        await session.commit()
        await session.refresh(agent_pro)
        await session.refresh(agent_opp)

    yield app_instance

    # Cleanup
    await test_engine.dispose()
    app.database.engine = old_engine
    app.database.async_session_factory = old_session_factory
    Config.SQLALCHEMY_DATABASE_URI = old_db_uri
    Config.OPENROUTER_API_KEY = old_openrouter_key
    DebateService._bert_instance = old_bert
    tmpdir.cleanup()


@pytest.mark.asyncio
async def test_create_agent_and_debate_lifecycle(test_env):
    """Test full lifecycle: create agents → create debate → start → advance → resolve."""
    app_instance = test_env
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        # 1. List agents (seeded)
        agents_res = await ac.get("/api/agents")
        assert agents_res.status_code == 200
        agents = agents_res.json()
        assert len(agents) == 2
        pro_agent = [a for a in agents if a["role"] == "proponent"][0]
        opp_agent = [a for a in agents if a["role"] == "opponent"][0]

        # 2. Create debate
        debate_res = await ac.post(
            "/api/debates",
            json={
                "topic": "Yapay zeka eğitim sistemlerini iyileştirir mi?",
                "agent_ids": [pro_agent["id"], opp_agent["id"]],
                "max_rounds": 2,
            },
        )
        assert debate_res.status_code == 201
        debate = debate_res.json()
        debate_id = debate["id"]
        assert debate["status"] == "pending"
        assert debate["topic"] == "Yapay zeka eğitim sistemlerini iyileştirir mi?"
        assert debate["auto_advance"] is False

        # 3. Start debate
        start_res = await ac.post(f"/api/debates/{debate_id}/start")
        assert start_res.status_code == 200
        assert start_res.json()["started"] is True

        # 4. Get debate (should be active with opening claim)
        get_res = await ac.get(f"/api/debates/{debate_id}")
        assert get_res.status_code == 200
        debate_data = get_res.json()
        assert debate_data["status"] == "active"
        assert debate_data["current_round"] == 1

        # 5. Get messages (should have opening claim)
        messages_res = await ac.get(f"/api/debates/{debate_id}/messages")
        assert messages_res.status_code == 200
        messages = messages_res.json()
        assert len(messages) == 1
        assert messages[0]["message_type"] == "claim"
        assert messages[0]["current_version"] is not None
        assert messages[0]["current_version"]["content"] != ""

        # 6. Advance debate (first attack)
        advance_res = await ac.post(f"/api/debates/{debate_id}/advance")
        assert advance_res.status_code == 200
        assert advance_res.json()["advanced"] is True

        # 7. Check messages after advance
        messages_res = await ac.get(f"/api/debates/{debate_id}/messages")
        messages = messages_res.json()
        assert len(messages) == 2
        assert messages[1]["message_type"] == "attack"
        assert messages[1]["parent_id"] == messages[0]["id"]

        # 8. Test auto-advance toggle
        auto_res = await ac.post(
            f"/api/debates/{debate_id}/auto-advance",
            json={"enabled": True},
        )
        assert auto_res.status_code == 200
        assert auto_res.json()["enabled"] is True

        # 9. Check debate reflects auto_advance
        get_res = await ac.get(f"/api/debates/{debate_id}")
        assert get_res.json()["auto_advance"] is True

        # 10. Resolve debate
        resolve_res = await ac.post(f"/api/debates/{debate_id}/resolve")
        assert resolve_res.status_code == 200
        assert resolve_res.json()["resolved"] is True

        # 11. Verify resolved status
        get_res = await ac.get(f"/api/debates/{debate_id}")
        assert get_res.json()["status"] == "resolved"

        # 12. Argument map should exist even if empty
        map_res = await ac.get(f"/api/debates/{debate_id}/argument-map")
        assert map_res.status_code == 200
        map_data = map_res.json()
        assert "status" in map_data


@pytest.mark.asyncio
async def test_run_full_debate_auto_advance(test_env):
    """Test the run-full endpoint runs a debate to completion."""
    app_instance = test_env
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        # Seed agents
        agents_res = await ac.get("/api/agents")
        agents = agents_res.json()
        pro_agent = [a for a in agents if a["role"] == "proponent"][0]
        opp_agent = [a for a in agents if a["role"] == "opponent"][0]

        # Create debate with max_rounds=1
        debate_res = await ac.post(
            "/api/debates",
            json={
                "topic": "Test otomatik tartışma",
                "agent_ids": [pro_agent["id"], opp_agent["id"]],
                "max_rounds": 1,
            },
        )
        debate_id = debate_res.json()["id"]

        # Run full debate
        run_res = await ac.post(f"/api/debates/{debate_id}/run-full")
        assert run_res.status_code == 200
        assert run_res.json()["status"] == "running"

        # Allow time for async processing
        import asyncio
        await asyncio.sleep(1)

        # Verify debate has messages and is resolved
        get_res = await ac.get(f"/api/debates/{debate_id}")
        debate_data = get_res.json()
        assert debate_data["status"] == "resolved"
        assert debate_data["auto_advance"] is True

        messages_res = await ac.get(f"/api/debates/{debate_id}/messages")
        messages = messages_res.json()
        assert len(messages) >= 2  # At least claim + attack for 1 round


@pytest.mark.asyncio
async def test_regenerate_message_endpoint(test_env):
    """Test manual message regeneration via ModernBERT feedback."""
    app_instance = test_env
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        agents_res = await ac.get("/api/agents")
        agents = agents_res.json()
        pro_agent = [a for a in agents if a["role"] == "proponent"][0]
        opp_agent = [a for a in agents if a["role"] == "opponent"][0]

        debate_res = await ac.post(
            "/api/debates",
            json={
                "topic": "Regenerasyon testi",
                "agent_ids": [pro_agent["id"], opp_agent["id"]],
                "max_rounds": 2,
            },
        )
        debate_id = debate_res.json()["id"]

        # Start and advance to create attack message
        await ac.post(f"/api/debates/{debate_id}/start")
        await ac.post(f"/api/debates/{debate_id}/advance")

        messages_res = await ac.get(f"/api/debates/{debate_id}/messages")
        messages = messages_res.json()
        attack_msg = [m for m in messages if m["message_type"] == "attack"][0]

        # Trigger regeneration
        regen_res = await ac.post(
            f"/api/debates/{debate_id}/messages/{attack_msg['id']}/regenerate"
        )
        assert regen_res.status_code == 200
        regen_data = regen_res.json()
        assert regen_data["success"] is True
        assert "version" in regen_data
        assert "feedback" in regen_data
        assert regen_data["version"]["version_number"] > 1
