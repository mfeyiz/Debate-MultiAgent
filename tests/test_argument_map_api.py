"""Argument map API contract tests with async database operations."""

from __future__ import annotations

import asyncio
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
    """Small deterministic stand-in for ModernBERT in API tests."""

    def extract_components(self, text: str, default_type: str) -> list[ArgumentComponent]:
        return [
            ArgumentComponent(
                text=text,
                component_type=default_type,
                start_idx=0,
                end_idx=len(text),
                confidence=0.91,
            )
        ]

    def classify_relation(
        self,
        claim_text: str,
        evidence_text: str,
    ) -> tuple[str, float, dict[str, float]]:
        if "çürütür" in evidence_text.lower():
            return "attack", 0.92, {"support": 0.04, "attack": 0.92, "neutral": 0.04}
        return "support", 0.88, {"support": 0.88, "attack": 0.05, "neutral": 0.07}

    def analyze(self, source_text: str, target_text: str, source_type: str, target_type: str):
        from app.services.bert_service import AnalysisResult, RelationPrediction
        return AnalysisResult(
            overall_strength=0.85,
            components=[],
            relations=[RelationPrediction(
                source_text=source_text,
                target_text=target_text,
                relation_type="support",
                confidence=0.85,
                probabilities={"support": 0.85, "attack": 0.05, "neutral": 0.10}
            )],
            feedback="İyi bir destekleme kurulmuş."
        )


@pytest.fixture(scope="function")
async def test_env():
    """Fixture to set up temporary database, mock BERT, and create test app."""
    # Mock BERT
    old_bert = DebateService._bert_instance
    DebateService._bert_instance = FakeBert()

    # Create temporary database
    tmpdir = tempfile.TemporaryDirectory()
    db_path = Path(tmpdir.name) / "test.db"
    test_db_url = f"sqlite+aiosqlite:///{db_path}"

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

    # Seed agents and test data
    async with app.database.async_session_factory() as session:
        agent_pro = Agent(
            name="Ajan A",
            model_name="openai/gpt-4o-mini",
            role="proponent",
            system_prompt="Defend claim",
            temperature=0.7,
        )
        agent_opp = Agent(
            name="Ajan B",
            model_name="openai/gpt-4o-mini",
            role="opponent",
            system_prompt="Refute claim",
            temperature=0.7,
        )
        session.add_all([agent_pro, agent_opp])
        await session.commit()
        await session.refresh(agent_pro)
        await session.refresh(agent_opp)

        # Create debate
        debate = Debate(topic="Test konusu", status="resolved", max_rounds=1)
        session.add(debate)
        await session.flush()

        # Add participants
        dp1 = DebateParticipant(
            debate_id=debate.id,
            agent_id=agent_pro.id,
            position=0,
            role_in_debate="proponent",
        )
        dp2 = DebateParticipant(
            debate_id=debate.id,
            agent_id=agent_opp.id,
            position=1,
            role_in_debate="opponent",
        )
        session.add_all([dp1, dp2])

        # Add messages
        first = Message(
            debate_id=debate.id,
            agent_id=agent_pro.id,
            message_type="claim",
            position=0,
        )
        session.add(first)
        await session.flush()

        v1 = MessageVersion(
            message_id=first.id,
            version_number=1,
            content="Ajan A iddiası.",
            is_current=True,
        )
        session.add(v1)

        second = Message(
            debate_id=debate.id,
            agent_id=agent_opp.id,
            parent_id=first.id,
            message_type="attack",
            position=1,
        )
        session.add(second)
        await session.flush()

        v2 = MessageVersion(
            message_id=second.id,
            version_number=1,
            content="Ajan B bunu çürütür.",
            is_current=True,
        )
        session.add(v2)

        await session.commit()
        debate_id = debate.id

    yield app_instance, debate_id

    # Cleanup
    await test_engine.dispose()
    app.database.engine = old_engine
    app.database.async_session_factory = old_session_factory
    DebateService._bert_instance = old_bert
    tmpdir.cleanup()


@pytest.mark.asyncio
async def test_analyze_endpoint_returns_argument_map_contract(test_env):
    app_instance, debate_id = test_env
    async with AsyncClient(transport=ASGITransport(app=app_instance), base_url="http://test") as ac:
        # Analyze debate
        analyze_response = await ac.post(f"/api/debates/{debate_id}/analyze")
        assert analyze_response.status_code == 200
        analyze_payload = analyze_response.json()
        assert analyze_payload["status"] == "ready"

        # Get argument map
        map_response = await ac.get(f"/api/debates/{debate_id}/argument-map")
        assert map_response.status_code == 200
        payload = map_response.json()

        assert payload["status"] == "ready"
        assert len(payload["nodes"]) >= 2
        assert len(payload["edges"]) >= 1
        assert len(payload["relations"]) >= 2
        assert "annotations" in payload
        assert "findings" in payload
        assert "impact" in payload
        assert "semantic_edges" in payload["impact"]

        topic_relations = {
            node["data"]["topicRelationType"]
            for node in payload["nodes"]
        }
        assert "support" in topic_relations
        assert "attack" in topic_relations

        annotation_spans = [
            span
            for spans in payload["annotations"].values()
            for span in spans
        ]
        assert annotation_spans
        assert all("topic_relation_type" in span for span in annotation_spans)
        assert all("topic_relation_confidence" in span for span in annotation_spans)
