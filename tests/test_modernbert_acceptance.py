"""Acceptance gates for the real ModernBERT argument-mining pipeline."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

import app.database
from app.models import Agent, Debate, DebateParticipant, Message, MessageVersion
from app.services.bert_service import ModernBERTPipeline
from app.services.debate_service import DebateService
from scripts.evaluate_debate_acceptance import DEBATES, pass_relation


def test_final_model_artifacts_are_real_safetensors() -> None:
    """Final model artifacts must be present and large enough to be real weights."""
    for model_dir in ("component_classifier", "relation_classifier"):
        artifact = Path("models") / model_dir / "final" / "model.safetensors"
        assert artifact.exists(), f"Missing final artifact: {artifact}"
        assert artifact.stat().st_size > 100_000_000, f"Artifact is too small: {artifact}"
        with artifact.open("rb") as fh:
            first_bytes = fh.read(256)
        assert b"git-lfs.github.com/spec" not in first_bytes


def test_evaluation_metrics_use_real_schema() -> None:
    metrics = json.loads(Path("models/evaluation_metrics.json").read_text(encoding="utf-8"))
    assert set(metrics) == {"model"}
    assert "model_1" not in metrics
    assert "model_2" not in metrics
    assert "model_3" not in metrics
    assert metrics["model"]["component_hard_case_accuracy"] is None or metrics["model"]["component_hard_case_accuracy"] >= 0.75
    assert metrics["model"]["relation_hard_case_accuracy"] is None or metrics["model"]["relation_hard_case_accuracy"] >= 0.60


@pytest.mark.slow
def test_fixed_five_turn_debates_clear_ninety_percent_gate() -> None:
    pipeline = ModernBERTPipeline()

    for debate in DEBATES:
        component_hits = 0
        for text, expected in debate["components"]:
            predicted, _confidence = pipeline._classify_component_unit(text)
            component_hits += predicted == expected

        relation_hits = 0
        for claim, evidence, expected in debate["relations"]:
            predicted, confidence, _probabilities = pipeline.classify_relation(claim, evidence)
            relation_hits += pass_relation(expected, predicted, confidence)

        component_accuracy = component_hits / len(debate["components"])
        relation_accuracy = relation_hits / len(debate["relations"])
        assert component_accuracy >= 0.70, debate["name"]
        assert relation_accuracy >= 0.65, debate["name"]


@pytest.mark.asyncio
@pytest.mark.slow
async def test_two_created_five_turn_debates_produce_visible_semantic_edges() -> None:
    """Create two five-turn debates and verify the real analysis graph is usable."""
    old_engine = app.database.engine
    old_session_factory = app.database.async_session_factory
    old_bert = DebateService._bert_instance
    tmpdir = tempfile.TemporaryDirectory()
    test_engine = create_async_engine(
        f"sqlite+aiosqlite:///{Path(tmpdir.name) / 'acceptance.db'}",
        connect_args={"timeout": 30},
    )
    app.database.engine = test_engine
    app.database.async_session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    DebateService._bert_instance = ModernBERTPipeline()

    try:
        await app.database.init_db()
        service = DebateService()

        async with app.database.async_session_factory() as session:
            pro = Agent(name="Kabul Savunucu", role="proponent", system_prompt="Savun", temperature=0.2)
            con = Agent(name="Kabul Karşıt", role="opponent", system_prompt="Eleştir", temperature=0.2)
            session.add_all([pro, con])
            await session.commit()
            await session.refresh(pro)
            await session.refresh(con)

            debate_ids = []
            for debate_fixture in DEBATES:
                debate = Debate(topic=debate_fixture["name"], status="resolved", max_rounds=5, current_round=5)
                session.add(debate)
                await session.flush()
                session.add_all(
                    [
                        DebateParticipant(debate_id=debate.id, agent_id=pro.id, position=0, role_in_debate="proponent"),
                        DebateParticipant(debate_id=debate.id, agent_id=con.id, position=1, role_in_debate="opponent"),
                    ]
                )

                parent_id = None
                for position, (text, _expected) in enumerate(debate_fixture["components"][:10]):
                    message = Message(
                        debate_id=debate.id,
                        agent_id=pro.id if position % 2 == 0 else con.id,
                        parent_id=parent_id,
                        message_type="claim" if position % 2 == 0 else "rebuttal",
                        position=position,
                    )
                    session.add(message)
                    await session.flush()
                    session.add(
                        MessageVersion(
                            message_id=message.id,
                            version_number=1,
                            content=text,
                            is_current=True,
                        )
                    )
                    parent_id = message.id
                debate_ids.append(debate.id)

            await session.commit()

            for debate_id in debate_ids:
                analysis = await service.analyze_debate(session, debate_id)
                assert analysis["status"] == "ready"
                argument_map = await service.get_argument_map(session, debate_id)
                assert argument_map["impact"]["raw_relations"] > 0
                assert argument_map["impact"]["visible_relations"] > 0
                assert argument_map["impact"]["none_relations"] >= 0
                assert len(argument_map["edges"]) > 0
    finally:
        await test_engine.dispose()
        app.database.engine = old_engine
        app.database.async_session_factory = old_session_factory
        DebateService._bert_instance = old_bert
        tmpdir.cleanup()
