"""SQLAlchemy database models for pure async operations."""

from __future__ import annotations

import datetime
import json
from typing import Any, List

from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.config import Config
from app.database import Base


class QueryList(list):
    """Custom collection class mimicking Flask-SQLAlchemy list querying methods.

    Enables backwards-compatibility for existing services and templates.
    """

    def all(self) -> list:
        return self

    def filter_by(self, **kwargs) -> QueryList:
        filtered = []
        for item in self:
            match = True
            for k, v in kwargs.items():
                if getattr(item, k, None) != v:
                    match = False
                    break
            if match:
                filtered.append(item)
        return QueryList(filtered)

    def first(self) -> Any | None:
        return self[0] if len(self) > 0 else None


class Agent(Base):
    """A configurable debating agent backed by an LLM."""

    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(120), nullable=False, unique=True)
    model_provider: Mapped[str] = mapped_column(String(50), default="openrouter")
    model_name: Mapped[str] = mapped_column(String(120), default="openai/gpt-4o-mini")
    role: Mapped[str] = mapped_column(
        String(20), default="proponent"
    )  # proponent | opponent | moderator
    system_prompt: Mapped[str] = mapped_column(Text, default="")
    temperature: Mapped[float] = mapped_column(Float, default=0.7)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="agent",
        lazy="selectin",
        collection_class=QueryList,
    )
    participations: Mapped[List["DebateParticipant"]] = relationship(
        back_populates="agent",
        lazy="selectin",
        collection_class=QueryList,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "model_name": self.model_name,
            "role": self.role,
            "system_prompt": self.system_prompt,
            "temperature": self.temperature,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Debate(Base):
    """A debate session between two or more agents on a specific topic."""

    __tablename__ = "debates"

    id: Mapped[int] = mapped_column(primary_key=True)
    topic: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(
        String(20), default="pending"
    )  # pending | active | paused | resolved
    max_rounds: Mapped[int] = mapped_column(Integer, default=5)
    current_round: Mapped[int] = mapped_column(Integer, default=0)
    auto_advance: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    participants: Mapped[List["DebateParticipant"]] = relationship(
        back_populates="debate",
        lazy="selectin",
        collection_class=QueryList,
    )
    messages: Mapped[List["Message"]] = relationship(
        back_populates="debate",
        lazy="selectin",
        order_by="Message.position.asc()",
        collection_class=QueryList,
    )
    # NOT selectin: eager-loading analysis runs cascades into every run's
    # components + relations (can be ~2k rows per debate), making simple Debate
    # queries (dashboard, list pages) crawl. Runs are loaded via explicit
    # select() where needed (get_argument_map), so on-demand loading is safe.
    analysis_runs: Mapped[List["AnalysisRun"]] = relationship(
        back_populates="debate",
        lazy="select",
        order_by="AnalysisRun.created_at.desc()",
        collection_class=QueryList,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "status": self.status,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "auto_advance": self.auto_advance,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "participant_ids": [p.agent_id for p in self.participants],
        }


class DebateParticipant(Base):
    """Join table linking debates and agents with positional ordering."""

    __tablename__ = "debate_participants"

    id: Mapped[int] = mapped_column(primary_key=True)
    debate_id: Mapped[int] = mapped_column(ForeignKey("debates.id"), nullable=False)
    agent_id: Mapped[int] = mapped_column(ForeignKey("agents.id"), nullable=False)
    position: Mapped[int] = mapped_column(Integer, default=0)
    role_in_debate: Mapped[str] = mapped_column(
        String(20), default="proponent"
    )  # proponent | opponent | moderator

    debate: Mapped[Debate] = relationship(back_populates="participants", lazy="selectin")
    agent: Mapped[Agent] = relationship(back_populates="participations", lazy="selectin")


class Message(Base):
    """A logical slot in a debate thread (supports versioning)."""

    __tablename__ = "messages"

    id: Mapped[int] = mapped_column(primary_key=True)
    debate_id: Mapped[int] = mapped_column(ForeignKey("debates.id"), nullable=False)
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id"), nullable=True)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("messages.id"), nullable=True)
    message_type: Mapped[str] = mapped_column(
        String(20), default="claim"
    )  # claim | evidence | attack | rebuttal | system
    position: Mapped[int] = mapped_column(Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    debate: Mapped[Debate] = relationship(back_populates="messages", lazy="selectin")
    agent: Mapped[Agent | None] = relationship(back_populates="messages", lazy="selectin")
    versions: Mapped[List["MessageVersion"]] = relationship(
        back_populates="message",
        lazy="selectin",
        order_by="MessageVersion.version_number.desc()",
        collection_class=QueryList,
    )
    
    parent: Mapped[Message | None] = relationship(
        "Message",
        remote_side=[id],
        back_populates="children",
        lazy="selectin",
    )
    children: Mapped[List["Message"]] = relationship(
        "Message",
        back_populates="parent",
        lazy="selectin",
        collection_class=QueryList,
    )

    @property
    def current_version(self) -> MessageVersion | None:
        return self.versions.filter_by(is_current=True).first()

    def to_dict(self) -> dict:
        cv = self.current_version
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent.name if self.agent else None,
            "parent_id": self.parent_id,
            "message_type": self.message_type,
            "position": self.position,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "current_version": cv.to_dict() if cv else None,
        }


class MessageVersion(Base):
    """A specific revision of a message's content and its analysis."""

    __tablename__ = "message_versions"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=False)
    version_number: Mapped[int] = mapped_column(Integer, default=1)
    content: Mapped[str] = mapped_column(Text, nullable=False)
    is_current: Mapped[bool] = mapped_column(Boolean, default=True)
    strength_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    message: Mapped[Message] = relationship(back_populates="versions", lazy="selectin")
    analyses: Mapped[List["Analysis"]] = relationship(
        back_populates="message_version",
        lazy="selectin",
        foreign_keys="Analysis.message_version_id",
        collection_class=QueryList,
    )
    argument_components: Mapped[List["ArgumentComponent"]] = relationship(
        back_populates="message_version",
        lazy="selectin",
        collection_class=QueryList,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "message_id": self.message_id,
            "version_number": self.version_number,
            "content": self.content,
            "is_current": self.is_current,
            "strength_score": self.strength_score,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class Analysis(Base):
    """ModernBERT pipeline output for a specific message version."""

    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(primary_key=True)
    message_version_id: Mapped[int] = mapped_column(
        ForeignKey("message_versions.id"), nullable=False
    )
    target_message_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("message_versions.id"), nullable=True
    )
    component_type: Mapped[str] = mapped_column(
        String(20), default="evidence"
    )  # claim | evidence | other
    relation_type: Mapped[str] = mapped_column(
        String(20), default="none"
    )  # support | attack | none
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    feedback_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    message_version: Mapped[MessageVersion] = relationship(
        foreign_keys=[message_version_id],
        back_populates="analyses",
        lazy="selectin",
    )
    target_message_version: Mapped[MessageVersion | None] = relationship(
        foreign_keys=[target_message_version_id],
        lazy="selectin",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "message_version_id": self.message_version_id,
            "target_message_version_id": self.target_message_version_id,
            "component_type": self.component_type,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "feedback_text": self.feedback_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class AnalysisRun(Base):
    """A full ModernBERT analysis pass over an entire debate."""

    __tablename__ = "analysis_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    debate_id: Mapped[int] = mapped_column(ForeignKey("debates.id"), nullable=False)
    status: Mapped[str] = mapped_column(String(20), default="running")
    relation_threshold: Mapped[float] = mapped_column(Float, default=0.60)
    attack_threshold: Mapped[float] = mapped_column(Float, default=0.70)
    component_model: Mapped[str] = mapped_column(String(240), default="")
    relation_model: Mapped[str] = mapped_column(String(240), default="")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    debate: Mapped[Debate] = relationship(back_populates="analysis_runs", lazy="selectin")
    # NOT selectin: a run can hold ~2k relations; eager-loading them whenever an
    # AnalysisRun is touched is the main source of page slowness. Loaded via
    # explicit select() in get_argument_map.
    components: Mapped[List["ArgumentComponent"]] = relationship(
        back_populates="analysis_run",
        lazy="select",
        cascade="all, delete-orphan",
        collection_class=QueryList,
    )
    relations: Mapped[List["ArgumentRelation"]] = relationship(
        back_populates="analysis_run",
        lazy="select",
        cascade="all, delete-orphan",
        foreign_keys="ArgumentRelation.analysis_run_id",
        collection_class=QueryList,
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "debate_id": self.debate_id,
            "status": self.status,
            "relation_threshold": self.relation_threshold,
            "attack_threshold": self.attack_threshold,
            "component_model": self.component_model,
            "relation_model": self.relation_model,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class ArgumentComponent(Base):
    """A claim/evidence span extracted by ModernBERT for a full analysis run."""

    __tablename__ = "argument_components"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id"), nullable=False
    )
    message_version_id: Mapped[int] = mapped_column(
        ForeignKey("message_versions.id"), nullable=False
    )
    message_id: Mapped[int] = mapped_column(ForeignKey("messages.id"), nullable=False)
    agent_id: Mapped[int | None] = mapped_column(ForeignKey("agents.id"), nullable=True)
    component_type: Mapped[str] = mapped_column(String(20), nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    end_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    topic_relation_type: Mapped[str | None] = mapped_column(String(20), nullable=True)
    topic_relation_confidence: Mapped[float | None] = mapped_column(Float, nullable=True)
    topic_relation_probabilities_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="components", lazy="selectin")
    message_version: Mapped[MessageVersion] = relationship(
        back_populates="argument_components", lazy="selectin"
    )
    message: Mapped[Message] = relationship(lazy="selectin")
    agent: Mapped[Agent | None] = relationship(lazy="selectin")

    def to_dict(self) -> dict:
        try:
            topic_probabilities = json.loads(self.topic_relation_probabilities_json or "{}")
        except json.JSONDecodeError:
            topic_probabilities = {}
        return {
            "id": self.id,
            "analysis_run_id": self.analysis_run_id,
            "message_version_id": self.message_version_id,
            "message_id": self.message_id,
            "agent_id": self.agent_id,
            "agent_name": self.agent.name if self.agent else None,
            "component_type": self.component_type,
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "confidence": self.confidence,
            "topic_relation_type": self.topic_relation_type,
            "topic_relation_confidence": self.topic_relation_confidence,
            "topic_relation_probabilities": topic_probabilities,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class ArgumentRelation(Base):
    """A ModernBERT support/attack/none relation between two components."""

    __tablename__ = "argument_relations"

    id: Mapped[int] = mapped_column(primary_key=True)
    analysis_run_id: Mapped[int] = mapped_column(
        ForeignKey("analysis_runs.id"), nullable=False
    )
    source_component_id: Mapped[int] = mapped_column(
        ForeignKey("argument_components.id"), nullable=False
    )
    target_component_id: Mapped[int] = mapped_column(
        ForeignKey("argument_components.id"), nullable=False
    )
    relation_type: Mapped[str] = mapped_column(String(20), nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    probabilities_json: Mapped[str] = mapped_column(Text, default="{}")
    distance_turns: Mapped[int] = mapped_column(Integer, default=0)
    is_long_range: Mapped[bool] = mapped_column(Boolean, default=False)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    analysis_run: Mapped[AnalysisRun] = relationship(back_populates="relations", lazy="selectin")
    source_component: Mapped[ArgumentComponent] = relationship(
        foreign_keys=[source_component_id], lazy="selectin"
    )
    target_component: Mapped[ArgumentComponent] = relationship(
        foreign_keys=[target_component_id], lazy="selectin"
    )

    def to_dict(self) -> dict:
        try:
            probabilities = json.loads(self.probabilities_json or "{}")
        except json.JSONDecodeError:
            probabilities = {}
        return {
            "id": self.id,
            "analysis_run_id": self.analysis_run_id,
            "source_component_id": self.source_component_id,
            "target_component_id": self.target_component_id,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "probabilities": probabilities,
            "distance_turns": self.distance_turns,
            "is_long_range": self.is_long_range,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class FactCheckRun(Base):
    """A media-literacy fact-check analysis run for a URL or pasted article."""

    __tablename__ = "fact_check_runs"

    id: Mapped[int] = mapped_column(primary_key=True)
    input_type: Mapped[str] = mapped_column(String(20), default="text")
    url: Mapped[str | None] = mapped_column(String(1000), nullable=True)
    title: Mapped[str | None] = mapped_column(String(500), nullable=True)
    raw_text: Mapped[str] = mapped_column(Text, nullable=False)
    status: Mapped[str] = mapped_column(String(30), default="running")
    source_metadata_json: Mapped[str] = mapped_column(Text, default="{}")
    summary: Mapped[str | None] = mapped_column(Text, nullable=True)
    export_token: Mapped[str | None] = mapped_column(
        String(64), nullable=True, unique=True, index=True
    )
    balanced_reporting_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    manipulation_findings_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )
    completed_at: Mapped[datetime.datetime | None] = mapped_column(
        DateTime, nullable=True
    )

    claims: Mapped[List["FactClaim"]] = relationship(
        back_populates="run",
        lazy="selectin",
        cascade="all, delete-orphan",
        collection_class=QueryList,
    )
    evidence: Mapped[List["FactEvidence"]] = relationship(
        back_populates="run",
        lazy="selectin",
        cascade="all, delete-orphan",
        collection_class=QueryList,
    )
    relations: Mapped[List["FactRelation"]] = relationship(
        back_populates="run",
        lazy="selectin",
        cascade="all, delete-orphan",
        collection_class=QueryList,
    )

    def to_dict(self) -> dict:
        try:
            metadata = json.loads(self.source_metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        try:
            balanced = json.loads(self.balanced_reporting_json or "{}")
        except json.JSONDecodeError:
            balanced = {}
        try:
            manipulation = json.loads(self.manipulation_findings_json or "{}")
        except json.JSONDecodeError:
            manipulation = {}
        return {
            "id": self.id,
            "input_type": self.input_type,
            "url": self.url,
            "title": self.title,
            "raw_text": self.raw_text,
            "status": self.status,
            "source_metadata": metadata,
            "summary": self.summary,
            "export_token": self.export_token,
            "balanced_reporting": balanced,
            "manipulation_findings": manipulation,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
        }


class FactClaim(Base):
    """A text component extracted from an article for fact-check mapping."""

    __tablename__ = "fact_claims"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("fact_check_runs.id"), nullable=False)
    component_type: Mapped[str] = mapped_column(String(20), default="claim")
    text: Mapped[str] = mapped_column(Text, nullable=False)
    start_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    end_idx: Mapped[int] = mapped_column(Integer, nullable=False)
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    claim_type: Mapped[str] = mapped_column(String(50), default="factual")
    verdict_status: Mapped[str] = mapped_column(String(80), default="Kaynak bulunamadı")
    explanation: Mapped[str | None] = mapped_column(Text, nullable=True)
    claim_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    embedding_hash: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    is_statistical: Mapped[bool] = mapped_column(Boolean, default=False)
    statistical_data_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    manipulation_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    quote_data_json: Mapped[str | None] = mapped_column(Text, nullable=True)
    extraction_reason: Mapped[str] = mapped_column(String(80), default="model_claim")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    run: Mapped[FactCheckRun] = relationship(back_populates="claims", lazy="selectin")

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "run_id": self.run_id,
            "component_type": self.component_type,
            "text": self.text,
            "start_idx": self.start_idx,
            "end_idx": self.end_idx,
            "confidence": self.confidence,
            "claim_type": self.claim_type,
            "verdict_status": self.verdict_status,
            "explanation": self.explanation,
            "is_statistical": self.is_statistical,
            "statistical_data": json.loads(self.statistical_data_json or "{}"),
            "manipulation_score": self.manipulation_score,
            "quote_data": json.loads(self.quote_data_json or "{}"),
            "extraction_reason": self.extraction_reason,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class FactEvidence(Base):
    """A source snippet found during internet-backed fact-check retrieval."""

    __tablename__ = "fact_evidence"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("fact_check_runs.id"), nullable=False)
    claim_id: Mapped[int | None] = mapped_column(ForeignKey("fact_claims.id"), nullable=True)
    search_query: Mapped[str] = mapped_column("query", Text, default="")
    url: Mapped[str] = mapped_column(String(1000), default="")
    title: Mapped[str] = mapped_column(String(500), default="")
    source_domain: Mapped[str] = mapped_column(String(240), default="")
    snippet: Mapped[str] = mapped_column(Text, default="")
    published_at: Mapped[str | None] = mapped_column(String(80), nullable=True)
    score: Mapped[float] = mapped_column(Float, default=0.0)
    credibility_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    source_bias: Mapped[str | None] = mapped_column(String(20), nullable=True)
    relevance_score: Mapped[float] = mapped_column(Float, default=0.0)
    source_quality: Mapped[str] = mapped_column(String(40), default="unscored")
    accepted_for_verdict: Mapped[bool] = mapped_column(Boolean, default=False)
    is_turkish_archive: Mapped[bool] = mapped_column(Boolean, default=False)
    is_public_data_source: Mapped[bool] = mapped_column(Boolean, default=False)
    archive_match_claim_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    run: Mapped[FactCheckRun] = relationship(back_populates="evidence", lazy="selectin")
    claim: Mapped[FactClaim | None] = relationship(lazy="selectin")

    def to_dict(self) -> dict:
        source_type = (
            "fact_check_archive"
            if any(domain in self.source_domain for domain in Config.FACT_CHECK_ARCHIVE_DOMAINS)
            else "knowledge_base"
            if any(domain in self.source_domain for domain in Config.FACT_CHECK_TRUSTED_DOMAINS)
            else "web"
        )
        return {
            "id": self.id,
            "run_id": self.run_id,
            "claim_id": self.claim_id,
            "query": self.search_query,
            "url": self.url,
            "title": self.title,
            "source_domain": self.source_domain,
            "source_type": source_type,
            "snippet": self.snippet,
            "published_at": self.published_at,
            "score": self.score,
            "credibility_score": self.credibility_score,
            "source_bias": self.source_bias,
            "relevance_score": self.relevance_score,
            "source_quality": self.source_quality,
            "accepted_for_verdict": self.accepted_for_verdict,
            "is_turkish_archive": self.is_turkish_archive,
            "is_public_data_source": self.is_public_data_source,
            "archive_match_claim_text": self.archive_match_claim_text,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class FactRelation(Base):
    """A ModernBERT relation from article/source evidence to an article claim."""

    __tablename__ = "fact_relations"

    id: Mapped[int] = mapped_column(primary_key=True)
    run_id: Mapped[int] = mapped_column(ForeignKey("fact_check_runs.id"), nullable=False)
    source_claim_id: Mapped[int | None] = mapped_column(
        ForeignKey("fact_claims.id"), nullable=True
    )
    evidence_id: Mapped[int | None] = mapped_column(ForeignKey("fact_evidence.id"), nullable=True)
    target_claim_id: Mapped[int] = mapped_column(ForeignKey("fact_claims.id"), nullable=False)
    relation_scope: Mapped[str] = mapped_column(String(20), default="internal")
    relation_type: Mapped[str] = mapped_column(String(20), default="none")
    confidence: Mapped[float] = mapped_column(Float, default=0.0)
    probabilities_json: Mapped[str] = mapped_column(Text, default="{}")
    created_at: Mapped[datetime.datetime] = mapped_column(
        DateTime, default=datetime.datetime.utcnow
    )

    run: Mapped[FactCheckRun] = relationship(back_populates="relations", lazy="selectin")
    source_claim: Mapped[FactClaim | None] = relationship(
        foreign_keys=[source_claim_id], lazy="selectin"
    )
    evidence: Mapped[FactEvidence | None] = relationship(lazy="selectin")
    target_claim: Mapped[FactClaim] = relationship(
        foreign_keys=[target_claim_id], lazy="selectin"
    )

    def to_dict(self) -> dict:
        try:
            probabilities = json.loads(self.probabilities_json or "{}")
        except json.JSONDecodeError:
            probabilities = {}
        return {
            "id": self.id,
            "run_id": self.run_id,
            "source_claim_id": self.source_claim_id,
            "evidence_id": self.evidence_id,
            "target_claim_id": self.target_claim_id,
            "relation_scope": self.relation_scope,
            "relation_type": self.relation_type,
            "confidence": self.confidence,
            "probabilities": probabilities,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
