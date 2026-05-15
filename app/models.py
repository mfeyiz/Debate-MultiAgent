"""SQLAlchemy database models."""

from __future__ import annotations

import datetime
from typing import List

from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.extensions import db


class Agent(db.Model):
    """A configurable debating agent backed by an LLM."""

    __tablename__ = "agents"
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(db.String(120), nullable=False, unique=True)
    model_provider: Mapped[str] = mapped_column(db.String(50), default="openrouter")
    model_name: Mapped[str] = mapped_column(db.String(120), default="openai/gpt-4o-mini")
    role: Mapped[str] = mapped_column(
        db.String(20), default="proponent"
    )  # proponent | opponent | moderator
    system_prompt: Mapped[str] = mapped_column(db.Text, default="")
    temperature: Mapped[float] = mapped_column(db.Float, default=0.7)
    created_at: Mapped[datetime.datetime] = mapped_column(
        db.DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        db.DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    messages: Mapped[List["Message"]] = relationship(
        back_populates="agent", lazy="dynamic"
    )
    participations: Mapped[List["DebateParticipant"]] = relationship(
        back_populates="agent", lazy="dynamic"
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


class Debate(db.Model):
    """A debate session between two or more agents on a specific topic."""

    __tablename__ = "debates"
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    topic: Mapped[str] = mapped_column(db.Text, nullable=False)
    status: Mapped[str] = mapped_column(
        db.String(20), default="pending"
    )  # pending | active | paused | resolved
    max_rounds: Mapped[int] = mapped_column(db.Integer, default=5)
    current_round: Mapped[int] = mapped_column(db.Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        db.DateTime, default=datetime.datetime.utcnow
    )
    updated_at: Mapped[datetime.datetime] = mapped_column(
        db.DateTime,
        default=datetime.datetime.utcnow,
        onupdate=datetime.datetime.utcnow,
    )

    participants: Mapped[List["DebateParticipant"]] = relationship(
        back_populates="debate", lazy="dynamic"
    )
    messages: Mapped[List["Message"]] = relationship(
        back_populates="debate",
        lazy="dynamic",
        order_by="Message.position.asc()",
    )

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "topic": self.topic,
            "status": self.status,
            "max_rounds": self.max_rounds,
            "current_round": self.current_round,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "participant_ids": [p.agent_id for p in self.participants],
        }


class DebateParticipant(db.Model):
    """Join table linking debates and agents with positional ordering."""

    __tablename__ = "debate_participants"
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    debate_id: Mapped[int] = mapped_column(db.ForeignKey("debates.id"), nullable=False)
    agent_id: Mapped[int] = mapped_column(db.ForeignKey("agents.id"), nullable=False)
    position: Mapped[int] = mapped_column(db.Integer, default=0)
    role_in_debate: Mapped[str] = mapped_column(
        db.String(20), default="proponent"
    )  # proponent | opponent | moderator

    debate: Mapped[Debate] = relationship(back_populates="participants")
    agent: Mapped[Agent] = relationship(back_populates="participations")


class Message(db.Model):
    """A logical slot in a debate thread (supports versioning)."""

    __tablename__ = "messages"
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    debate_id: Mapped[int] = mapped_column(
        db.ForeignKey("debates.id"), nullable=False
    )
    agent_id: Mapped[int | None] = mapped_column(
        db.ForeignKey("agents.id"), nullable=True
    )
    parent_id: Mapped[int | None] = mapped_column(
        db.ForeignKey("messages.id"), nullable=True
    )
    message_type: Mapped[str] = mapped_column(
        db.String(20), default="claim"
    )  # claim | evidence | attack | rebuttal | system
    position: Mapped[int] = mapped_column(db.Integer, default=0)
    created_at: Mapped[datetime.datetime] = mapped_column(
        db.DateTime, default=datetime.datetime.utcnow
    )

    debate: Mapped[Debate] = relationship(back_populates="messages")
    agent: Mapped[Agent | None] = relationship(back_populates="messages")
    versions: Mapped[List["MessageVersion"]] = relationship(
        back_populates="message",
        lazy="dynamic",
        order_by="MessageVersion.version_number.desc()",
    )
    children: Mapped[List["Message"]] = relationship(
        "Message",
        backref=db.backref("parent", remote_side=[id]),
        lazy="dynamic",
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


class MessageVersion(db.Model):
    """A specific revision of a message's content and its analysis."""

    __tablename__ = "message_versions"
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    message_id: Mapped[int] = mapped_column(
        db.ForeignKey("messages.id"), nullable=False
    )
    version_number: Mapped[int] = mapped_column(db.Integer, default=1)
    content: Mapped[str] = mapped_column(db.Text, nullable=False)
    is_current: Mapped[bool] = mapped_column(db.Boolean, default=True)
    strength_score: Mapped[float | None] = mapped_column(db.Float, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        db.DateTime, default=datetime.datetime.utcnow
    )

    message: Mapped[Message] = relationship(back_populates="versions")
    analyses: Mapped[List["Analysis"]] = relationship(
        back_populates="message_version",
        lazy="dynamic",
        foreign_keys="Analysis.message_version_id",
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


class Analysis(db.Model):
    """ModernBERT pipeline output for a specific message version."""

    __tablename__ = "analyses"
    __allow_unmapped__ = True

    id: Mapped[int] = mapped_column(primary_key=True)
    message_version_id: Mapped[int] = mapped_column(
        db.ForeignKey("message_versions.id"), nullable=False
    )
    target_message_version_id: Mapped[int | None] = mapped_column(
        db.ForeignKey("message_versions.id"), nullable=True
    )
    component_type: Mapped[str] = mapped_column(
        db.String(20), default="evidence"
    )  # claim | evidence
    relation_type: Mapped[str] = mapped_column(
        db.String(20), default="neutral"
    )  # support | attack | neutral
    confidence: Mapped[float] = mapped_column(db.Float, default=0.0)
    feedback_text: Mapped[str | None] = mapped_column(db.Text, nullable=True)
    created_at: Mapped[datetime.datetime] = mapped_column(
        db.DateTime, default=datetime.datetime.utcnow
    )

    message_version: Mapped[MessageVersion] = relationship(
        foreign_keys=[message_version_id],
        back_populates="analyses",
    )
    target_message_version: Mapped[MessageVersion | None] = relationship(
        foreign_keys=[target_message_version_id]
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
