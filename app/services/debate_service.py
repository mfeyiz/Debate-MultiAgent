"""Debate orchestration service."""

from __future__ import annotations

import datetime
import json
import queue
import threading
import time
from typing import Dict, List

from app.config import Config
from app.extensions import db
from app.models import Agent, Analysis, Debate, DebateParticipant, Message, MessageVersion
from app.services.agent_service import AgentService
from app.services.bert_service import ModernBERTPipeline


class _InMemoryEventBroker:
    """Fallback in-memory pub/sub broker for local development."""

    _lock = threading.Lock()
    _queues: Dict[int, List[queue.Queue]] = {}

    @classmethod
    def subscribe(cls, debate_id: int) -> queue.Queue:
        with cls._lock:
            q = queue.Queue()
            cls._queues.setdefault(debate_id, []).append(q)
            return q

    @classmethod
    def unsubscribe(cls, debate_id: int, q: queue.Queue) -> None:
        with cls._lock:
            qs = cls._queues.get(debate_id, [])
            if q in qs:
                qs.remove(q)

    @classmethod
    def publish(cls, debate_id: int, event: dict) -> None:
        with cls._lock:
            qs = cls._queues.get(debate_id, [])
            for q in qs:
                try:
                    q.put_nowait(event)
                except queue.Full:
                    pass


class _RedisEventBroker:
    """Redis-backed pub/sub broker for multi-pod deployments."""

    def __init__(self) -> None:
        import redis

        self._redis = redis.from_url(Config.REDIS_URL)
        self._pubsub = self._redis.pubsub(ignore_subscribe_messages=True)
        self._local_queues: Dict[int, List[queue.Queue]] = {}
        self._lock = threading.Lock()
        self._listener_thread = threading.Thread(target=self._listen, daemon=True)
        self._listener_thread.start()

    def _channel(self, debate_id: int) -> str:
        return f"debate:{debate_id}:events"

    def _listen(self) -> None:
        """Background thread that forwards Redis messages to local queues."""
        for message in self._pubsub.listen():
            if message["type"] != "message":
                continue
            try:
                data = json.loads(message["data"])
                debate_id = data.get("_debate_id")
                if debate_id is None:
                    continue
                with self._lock:
                    qs = self._local_queues.get(debate_id, [])
                for q in qs:
                    try:
                        q.put_nowait(data["event"])
                    except queue.Full:
                        pass
            except Exception:
                pass

    def subscribe(self, debate_id: int) -> queue.Queue:
        with self._lock:
            q = queue.Queue()
            self._local_queues.setdefault(debate_id, []).append(q)
        channel = self._channel(debate_id)
        self._pubsub.subscribe(channel)
        return q

    def unsubscribe(self, debate_id: int, q: queue.Queue) -> None:
        with self._lock:
            qs = self._local_queues.get(debate_id, [])
            if q in qs:
                qs.remove(q)
            if not qs:
                channel = self._channel(debate_id)
                self._pubsub.unsubscribe(channel)

    def publish(self, debate_id: int, event: dict) -> None:
        channel = self._channel(debate_id)
        payload = json.dumps({"_debate_id": debate_id, "event": event})
        self._redis.publish(channel, payload)


def _get_broker():
    """Return the appropriate event broker based on configuration."""
    if Config.REDIS_URL:
        return _RedisEventBroker()
    return _InMemoryEventBroker()


EventBroker = _get_broker()


class DebateService:
    """High-level service for managing debate lifecycle."""

    def __init__(self) -> None:
        self.agent_svc = AgentService()
        self.bert = ModernBERTPipeline()

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    def create_agent(
        self,
        name: str,
        model_name: str = "",
        role: str = "proponent",
        system_prompt: str = "",
        temperature: float = 0.7,
    ) -> Agent:
        agent = Agent(
            name=name,
            model_name=model_name or Config.DEFAULT_MODEL,
            role=role,
            system_prompt=system_prompt or self._default_prompt(role),
            temperature=temperature,
        )
        db.session.add(agent)
        db.session.commit()
        return agent

    @staticmethod
    def _default_prompt(role: str) -> str:
        defaults = {
            "proponent": (
                "Kesin bir savunucusunuz. Kanıt ve mantıksal akıl yürütme kullanarak "
                "iyi desteklenmiş iddialar oluşturun."
            ),
            "opponent": (
                "Eleştirel bir karşıtsınız. Karşı görüşteki zayıflıkları ve çelişkileri "
                "hassasiyetle tespit edin."
            ),
            "moderator": (
                "Tarafsız bir moderatörsünüz. Argümanları sentezleyin, mantıksal hataları vurgulayın "
                "ve yeni birincil argümanlar sunmadan tartışmayı yönetin."
            ),
        }
        return defaults.get(role, "")

    def create_debate(self, topic: str, agent_ids: List[int], max_rounds: int = 5) -> Debate:
        debate = Debate(topic=topic, status="pending", max_rounds=max_rounds)
        db.session.add(debate)
        db.session.flush()

        for idx, agent_id in enumerate(agent_ids):
            agent = db.session.get(Agent, agent_id)
            if not agent:
                raise ValueError(f"Ajan {agent_id} bulunamadı")
            role = "proponent" if idx == 0 else "opponent"
            dp = DebateParticipant(
                debate_id=debate.id,
                agent_id=agent.id,
                position=idx,
                role_in_debate=role,
            )
            db.session.add(dp)

        db.session.commit()
        return debate

    def get_debate(self, debate_id: int) -> Debate | None:
        return db.session.get(Debate, debate_id)

    def list_debates(self) -> List[Debate]:
        return Debate.query.order_by(Debate.created_at.desc()).all()

    def list_agents(self) -> List[Agent]:
        return Agent.query.order_by(Agent.created_at.desc()).all()

    # ------------------------------------------------------------------
    # Debate flow
    # ------------------------------------------------------------------

    def start_debate(self, debate_id: int) -> None:
        """Kick off a debate with the opening claim from the first agent."""
        debate = self.get_debate(debate_id)
        if not debate:
            return

        # Allow re-starting an active debate that has no messages yet
        if debate.status not in ("pending", "active"):
            return

        participants = (
            DebateParticipant.query.filter_by(debate_id=debate_id)
            .order_by(DebateParticipant.position.asc())
            .all()
        )
        if len(participants) < 2:
            raise ValueError("Tartışma başlatmak için en az 2 ajan gerekli")

        # If messages already exist, debate is truly started
        if debate.messages.first():
            return

        debate.status = "active"
        debate.current_round = 1
        db.session.commit()

        proponent = participants[0].agent
        opening_text = self.agent_svc.generate_opening_claim(proponent, debate.topic)

        self._post_message(
            debate_id=debate.id,
            agent_id=proponent.id,
            content=opening_text,
            message_type="claim",
            position=0,
        )

        self._broadcast(debate.id, {"type": "status", "status": debate.status})
        self._advance_if_needed(debate.id)

    def advance_debate(self, debate_id: int) -> None:
        """Trigger the next agent to speak."""
        debate = self.get_debate(debate_id)
        if not debate or debate.status != "active":
            return

        messages = debate.messages.all()
        last_msg = messages[-1] if messages else None
        participants = (
            DebateParticipant.query.filter_by(debate_id=debate_id)
            .order_by(DebateParticipant.position.asc())
            .all()
        )

        # Determine whose turn it is
        if last_msg is None:
            next_agent = participants[0].agent
            msg_type = "claim"
            parent_id = None
        else:
            last_participant = next(
                (p for p in participants if p.agent_id == last_msg.agent_id), None
            )
            last_pos = last_participant.position if last_participant else -1
            next_pos = (last_pos + 1) % len(participants)
            next_participant = participants[next_pos]
            next_agent = next_participant.agent

            if last_msg.message_type in ("claim", "rebuttal"):
                msg_type = "attack"
            else:
                msg_type = "rebuttal"
            parent_id = last_msg.id

        # Build prompt context
        context_messages = messages

        response_text = self.agent_svc.generate_response(
            agent_config=next_agent,
            topic=debate.topic,
            messages=context_messages,
        )

        position = len(messages)
        self._post_message(
            debate_id=debate.id,
            agent_id=next_agent.id,
            content=response_text,
            message_type=msg_type,
            position=position,
            parent_id=parent_id,
        )

        self._broadcast(debate.id, {"type": "turn_complete", "agent_id": next_agent.id})
        self._evaluate_pair_if_ready(debate.id)
        self._advance_if_needed(debate.id)

    def force_resolution(self, debate_id: int) -> None:
        """Manually resolve a debate."""
        debate = self.get_debate(debate_id)
        if debate:
            debate.status = "resolved"
            db.session.commit()
            self._broadcast(debate.id, {"type": "status", "status": "resolved"})

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _post_message(
        self,
        debate_id: int,
        agent_id: int,
        content: str,
        message_type: str,
        position: int,
        parent_id: int | None = None,
    ) -> Message:
        msg = Message(
            debate_id=debate_id,
            agent_id=agent_id,
            parent_id=parent_id,
            message_type=message_type,
            position=position,
        )
        db.session.add(msg)
        db.session.flush()

        version = MessageVersion(
            message_id=msg.id,
            version_number=1,
            content=content,
            is_current=True,
        )
        db.session.add(version)
        db.session.commit()

        self._broadcast(
            debate_id,
            {
                "type": "message",
                "message": msg.to_dict(),
            },
        )
        return msg

    def _evaluate_pair_if_ready(self, debate_id: int) -> None:
        """After every pair of messages, run the ModernBERT pipeline."""
        debate = self.get_debate(debate_id)
        if not debate:
            return

        messages = debate.messages.all()
        if len(messages) < 2:
            return

        # We evaluate the *latest* message against the one it replies to.
        latest_msg = messages[-1]
        if not latest_msg.parent_id:
            return

        parent_msg = db.session.get(Message, latest_msg.parent_id)
        if not parent_msg or not parent_msg.current_version or not latest_msg.current_version:
            return

        # Run pipeline
        result = self.bert.analyze(
            source_text=latest_msg.current_version.content,
            target_text=parent_msg.current_version.content,
            source_type="evidence",
            target_type="claim",
        )

        # Store analysis
        analysis = Analysis(
            message_version_id=latest_msg.current_version.id,
            target_message_version_id=parent_msg.current_version.id,
            component_type="evidence",
            relation_type=result.relations[0].relation_type if result.relations else "neutral",
            confidence=result.relations[0].confidence if result.relations else 0.0,
            feedback_text=result.feedback,
        )
        db.session.add(analysis)

        # Update strength score on the version
        latest_msg.current_version.strength_score = result.overall_strength
        db.session.commit()

        self._broadcast(
            debate_id,
            {
                "type": "analysis",
                "analysis": analysis.to_dict(),
                "message_id": latest_msg.id,
                "strength": result.overall_strength,
            },
        )

    def _advance_if_needed(self, debate_id: int) -> None:
        """Auto-advance the debate if conditions are met."""
        debate = self.get_debate(debate_id)
        if not debate or debate.status not in ("active",):
            return

        messages = debate.messages.all()
        total_messages = len(messages)

        # Resolve if max rounds reached (each round = 2 messages)
        if total_messages >= debate.max_rounds * 2:
            debate.status = "resolved"
            db.session.commit()
            self._broadcast(debate_id, {"type": "status", "status": "resolved"})
            return

        # Update round counter: claim starts a round, attack completes it
        debate.current_round = (total_messages + 1) // 2
        db.session.commit()

    def _broadcast(self, debate_id: int, event: dict) -> None:
        EventBroker.publish(debate_id, event)
