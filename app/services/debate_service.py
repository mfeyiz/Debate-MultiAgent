"""Debate orchestration service with async DB operations and automated regeneration loop."""

from __future__ import annotations

import asyncio
import datetime
import json
import queue
import threading
from pathlib import Path
from collections import defaultdict
from typing import Dict, List

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Config
from app.models import (
    Agent,
    Analysis,
    AnalysisRun,
    ArgumentComponent as ArgumentComponentModel,
    ArgumentRelation,
    Debate,
    DebateParticipant,
    Message,
    MessageVersion,
)
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
    """High-level service for managing debate lifecycle asynchronously."""

    RELATION_GRAPH_THRESHOLD = 0.60
    ATTACK_RESPONSE_THRESHOLD = 0.70
    MAX_RELATION_COMPONENTS_PER_MESSAGE = 5
    MAX_GRAPH_EDGES = 48
    MAX_GRAPH_ATTACK_EDGES = 18
    MAX_GRAPH_SUPPORT_EDGES = 18
    MAX_GRAPH_EDGES_PER_SOURCE_TYPE = 2
    MAX_GRAPH_EDGES_PER_TARGET_TYPE = 6
    MAX_FINDINGS_PER_TYPE = 8

    _bert_lock = threading.Lock()
    _bert_instance: ModernBERTPipeline | None = None
    _bert_signature: tuple[float, float] | None = None
    _operation_locks: defaultdict[int, asyncio.Lock] = defaultdict(asyncio.Lock)

    def __init__(self) -> None:
        self.agent_svc = AgentService()

    async def acquire_operation_lock(self, debate_id: int, operation: str) -> asyncio.Lock:
        """Prevent overlapping mutating operations for a single debate."""
        lock = self.__class__._operation_locks[debate_id]
        if lock.locked():
            raise RuntimeError(
                f"'{operation}' başlatılamadı; bu tartışma için başka bir işlem halen çalışıyor."
            )
        await lock.acquire()
        return lock

    @property
    def bert(self) -> ModernBERTPipeline:
        """Load the ModernBERT models and refresh when local weights change."""
        signature = self._model_signature()
        should_reload = (
            self.__class__._bert_instance is None
            or (
                isinstance(self.__class__._bert_instance, ModernBERTPipeline)
                and self.__class__._bert_signature != signature
            )
        )
        if should_reload:
            with self.__class__._bert_lock:
                should_reload = (
                    self.__class__._bert_instance is None
                    or (
                        isinstance(self.__class__._bert_instance, ModernBERTPipeline)
                        and self.__class__._bert_signature != signature
                    )
                )
                if should_reload:
                    self.__class__._bert_instance = ModernBERTPipeline()
                    self.__class__._bert_signature = signature
        return self.__class__._bert_instance

    @staticmethod
    def _model_signature() -> tuple[float, float]:
        component_file = Path(Config.COMPONENT_MODEL_DIR) / "model.safetensors"
        relation_file = Path(Config.RELATION_MODEL_DIR) / "model.safetensors"
        return (
            component_file.stat().st_mtime if component_file.exists() else 0.0,
            relation_file.stat().st_mtime if relation_file.exists() else 0.0,
        )

    @staticmethod
    def _model_file_version(path: str) -> str:
        model_file = Path(path) / "model.safetensors"
        if not model_file.exists():
            return f"{path}:missing"
        stat = model_file.stat()
        return f"{path}:mtime={stat.st_mtime_ns}:size={stat.st_size}"

    @classmethod
    def _component_model_version(cls) -> str:
        return cls._model_file_version(Config.COMPONENT_MODEL_DIR)

    @classmethod
    def _relation_model_version(cls) -> str:
        return cls._model_file_version(Config.RELATION_MODEL_DIR)

    # ------------------------------------------------------------------
    # CRUD helpers
    # ------------------------------------------------------------------

    async def create_agent(
        self,
        db: AsyncSession,
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
        db.add(agent)
        await db.commit()
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

    async def create_debate(self, db: AsyncSession, topic: str, agent_ids: List[int], max_rounds: int = 5) -> Debate:
        debate = Debate(topic=topic, status="pending", max_rounds=max_rounds)
        db.add(debate)
        await db.flush()

        for idx, agent_id in enumerate(agent_ids):
            agent = await db.get(Agent, agent_id)
            if not agent:
                raise ValueError(f"Ajan {agent_id} bulunamadı")
            role = "proponent" if idx == 0 else "opponent"
            dp = DebateParticipant(
                debate_id=debate.id,
                agent_id=agent.id,
                position=idx,
                role_in_debate=role,
            )
            db.add(dp)

        await db.commit()
        # Refresh to load relationships
        await db.refresh(debate)
        return debate

    async def get_debate(self, db: AsyncSession, debate_id: int) -> Debate | None:
        return await db.get(Debate, debate_id)

    async def list_debates(self, db: AsyncSession) -> List[Debate]:
        stmt = select(Debate).order_by(Debate.created_at.desc())
        res = await db.execute(stmt)
        return list(res.scalars().all())

    async def list_agents(self, db: AsyncSession) -> List[Agent]:
        stmt = select(Agent).order_by(Agent.created_at.desc())
        res = await db.execute(stmt)
        return list(res.scalars().all())

    # ------------------------------------------------------------------
    # Debate flow
    # ------------------------------------------------------------------

    async def start_debate(self, db: AsyncSession, debate_id: int) -> None:
        """Kick off a debate with the opening claim from the first agent."""
        debate = await self.get_debate(db, debate_id)
        if not debate:
            return

        # Allow re-starting an active debate that has no messages yet
        if debate.status not in ("pending", "active"):
            return

        stmt = select(DebateParticipant).filter_by(debate_id=debate_id).order_by(DebateParticipant.position.asc())
        res = await db.execute(stmt)
        participants = res.scalars().all()
        if len(participants) < 2:
            raise ValueError("Tartışma başlatmak için en az 2 ajan gerekli")

        # If messages already exist, debate is truly started
        if debate.messages:
            return

        debate.status = "active"
        debate.current_round = 1
        await db.commit()

        proponent = participants[0].agent
        opening_text = await self.agent_svc.generate_opening_claim(proponent, debate.topic)

        await self._post_message(
            db=db,
            debate_id=debate.id,
            agent_id=proponent.id,
            content=opening_text,
            message_type="claim",
            position=0,
        )

        # Refresh debate to load newly posted message in relationship
        await db.refresh(debate)

        self._broadcast(debate.id, {"type": "status", "status": debate.status})
        await self._advance_if_needed(db, debate.id)

    async def advance_debate(self, db: AsyncSession, debate_id: int) -> None:
        """Trigger the next agent to speak."""
        debate = await self.get_debate(db, debate_id)
        if not debate or debate.status != "active":
            return

        messages = list(debate.messages)
        last_msg = messages[-1] if messages else None
        
        stmt = select(DebateParticipant).filter_by(debate_id=debate_id).order_by(DebateParticipant.position.asc())
        res = await db.execute(stmt)
        participants = res.scalars().all()

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

        response_text = await self.agent_svc.generate_response(
            agent_config=next_agent,
            topic=debate.topic,
            messages=context_messages,
        )

        position = len(messages)
        await self._post_message(
            db=db,
            debate_id=debate.id,
            agent_id=next_agent.id,
            content=response_text,
            message_type=msg_type,
            position=position,
            parent_id=parent_id,
        )

        # Refresh debate to load newly posted message in relationship
        await db.refresh(debate)

        self._broadcast(debate.id, {"type": "turn_complete", "agent_id": next_agent.id})
        await self._evaluate_pair_if_ready(db, debate.id)
        await self._advance_if_needed(db, debate.id)

    async def force_resolution(self, db: AsyncSession, debate_id: int) -> None:
        """Manually resolve a debate."""
        debate = await self.get_debate(db, debate_id)
        if debate:
            debate.status = "resolved"
            await db.commit()
            self._broadcast(debate.id, {"type": "status", "status": "resolved"})
            await self.analyze_debate(db, debate.id)

    # ------------------------------------------------------------------
    # Full ModernBERT argument map
    # ------------------------------------------------------------------

    async def analyze_debate(self, db: AsyncSession, debate_id: int) -> dict:
        """Run a full ModernBERT component/relation pass over the debate."""
        debate = await self.get_debate(db, debate_id)
        if not debate:
            raise ValueError("Tartışma bulunamadı")

        messages = [msg for msg in debate.messages if msg.current_version]
        if not messages:
            return await self.get_argument_map(db, debate_id)

        self._broadcast(debate_id, {"type": "analysis_started"})

        run = AnalysisRun(
            debate_id=debate.id,
            status="running",
            relation_threshold=self.RELATION_GRAPH_THRESHOLD,
            attack_threshold=self.ATTACK_RESPONSE_THRESHOLD,
            component_model=self._component_model_version(),
            relation_model=self._relation_model_version(),
        )
        db.add(run)
        await db.flush()

        try:
            components_by_message: dict[int, list[ArgumentComponentModel]] = defaultdict(list)

            for msg in messages:
                version = msg.current_version
                if not version:
                    continue
                default_type = self._default_component_type(msg.message_type)
                extracted_components = await asyncio.to_thread(
                    self.bert.extract_components,
                    version.content,
                    default_type=default_type,
                )
                extracted_components = self._ensure_claim_anchor(
                    extracted_components,
                    default_type,
                    msg.message_type,
                )
                for extracted in extracted_components:
                    component = ArgumentComponentModel(
                        analysis_run_id=run.id,
                        message_version_id=version.id,
                        message_id=msg.id,
                        agent_id=msg.agent_id,
                        component_type=extracted.component_type,
                        text=extracted.text,
                        start_idx=extracted.start_idx,
                        end_idx=extracted.end_idx,
                        confidence=extracted.confidence,
                    )
                    topic_relation_type, topic_confidence, topic_probabilities = await asyncio.to_thread(
                        self.bert.classify_relation,
                        debate.topic,
                        extracted.text,
                    )
                    component.topic_relation_type = topic_relation_type
                    component.topic_relation_confidence = topic_confidence
                    component.topic_relation_probabilities_json = json.dumps(topic_probabilities)
                    db.add(component)
                    components_by_message[msg.id].append(component)

            await db.flush()
            await self._classify_full_debate_relations(db, run, messages, components_by_message)
            await db.flush()
            await self._update_full_debate_strengths(db, run, messages)

            run.status = "completed"
            run.completed_at = datetime.datetime.utcnow()
            await db.commit()
        except Exception:
            run.status = "failed"
            run.completed_at = datetime.datetime.utcnow()
            await db.commit()
            self._broadcast(
                debate_id,
                {
                    "type": "analysis_failed",
                    "error": "ModernBERT tam tartışma analizi tamamlanamadı.",
                },
            )
            raise

        argument_map = await self.get_argument_map(db, debate_id)
        self._broadcast(
            debate_id,
            {
                "type": "argument_map_ready",
                "run": argument_map.get("run"),
                "impact": argument_map.get("impact"),
            },
        )
        return argument_map

    async def get_argument_map(self, db: AsyncSession, debate_id: int) -> dict:
        """Return the latest full ModernBERT argument map for a debate."""
        debate = await self.get_debate(db, debate_id)
        if not debate:
            raise ValueError("Tartışma bulunamadı")

        messages = list(debate.messages)
        baseline_nodes, baseline_edges = self._build_baseline_graph(messages)

        stmt = (
            select(AnalysisRun)
            .filter_by(
                debate_id=debate_id,
                status="completed",
                component_model=self._component_model_version(),
                relation_model=self._relation_model_version(),
            )
            .order_by(AnalysisRun.created_at.desc())
            .limit(1)
        )
        res = await db.execute(stmt)
        run = res.scalars().first()

        if not run:
            return {
                "status": "empty",
                "debate": debate.to_dict(),
                "run": None,
                "nodes": [],
                "edges": [],
                "relations": [],
                "baseline_nodes": baseline_nodes,
                "baseline_edges": baseline_edges,
                "annotations": {},
                "timeline": self._build_timeline(messages, [], []),
                "findings": [],
                "impact": self._empty_impact(len(baseline_edges)),
            }

        stmt_components = select(ArgumentComponentModel).filter_by(analysis_run_id=run.id).order_by(ArgumentComponentModel.message_id.asc(), ArgumentComponentModel.start_idx.asc())
        res_components = await db.execute(stmt_components)
        components = res_components.scalars().all()

        stmt_relations = select(ArgumentRelation).filter_by(analysis_run_id=run.id)
        res_relations = await db.execute(stmt_relations)
        relations = res_relations.scalars().all()

        graph_relations = self._select_graph_relations(relations, run)
        nodes = [self._component_node(component) for component in components]
        edges = [self._relation_edge(relation) for relation in graph_relations]
        relation_payloads = [self._relation_payload(relation) for relation in graph_relations]
        annotations = self._build_annotations(components, graph_relations)
        findings = self._build_findings(messages, components, relations, run)
        impact = self._build_impact(messages, relations, graph_relations, findings, len(baseline_edges))

        return {
            "status": "ready",
            "debate": debate.to_dict(),
            "run": run.to_dict(),
            "nodes": nodes,
            "edges": edges,
            "relations": relation_payloads,
            "baseline_nodes": baseline_nodes,
            "baseline_edges": baseline_edges,
            "annotations": annotations,
            "timeline": self._build_timeline(messages, components, graph_relations),
            "findings": findings,
            "impact": impact,
        }

    async def _classify_full_debate_relations(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        messages: list[Message],
        components_by_message: dict[int, list[ArgumentComponentModel]],
    ) -> None:
        """Classify every directed component pair in the debate.

        The source component is interpreted as supporting, attacking, or having no
        relation to the target component. The UI can then show sentence-to-sentence
        relations when a transcript span is selected.
        """
        message_by_id = {message.id: message for message in messages}
        all_components = [
            component
            for message in messages
            for component in components_by_message.get(message.id, [])
        ]

        jobs: list[tuple[ArgumentComponentModel, ArgumentComponentModel, int]] = []
        pairs: list[tuple[str, str]] = []
        for source_component in all_components:
            source_msg = message_by_id.get(source_component.message_id)
            if not source_msg:
                continue
            for target_component in all_components:
                if source_component.id == target_component.id:
                    continue
                target_msg = message_by_id.get(target_component.message_id)
                if not target_msg:
                    continue
                distance_turns = abs(
                    (source_msg.position or 0) - (target_msg.position or 0)
                )
                jobs.append((source_component, target_component, distance_turns))
                pairs.append((target_component.text, source_component.text))

        if hasattr(self.bert, "classify_relations_batch"):
            predictions = await asyncio.to_thread(self.bert.classify_relations_batch, pairs)
        else:
            predictions = [
                await asyncio.to_thread(self.bert.classify_relation, claim, evidence)
                for claim, evidence in pairs
            ]

        for (source_component, target_component, distance_turns), (
            relation_type,
            confidence,
            probabilities,
        ) in zip(jobs, predictions):
            relation = ArgumentRelation(
                analysis_run_id=run.id,
                source_component_id=source_component.id,
                target_component_id=target_component.id,
                relation_type=relation_type,
                confidence=confidence,
                probabilities_json=json.dumps(probabilities),
                distance_turns=distance_turns,
                is_long_range=distance_turns >= 2,
            )
            db.add(relation)

    @staticmethod
    def _ensure_claim_anchor(
        components: list,
        default_type: str,
        message_type: str,
    ) -> list:
        """Guarantee one claim anchor for claim-like turns when the model collapses to evidence."""
        if not components or default_type != "claim" or message_type not in ("claim", "rebuttal"):
            return components
        if any(component.component_type == "claim" for component in components):
            return components
        candidates = [component for component in components if component.component_type != "other"] or components
        anchor = max(candidates, key=lambda component: (component.confidence, len(component.text)))
        anchor.component_type = "claim"
        anchor.confidence = max(anchor.confidence, 0.72)
        return components

    def _top_relation_components(
        self,
        components: list[ArgumentComponentModel],
    ) -> list[ArgumentComponentModel]:
        """Keep relation classification focused on the clearest model spans."""
        return sorted(
            components,
            key=lambda component: (
                component.confidence,
                component.end_idx - component.start_idx,
            ),
            reverse=True,
        )[: self.MAX_RELATION_COMPONENTS_PER_MESSAGE]

    def _select_graph_relations(
        self,
        relations: list[ArgumentRelation],
        run: AnalysisRun,
    ) -> list[ArgumentRelation]:
        """Select the clearest, structurally plausible relations for visualization."""
        source_counts: defaultdict[tuple[int, str], int] = defaultdict(int)
        target_counts: defaultdict[tuple[int, str], int] = defaultdict(int)
        relation_type_counts: defaultdict[str, int] = defaultdict(int)
        selected: list[ArgumentRelation] = []

        local_supports = [
            relation
            for relation in relations
            if self._is_visible_graph_relation(relation, run)
            and relation.relation_type == "support"
            and relation.source_component.message_id == relation.target_component.message_id
        ]
        self._append_graph_relation_candidates(
            sorted(local_supports, key=lambda relation: relation.confidence, reverse=True),
            selected,
            source_counts,
            target_counts,
            relation_type_counts,
        )

        candidates = [
            relation
            for relation in relations
            if self._is_visible_graph_relation(relation, run)
            and relation not in selected
        ]
        candidates.sort(
            key=lambda relation: (
                relation.is_long_range,
                relation.confidence,
                relation.distance_turns,
            ),
            reverse=True,
        )

        self._append_graph_relation_candidates(
            candidates,
            selected,
            source_counts,
            target_counts,
            relation_type_counts,
        )

        return sorted(
            selected,
            key=lambda relation: (
                relation.source_component.message.position or 0,
                relation.target_component.message.position or 0,
                relation.relation_type,
            ),
        )

    def _append_graph_relation_candidates(
        self,
        candidates: list[ArgumentRelation],
        selected: list[ArgumentRelation],
        source_counts: defaultdict[tuple[int, str], int],
        target_counts: defaultdict[tuple[int, str], int],
        relation_type_counts: defaultdict[str, int],
    ) -> None:
        """Append relation candidates while preserving graph readability caps."""
        for relation in candidates:
            source_key = (relation.source_component_id, relation.relation_type)
            target_key = (relation.target_component_id, relation.relation_type)
            if source_counts[source_key] >= self.MAX_GRAPH_EDGES_PER_SOURCE_TYPE:
                continue
            if target_counts[target_key] >= self.MAX_GRAPH_EDGES_PER_TARGET_TYPE:
                continue
            if (
                relation.relation_type == "attack"
                and relation_type_counts["attack"] >= self.MAX_GRAPH_ATTACK_EDGES
            ):
                continue
            if (
                relation.relation_type == "support"
                and relation_type_counts["support"] >= self.MAX_GRAPH_SUPPORT_EDGES
            ):
                continue
            selected.append(relation)
            source_counts[source_key] += 1
            target_counts[target_key] += 1
            relation_type_counts[relation.relation_type] += 1
            if len(selected) >= self.MAX_GRAPH_EDGES:
                break

    @staticmethod
    def _is_visible_graph_relation(relation: ArgumentRelation, run: AnalysisRun) -> bool:
        """Filter noisy relation predictions before they become graph edges."""
        if relation.relation_type not in ("support", "attack"):
            return False

        source = relation.source_component
        target = relation.target_component
        if not source or not target:
            return False

        same_message = source.message_id == target.message_id
        same_agent = source.agent_id == target.agent_id

        if relation.relation_type == "attack":
            if relation.confidence < run.attack_threshold:
                return False
            if same_message:
                return source.component_type == "evidence" and target.component_type == "claim"
            if same_agent:
                return False
            return target.component_type == "claim"

        if relation.confidence < run.relation_threshold:
            return False
        if same_message:
            return source.component_type == "evidence" and target.component_type == "claim"
        return target.component_type == "claim"

    async def _update_full_debate_strengths(
        self,
        db: AsyncSession,
        run: AnalysisRun,
        messages: list[Message],
    ) -> None:
        """Update each message version with a model-only aggregate strength."""
        stmt_components = select(ArgumentComponentModel).filter_by(analysis_run_id=run.id)
        res_components = await db.execute(stmt_components)
        components = res_components.scalars().all()

        stmt_relations = select(ArgumentRelation).filter_by(analysis_run_id=run.id)
        res_relations = await db.execute(stmt_relations)
        relations = res_relations.scalars().all()

        components_by_message: dict[int, list[ArgumentComponentModel]] = defaultdict(list)
        relations_by_source_message: dict[int, list[ArgumentRelation]] = defaultdict(list)

        for component in components:
            components_by_message[component.message_id].append(component)
        for relation in relations:
            relations_by_source_message[relation.source_component.message_id].append(relation)

        for msg in messages:
            if not msg.current_version:
                continue
            msg.current_version.strength_score = self._aggregate_model_strength(
                components_by_message.get(msg.id, []),
                relations_by_source_message.get(msg.id, []),
                run,
            )

    @staticmethod
    def _top_actionable_pair_relation(relations: list) -> object | None:
        """Return the strongest relation that should be exposed as pair verdict."""
        actionable = [
            relation
            for relation in relations
            if (
                relation.relation_type == "support"
                and relation.confidence >= DebateService.RELATION_GRAPH_THRESHOLD
            )
            or (
                relation.relation_type == "attack"
                and relation.confidence >= DebateService.ATTACK_RESPONSE_THRESHOLD
            )
        ]
        if not actionable:
            return None
        return max(actionable, key=lambda relation: relation.confidence)

    @staticmethod
    def _aggregate_model_strength(
        components: list[ArgumentComponentModel],
        relations: list[ArgumentRelation],
        run: AnalysisRun | None = None,
    ) -> float:
        """Aggregate strength from filtered, visible ModernBERT signals only."""
        component_score = (
            sum(component.confidence for component in components) / len(components)
            if components
            else 0.0
        )

        visible_relations = [
            relation
            for relation in relations
            if run is None or DebateService._is_visible_graph_relation(relation, run)
        ]

        if not visible_relations:
            return round(max(0.0, min(1.0, component_score * 0.5)), 2)

        non_none = [
            relation
            for relation in visible_relations
            if relation.relation_type in ("support", "attack")
        ]
        if non_none:
            relation_score = sum(r.confidence for r in non_none) / len(non_none)
        else:
            none_score = sum(r.confidence for r in visible_relations) / len(visible_relations)
            relation_score = 1.0 - none_score

        score = (component_score * 0.35) + (relation_score * 0.65)
        return round(max(0.0, min(1.0, score)), 2)

    @staticmethod
    def _default_component_type(message_type: str) -> str:
        if message_type in ("claim", "rebuttal"):
            return "claim"
        return "evidence"

    def _component_node(self, component: ArgumentComponentModel) -> dict:
        msg = component.message
        round_number = ((msg.position or 0) // 2) + 1
        label_type = {
            "claim": "Claim",
            "evidence": "Evidence",
            "other": "Other",
        }.get(component.component_type, component.component_type.title())
        agent_name = component.agent.name if component.agent else "Sistem"

        return {
            "data": {
                "id": f"c-{component.id}",
                "messageId": component.message_id,
                "agentId": component.agent_id,
                "agentName": agent_name,
                "text": component.text,
                "kind": component.component_type,
                "round": round_number,
                "labelType": label_type,
                "stance": self._component_stance(component),
                "confidence": component.confidence,
                "topicRelationType": component.topic_relation_type,
                "topicRelationConfidence": component.topic_relation_confidence,
            }
        }

    @staticmethod
    def _relation_edge(relation: ArgumentRelation) -> dict:
        return {
            "data": {
                "id": f"r-{relation.id}",
                "source": f"c-{relation.source_component_id}",
                "target": f"c-{relation.target_component_id}",
                "relationType": relation.relation_type,
                "confidence": relation.confidence,
                "distanceTurns": relation.distance_turns,
                "isLongRange": relation.is_long_range,
            }
        }

    @staticmethod
    def _relation_payload(relation: ArgumentRelation) -> dict:
        """Return a rich relation payload for transcript selection."""
        try:
            probabilities = json.loads(relation.probabilities_json or "{}")
        except json.JSONDecodeError:
            probabilities = {}
        source = relation.source_component
        target = relation.target_component
        return {
            "id": relation.id,
            "source": f"c-{relation.source_component_id}",
            "target": f"c-{relation.target_component_id}",
            "source_component_id": relation.source_component_id,
            "target_component_id": relation.target_component_id,
            "source_message_id": source.message_id if source else None,
            "target_message_id": target.message_id if target else None,
            "relationType": relation.relation_type,
            "confidence": relation.confidence,
            "probabilities": probabilities,
            "distanceTurns": relation.distance_turns,
            "isLongRange": relation.is_long_range,
        }

    def _build_annotations(
        self,
        components: list[ArgumentComponentModel],
        graph_relations: list[ArgumentRelation],
    ) -> dict[str, list[dict]]:
        annotations: dict[str, list[dict]] = defaultdict(list)
        dominant_relation_by_component = self._dominant_relation_context(graph_relations)
        for component in components:
            relation_context = dominant_relation_by_component.get(component.id)
            annotations[str(component.message_id)].append(
                {
                    "id": component.id,
                    "type": component.component_type,
                    "start": component.start_idx,
                    "end": component.end_idx,
                    "confidence": component.confidence,
                    "text": component.text,
                    "stance": self._component_stance(component),
                    "relation": relation_context,
                    "topic_relation_type": component.topic_relation_type,
                    "topic_relation_confidence": component.topic_relation_confidence,
                }
            )
        return dict(annotations)

    @staticmethod
    def _component_stance(component: ArgumentComponentModel) -> str:
        """Return whether the component belongs to the topic-supporting or opposing side."""
        msg = component.message
        agent_role = (component.agent.role if component.agent else "") or ""
        message_type = (msg.message_type if msg else "") or ""
        if agent_role == "opponent" or message_type == "attack":
            return "con"
        return "pro"

    @staticmethod
    def _dominant_relation_context(
        graph_relations: list[ArgumentRelation],
    ) -> dict[int, dict]:
        """Return the strongest visible relation context for transcript tooltips."""
        contexts: dict[int, dict] = {}
        for relation in sorted(graph_relations, key=lambda rel: rel.confidence, reverse=True):
            relation_verb = "çürütüyor" if relation.relation_type == "attack" else "destekliyor"
            inverse_verb = "çürütülüyor" if relation.relation_type == "attack" else "destekleniyor"
            source_msg = relation.source_component.message
            target_msg = relation.target_component.message
            source_round = ((source_msg.position or 0) // 2) + 1
            target_round = ((target_msg.position or 0) // 2) + 1
            target_preview = relation.target_component.text[:120]
            source_preview = relation.source_component.text[:120]

            contexts.setdefault(
                relation.source_component_id,
                {
                    "role": "source",
                    "relation_type": relation.relation_type,
                    "confidence": relation.confidence,
                    "text": (
                        f"Bu parça Tur {target_round} iddiasını %{relation.confidence * 100:.0f} "
                        f"güvenle {relation_verb}: {target_preview}"
                    ),
                    "target_component_id": relation.target_component_id,
                    "target_message_id": relation.target_component.message_id,
                },
            )
            contexts.setdefault(
                relation.target_component_id,
                {
                    "role": "target",
                    "relation_type": relation.relation_type,
                    "confidence": relation.confidence,
                    "text": (
                        f"Bu iddia Tur {source_round} yanıtı tarafından %{relation.confidence * 100:.0f} "
                        f"güvenle {inverse_verb}: {source_preview}"
                    ),
                    "source_component_id": relation.source_component_id,
                    "source_message_id": relation.source_component.message_id,
                },
            )
        return contexts

    @staticmethod
    def _build_baseline_graph(messages: list[Message]) -> tuple[list[dict], list[dict]]:
        nodes = []
        edges = []
        for msg in messages:
            cv = msg.current_version
            txt = cv.content if cv else ""
            role_name = msg.agent.name if msg.agent else "Sistem"
            round_num = ((msg.position or 0) // 2) + 1
            nodes.append({
                "data": {
                    "id": f"msg-{msg.id}",
                    "text": f"{role_name}: {txt[:80]}...",
                    "kind": msg.message_type,
                    "round": round_num,
                    "labelType": msg.message_type.upper(),
                }
            })
            if msg.parent_id:
                edges.append({
                    "data": {
                        "id": f"edge-msg-{msg.id}",
                        "source": f"msg-{msg.id}",
                        "target": f"msg-{msg.parent_id}",
                        "relationType": "baseline",
                    }
                })
        return nodes, edges

    @staticmethod
    def _build_timeline(
        messages: list[Message],
        components: list[ArgumentComponentModel],
        relations: list[ArgumentRelation],
    ) -> list[dict]:
        timeline = []
        components_by_msg = defaultdict(list)
        for c in components:
            components_by_msg[c.message_id].append(c)

        incoming_by_msg = defaultdict(int)
        outgoing_by_msg = defaultdict(int)
        for r in relations:
            outgoing_by_msg[r.source_component.message_id] += 1
            incoming_by_msg[r.target_component.message_id] += 1

        for msg in messages:
            if not msg.agent:
                continue
            round_num = ((msg.position or 0) // 2) + 1
            timeline.append({
                "message_id": msg.id,
                "round": round_num,
                "agent_name": msg.agent.name,
                "component_count": len(components_by_msg[msg.id]),
                "incoming_count": incoming_by_msg[msg.id],
                "outgoing_count": outgoing_by_msg[msg.id],
            })
        return timeline

    def _build_findings(
        self,
        messages: list[Message],
        components: list[ArgumentComponentModel],
        relations: list[ArgumentRelation],
        run: AnalysisRun,
    ) -> list[dict]:
        findings = []
        components_by_msg = defaultdict(list)
        for c in components:
            components_by_msg[c.message_id].append(c)

        msg_by_id = {msg.id: msg for msg in messages}
        source_relations = defaultdict(list)
        target_relations = defaultdict(list)
        for r in relations:
            source_relations[r.source_component_id].append(r)
            target_relations[r.target_component_id].append(r)

        # 1. Unsupported Claims
        for component in components:
            if component.component_type != "claim":
                continue
            visible_supports = [
                r for r in target_relations[component.id]
                if r.relation_type == "support" and self._is_visible_graph_relation(r, run)
            ]
            if not visible_supports:
                msg = msg_by_id.get(component.message_id)
                round_num = ((msg.position or 0) // 2) + 1 if msg else 1
                agent_name = component.agent.name if component.agent else "Bilinmeyen"
                findings.append({
                    "type": "unsupported_claim",
                    "title": "Temelsiz İddia",
                    "message_id": component.message_id,
                    "detail": (
                        f"Tur {round_num} içinde {agent_name} tarafından öne sürülen "
                        f"'{component.text[:80]}...' iddiası için ModernBERT pipeline'ı "
                        f"hiçbir destekleyici kanıt bulamadı."
                    ),
                    "confidence": 1.0 - (component.confidence * 0.5),
                    "severity": "medium",
                })

        # 2. Missed Rebuttals
        for msg in messages:
            if msg.message_type != "attack" or not msg.parent_id:
                continue
            parent = msg_by_id.get(msg.parent_id)
            if not parent:
                continue
            parent_claims = [c for c in components_by_msg[parent.id] if c.component_type == "claim"]
            attack_evidences = [c for c in components_by_msg[msg.id] if c.component_type == "evidence"]
            
            for pc in parent_claims:
                has_attack = False
                for ae in attack_evidences:
                    attacks = [
                        r for r in relations
                        if r.source_component_id == ae.id
                        and r.target_component_id == pc.id
                        and r.relation_type == "attack"
                        and self._is_visible_graph_relation(r, run)
                    ]
                    if attacks:
                        has_attack = True
                        break
                if not has_attack and parent_claims:
                    round_num = ((msg.position or 0) // 2) + 1
                    agent_name = msg.agent.name if msg.agent else "Bilinmeyen"
                    findings.append({
                        "type": "missed_rebuttal",
                        "title": "Iskalanan Karşı Tez",
                        "message_id": msg.id,
                        "detail": (
                            f"Tur {round_num} saldırısında {agent_name}, karşı tarafın "
                            f"'{pc.text[:80]}...' iddiasını doğrudan hedef alan yapısal "
                            f"bir çürütme sunamadı."
                        ),
                        "confidence": 0.72,
                        "severity": "high",
                    })

        # 3. None Escapes
        for msg in messages:
            if msg.message_type not in ("attack", "rebuttal"):
                continue
            msg_components = components_by_msg[msg.id]
            if not msg_components:
                continue
            has_relations = False
            for c in msg_components:
                if source_relations[c.id] or target_relations[c.id]:
                    has_relations = True
                    break
            if not has_relations:
                round_num = ((msg.position or 0) // 2) + 1
                agent_name = msg.agent.name if msg.agent else "Bilinmeyen"
                findings.append({
                    "type": "none_escape",
                    "title": "Boş Tur",
                    "message_id": msg.id,
                    "detail": (
                        f"Tur {round_num} içinde {agent_name} tarafından sunulan metin, "
                        f"tartışmanın geri kalanındaki hiçbir iddia veya kanıt ile ModernBERT "
                        f"tarafından ilişkilendirilemedi (None kaçışı)."
                    ),
                    "confidence": 0.85,
                    "severity": "low",
                })

        return self._limit_findings(findings)

    def _limit_findings(self, findings: list[dict]) -> list[dict]:
        type_counts = defaultdict(int)
        limited = []
        severity_rank = {"high": 3, "medium": 2, "low": 1}
        for finding in sorted(findings, key=lambda f: severity_rank.get(f["severity"], 0), reverse=True):
            t = finding["type"]
            if type_counts[t] < self.MAX_FINDINGS_PER_TYPE:
                limited.append(finding)
                type_counts[t] += 1

        return sorted(
            limited,
            key=lambda finding: (
                severity_rank.get(finding.get("severity"), 0),
                finding.get("confidence") or 0.0,
            ),
            reverse=True,
        )

    @staticmethod
    def _build_impact(
        messages: list[Message],
        all_relations: list[ArgumentRelation],
        graph_relations: list[ArgumentRelation],
        findings: list[dict],
        baseline_edge_count: int,
    ) -> dict:
        message_by_id = {msg.id: msg for msg in messages}
        bert_only_edges = 0
        for relation in graph_relations:
            source_message = message_by_id.get(relation.source_component.message_id)
            if not source_message or source_message.parent_id != relation.target_component.message_id:
                bert_only_edges += 1

        raw_count = len(all_relations)
        visible_count = len(graph_relations)
        none_count = sum(1 for relation in all_relations if relation.relation_type == "none")
        filtered_count = max(0, raw_count - visible_count)

        return {
            "semantic_edges": len(graph_relations),
            "raw_relations": raw_count,
            "visible_relations": visible_count,
            "none_relations": none_count,
            "filtered_relations": filtered_count,
            "filter_reasons": DebateService._relation_filter_reasons(all_relations, graph_relations),
            "baseline_edges": baseline_edge_count,
            "bert_only_edges": bert_only_edges,
            "long_range_edges": sum(1 for relation in graph_relations if relation.is_long_range),
            "unsupported_claims": sum(
                1 for finding in findings if finding["type"] == "unsupported_claim"
            ),
            "missed_rebuttals": sum(
                1 for finding in findings if finding["type"] == "missed_rebuttal"
            ),
            "none_escapes": sum(
                1 for finding in findings if finding["type"] == "none_escape"
            ),
        }

    @staticmethod
    def _empty_impact(baseline_edge_count: int) -> dict:
        return {
            "semantic_edges": 0,
            "raw_relations": 0,
            "visible_relations": 0,
            "none_relations": 0,
            "filtered_relations": 0,
            "filter_reasons": {},
            "baseline_edges": baseline_edge_count,
            "bert_only_edges": 0,
            "long_range_edges": 0,
            "unsupported_claims": 0,
            "missed_rebuttals": 0,
            "none_escapes": 0,
        }

    @staticmethod
    def _relation_filter_reasons(
        all_relations: list[ArgumentRelation],
        graph_relations: list[ArgumentRelation],
    ) -> dict:
        visible_ids = {relation.id for relation in graph_relations}
        reasons: defaultdict[str, int] = defaultdict(int)
        for relation in all_relations:
            if relation.id in visible_ids:
                continue
            if relation.relation_type == "none":
                reasons["none_relation"] += 1
                continue
            source = relation.source_component
            target = relation.target_component
            if target and target.component_type != "claim":
                reasons["target_not_claim"] += 1
            elif relation.relation_type == "attack" and relation.confidence < DebateService.ATTACK_RESPONSE_THRESHOLD:
                reasons["below_attack_threshold"] += 1
            elif relation.relation_type == "support" and relation.confidence < DebateService.RELATION_GRAPH_THRESHOLD:
                reasons["below_support_threshold"] += 1
            elif source and target and source.agent_id == target.agent_id and source.message_id != target.message_id:
                reasons["same_agent_cross_message"] += 1
            else:
                reasons["graph_cap_or_structural_filter"] += 1
        return dict(reasons)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _post_message(
        self,
        db: AsyncSession,
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
        db.add(msg)
        await db.flush()

        version = MessageVersion(
            message_id=msg.id,
            version_number=1,
            content=content,
            is_current=True,
        )
        db.add(version)
        await db.commit()

        # Load the relationships so that msg.to_dict() won't fail or trigger lazy load
        await db.refresh(msg)

        self._broadcast(
            debate_id,
            {
                "type": "message",
                "message": msg.to_dict(),
            },
        )
        return msg

    async def _evaluate_pair_if_ready(self, db: AsyncSession, debate_id: int) -> None:
        """After every pair of messages, run the ModernBERT pipeline and trigger regeneration if strength is low."""
        debate = await self.get_debate(db, debate_id)
        if not debate:
            return

        messages = list(debate.messages)
        if len(messages) < 2:
            return

        # We evaluate the *latest* message against the one it replies to.
        latest_msg = messages[-1]
        if not latest_msg.parent_id:
            return

        stmt_parent = select(Message).filter_by(id=latest_msg.parent_id)
        res_parent = await db.execute(stmt_parent)
        parent_msg = res_parent.scalar_one_or_none()
        if not parent_msg or not parent_msg.current_version or not latest_msg.current_version:
            return

        # Loop to regenerate if strength is low, up to version 3
        while latest_msg.current_version.version_number < 3:
            current_version = latest_msg.current_version

            # Run ModernBERT pipeline on current_version content vs parent content
            result = await asyncio.to_thread(
                self.bert.analyze,
                source_text=current_version.content,
                target_text=parent_msg.current_version.content,
                source_type="evidence",
                target_type="claim",
            )

            # Store analysis
            top_relation = self._top_actionable_pair_relation(result.relations)
            analysis = Analysis(
                message_version_id=current_version.id,
                target_message_version_id=parent_msg.current_version.id,
                component_type="evidence",
                relation_type=top_relation.relation_type if top_relation else "none",
                confidence=top_relation.confidence if top_relation else 0.0,
                feedback_text=result.feedback,
            )
            db.add(analysis)

            # Update strength score on the version
            current_version.strength_score = result.overall_strength
            await db.commit()

            self._broadcast(
                debate_id,
                {
                    "type": "analysis",
                    "analysis": analysis.to_dict(),
                    "message_id": latest_msg.id,
                    "strength": result.overall_strength,
                },
            )

            # Threshold for triggering regeneration: strength < 0.60
            REGEN_THRESHOLD = 0.60
            if result.overall_strength >= REGEN_THRESHOLD:
                # Strong enough! Exit the loop.
                break

            # If it's too weak, regenerate!
            next_version_num = current_version.version_number + 1
            self._broadcast(
                debate_id,
                {
                    "type": "message_regeneration_started",
                    "message_id": latest_msg.id,
                    "version_number": next_version_num,
                    "feedback": result.feedback,
                },
            )

            # Get agent config
            agent_config = latest_msg.agent
            if not agent_config:
                break

            # The context for regenerating is all previous messages excluding this latest message itself
            context_messages = [m for m in messages if m.id != latest_msg.id]

            # Generate new response with feedback
            new_content = await self.agent_svc.generate_response(
                agent_config=agent_config,
                topic=debate.topic,
                messages=context_messages,
                feedback=result.feedback,
            )

            # Deactivate current version
            current_version.is_current = False

            # Create new version
            new_version = MessageVersion(
                message_id=latest_msg.id,
                version_number=next_version_num,
                content=new_content,
                is_current=True,
            )
            db.add(new_version)
            await db.commit()

            # Refresh to associate new version
            await db.refresh(latest_msg)

            # Broadcast message version created
            self._broadcast(
                debate_id,
                {
                    "type": "message_version_created",
                    "message_id": latest_msg.id,
                    "version": new_version.to_dict(),
                    "agent_name": latest_msg.agent.name if latest_msg.agent else "Sistem",
                    "message_type": latest_msg.message_type,
                    "position": latest_msg.position,
                    "previous_strength": current_version.strength_score,
                    "feedback": result.feedback,
                },
            )

    async def toggle_auto_advance(self, db: AsyncSession, debate_id: int, enabled: bool) -> Debate:
        """Toggle auto-advance mode for a debate."""
        debate = await self.get_debate(db, debate_id)
        if not debate:
            raise ValueError("Tartışma bulunamadı")
        debate.auto_advance = enabled
        await db.commit()
        self._broadcast(
            debate_id,
            {"type": "auto_advance_toggled", "enabled": enabled},
        )
        return debate

    async def run_full_debate(self, db: AsyncSession, debate_id: int) -> None:
        """Run the entire debate automatically until resolution.

        Enables auto_advance and triggers turns until max rounds are reached.
        """
        debate = await self.get_debate(db, debate_id)
        if not debate:
            raise ValueError("Tartışma bulunamadı")

        if debate.status == "pending":
            await self.start_debate(db, debate_id)

        debate.auto_advance = True
        await db.commit()

        # Advance until resolved or error
        max_iterations = debate.max_rounds * 2 + 2
        for _ in range(max_iterations):
            await db.refresh(debate)
            if debate.status != "active":
                break
            await self.advance_debate(db, debate_id)
            # Give a small delay to allow SSE events to propagate and model inference
            await asyncio.sleep(0.5)

    async def _advance_if_needed(self, db: AsyncSession, debate_id: int) -> None:
        """Auto-advance the debate if conditions are met.

        When auto_advance is enabled, automatically triggers the next turn
        unless max rounds have been reached.
        """
        debate = await self.get_debate(db, debate_id)
        if not debate or debate.status not in ("active",):
            return

        messages = list(debate.messages)
        total_messages = len(messages)

        # Resolve if max rounds reached (each round = 2 messages)
        if total_messages >= debate.max_rounds * 2:
            debate.status = "resolved"
            await db.commit()
            self._broadcast(debate_id, {"type": "status", "status": "resolved"})
            await self.analyze_debate(db, debate_id)
            return

        # Update round counter: claim starts a round, attack completes it
        debate.current_round = (total_messages + 1) // 2
        await db.commit()

        # Auto-advance if enabled
        if debate.auto_advance and total_messages > 0 and total_messages < debate.max_rounds * 2:
            # Small delay to allow frontend to render and SSE to propagate
            await asyncio.sleep(0.3)
            await self.advance_debate(db, debate_id)

    async def regenerate_message(self, db: AsyncSession, debate_id: int, message_id: int) -> dict:
        """Manually trigger regeneration for a specific message using ModernBERT feedback."""
        from sqlalchemy import select
        from app.models import Message, MessageVersion, Analysis

        debate = await self.get_debate(db, debate_id)
        if not debate:
            raise ValueError("Tartışma bulunamadı")

        stmt = select(Message).filter_by(id=message_id, debate_id=debate_id)
        res = await db.execute(stmt)
        msg = res.scalar_one_or_none()
        if not msg:
            raise ValueError("Mesaj bulunamadı")

        if not msg.current_version:
            raise ValueError("Mesaj sürümü bulunamadı")

        current_version = msg.current_version
        next_version_num = current_version.version_number + 1

        # Determine parent or fallback to debate topic
        parent_content = debate.topic
        parent_version_id = None
        if msg.parent_id:
            stmt_parent = select(Message).filter_by(id=msg.parent_id)
            res_parent = await db.execute(stmt_parent)
            parent_msg = res_parent.scalar_one_or_none()
            if parent_msg and parent_msg.current_version:
                parent_content = parent_msg.current_version.content
                parent_version_id = parent_msg.current_version.id

        # Broadcast regeneration started
        self._broadcast(
            debate_id,
            {
                "type": "message_regeneration_started",
                "message_id": msg.id,
                "version_number": next_version_num,
                "feedback": "ModernBERT analizi başlatıldı...",
            },
        )

        # Run ModernBERT pipeline on current_version content vs parent content
        result = await asyncio.to_thread(
            self.bert.analyze,
            source_text=current_version.content,
            target_text=parent_content,
            source_type="evidence" if msg.parent_id else "claim",
            target_type="claim" if msg.parent_id else "topic",
        )

        # Store analysis
        top_relation = self._top_actionable_pair_relation(result.relations)
        analysis = Analysis(
            message_version_id=current_version.id,
            target_message_version_id=parent_version_id,
            component_type="evidence" if msg.parent_id else "claim",
            relation_type=top_relation.relation_type if top_relation else "none",
            confidence=top_relation.confidence if top_relation else 0.0,
            feedback_text=result.feedback,
        )
        db.add(analysis)

        # Update current version strength
        current_version.strength_score = result.overall_strength
        await db.commit()

        self._broadcast(
            debate_id,
            {
                "type": "analysis",
                "analysis": analysis.to_dict(),
                "message_id": msg.id,
                "strength": result.overall_strength,
            },
        )

        # Get agent config
        agent_config = msg.agent
        if not agent_config:
            raise ValueError("Mesaj ajanı bulunamadı")

        # Get debate messages context (all prior messages in the debate)
        messages_list = list(debate.messages)
        context_messages = [m for m in messages_list if m.position < msg.position]

        # Generate new response with feedback
        new_content = await self.agent_svc.generate_response(
            agent_config=agent_config,
            topic=debate.topic,
            messages=context_messages,
            feedback=result.feedback,
        )

        # Deactivate current version
        current_version.is_current = False

        # Create new version
        new_version = MessageVersion(
            message_id=msg.id,
            version_number=next_version_num,
            content=new_content,
            is_current=True,
            strength_score=None,
        )
        db.add(new_version)
        await db.commit()

        # Refresh to associate new version
        await db.refresh(msg)

        # Broadcast message version created
        self._broadcast(
            debate_id,
            {
                "type": "message_version_created",
                "message_id": msg.id,
                "version": new_version.to_dict(),
                "agent_name": msg.agent.name if msg.agent else "Sistem",
                "message_type": msg.message_type,
                "position": msg.position,
                "previous_strength": current_version.strength_score,
                "feedback": result.feedback,
            },
        )

        return {
            "success": True,
            "version": new_version.to_dict(),
            "feedback": result.feedback,
            "strength": result.overall_strength,
        }

    def _broadcast(self, debate_id: int, event: dict) -> None:
        EventBroker.publish(debate_id, event)
