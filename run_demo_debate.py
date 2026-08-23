"""Throwaway runner: start a random 3-round debate and dump ModernBERT labels."""

import asyncio
import random

from sqlalchemy import select

from app.database import init_db, async_session_factory
from app.models import (
    Agent,
    Debate,
    AnalysisRun,
    ArgumentComponent as ArgumentComponentModel,
    ArgumentRelation,
    Message,
)
from app.services.debate_service import DebateService

TOPICS = [
    "Yapay zeka uzun vadede insan istihdamını azaltacaktır.",
    "Sosyal medya demokrasiye fayda yerine zarar veriyor.",
    "Uzaktan çalışma şirket verimliliğini artırır.",
    "Nükleer enerji iklim kriziyle mücadelede vazgeçilmezdir.",
    "Üniversite eğitimi herkes için ücretsiz olmalıdır.",
]


async def main() -> None:
    await init_db()
    topic = random.choice(TOPICS)
    print(f"\n=== KONU: {topic} ===\n")

    svc = DebateService()
    async with async_session_factory() as db:
        agents = (await db.execute(select(Agent).order_by(Agent.id))).scalars().all()
        agent_ids = [agents[0].id, agents[1].id]
        debate = await svc.create_debate(db, topic=topic, agent_ids=agent_ids, max_rounds=3)
        debate_id = debate.id

    print(f"Debate #{debate_id} oluşturuldu (3 tur). LLM cevapları üretiliyor + BERT analiz...\n")
    async with async_session_factory() as db:
        await svc.run_full_debate(db, debate_id)

    # Make sure final analysis exists
    async with async_session_factory() as db:
        await svc.analyze_debate(db, debate_id)

    # ---- Dump results ----
    async with async_session_factory() as db:
        msgs = (
            await db.execute(
                select(Message).where(Message.debate_id == debate_id).order_by(Message.position)
            )
        ).scalars().all()

        run = (
            await db.execute(
                select(AnalysisRun)
                .where(AnalysisRun.debate_id == debate_id)
                .order_by(AnalysisRun.id.desc())
            )
        ).scalars().first()

        comps = (
            await db.execute(
                select(ArgumentComponentModel)
                .where(ArgumentComponentModel.analysis_run_id == run.id)
                .order_by(ArgumentComponentModel.id)
            )
        ).scalars().all()

        rels = (
            await db.execute(
                select(ArgumentRelation)
                .where(ArgumentRelation.analysis_run_id == run.id)
                .order_by(ArgumentRelation.id)
            )
        ).scalars().all()

        agents_by_id = {a.id: a.name for a in (await db.execute(select(Agent))).scalars().all()}

        print("\n========== MESAJLAR ==========")
        for m in msgs:
            v = m.current_version
            content = v.content if v else ""
            print(f"\n[pos {m.position}] {agents_by_id.get(m.agent_id)} ({m.message_type}):")
            print(f"  {content}")

        comps_by_id = {c.id: c for c in comps}
        print("\n\n========== BERT BİLEŞENLERİ (component + konu ilişkisi) ==========")
        for c in comps:
            tr = f"{c.topic_relation_type} ({c.topic_relation_confidence:.2f})" if c.topic_relation_type else "-"
            print(
                f"\n#{c.id} [{c.component_type.upper()}] conf={c.confidence:.2f} | konu-ilişkisi: {tr}"
                f"\n   agent={agents_by_id.get(c.agent_id)}"
                f"\n   text: {c.text}"
            )

        print("\n\n========== BERT İLİŞKİLERİ (component <-> component) ==========")
        if not rels:
            print("  (graf ilişkisi yok)")
        for r in rels:
            s = comps_by_id.get(r.source_component_id)
            t = comps_by_id.get(r.target_component_id)
            st = (s.text[:60] + "…") if s else "?"
            tt = (t.text[:60] + "…") if t else "?"
            print(
                f"\n  [{r.relation_type.upper()}] conf={r.confidence:.2f} dist={r.distance_turns}"
                f"\n    KAYNAK #{r.source_component_id}: {st}"
                f"\n    HEDEF  #{r.target_component_id}: {tt}"
            )

        print(f"\n\nÖzet: {len(msgs)} mesaj, {len(comps)} bileşen, {len(rels)} ilişki. Run #{run.id} status={run.status}")


if __name__ == "__main__":
    asyncio.run(main())
