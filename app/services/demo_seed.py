"""Idempotent seeding of a fully-analyzed demo debate + fact-check run.

Demo mode is a presentation toggle: when active, the *real* app pages render
this pre-baked, perfectly-analyzed data. Nothing here calls ModernBERT, an LLM,
or any external API — the analysis is hand-authored so a live demo never depends
on inference latency or flaky services.

Component/relation offsets are exact because each message/article text is built
by concatenating its labeled spans (see `_assemble`).
"""

from __future__ import annotations

import datetime
import json

from sqlalchemy import delete, select

from app.models import (
    Agent,
    AnalysisRun,
    ArgumentComponent,
    ArgumentRelation,
    Debate,
    DebateParticipant,
    FactCheckRun,
    FactClaim,
    FactEvidence,
    FactRelation,
    Message,
    MessageVersion,
)

DEMO_DEBATE_TOPIC = "Sosyal medya, sağlıklı bir demokrasiye fayda yerine zarar veriyor."
DEMO_FACT_TOKEN = "demo-teyit-2026"


def _assemble(spans: list[dict]) -> tuple[str, list[dict]]:
    """Concatenate span texts into one body and record exact char offsets."""
    text = ""
    out = []
    for s in spans:
        seg = s["t"]
        start = len(text)
        text += seg
        out.append({**s, "start": start, "end": len(text)})
    return text, out


# ------------------------------------------------------------------
# DEBATE — 3 rounds (6 messages). Proponent (pro) argues FOR the topic
# (social media harms democracy) → his spans relate to the topic as "support".
# Opponent (con) argues against → his spans relate to the topic as "attack".
# Cited studies/data = premise; assertions/reasoning = claim; rhetoric = other.
# ------------------------------------------------------------------
DEBATE_MESSAGES = [
    {
        "side": "pro", "type": "claim", "strength": 0.95,
        "spans": [
            {"t": "Öncelikle çerçeveyi netleştirelim: ", "k": "other", "tr": "none", "c": 0.82},
            {"t": "değerlendirdiğimiz şey sosyal medyanın varlığı değil, etkileşimi azamiye çıkaran algoritmik mimarisidir. ", "k": "other", "tr": "none", "c": 0.8},
            {"t": "Sosyal medya, bilgiye dayalı kamusal müzakereyi sistematik olarak baltalar; ", "k": "claim", "tr": "support", "c": 0.97},
            {"t": "çünkü algoritmalar ölçülü ve doğrulanmış içeriği değil, öfke uyandıran ve kutuplaştırıcı içeriği öne çıkarır. ", "k": "premise", "tr": "support", "c": 0.95},
            {"t": "Nitekim Science'ta 2018'de yayımlanan ve 126 bin haber zincirini inceleyen bir çalışma, yalan haberlerin doğru haberlere kıyasla yüzde 70 daha fazla paylaşıldığını ortaya koymuştur. ", "k": "premise", "tr": "support", "c": 0.99},
            {"t": "Dolayısıyla, ", "k": "other", "tr": "none", "c": 0.78},
            {"t": "bu hız asimetrisi tesadüfi bir yan etki değil, doğrudan iş modelinin sonucudur. ", "k": "claim", "tr": "support", "c": 0.93},
            {"t": "Sonuç olarak seçim dönemlerinde demokratik irade, bilinçli tercihten çok algoritmik maruziyetle şekillenir.", "k": "claim", "tr": "support", "c": 0.92},
        ],
    },
    {
        "side": "con", "type": "attack", "strength": 0.86,
        "spans": [
            {"t": "Bu tablo teknik olarak kısmen doğru, ", "k": "other", "tr": "none", "c": 0.74},
            {"t": "ancak eksik bir resim sunuyor. ", "k": "claim", "tr": "attack", "c": 0.85},
            {"t": "Sosyal medyayı yalnızca dezenformasyon hızlandırıcısı olarak görmek, onun hesap verebilirliği ve şeffaflığı artıran kapasitesini yok sayar. ", "k": "claim", "tr": "attack", "c": 0.95},
            {"t": "Çünkü aynı platformlar, devlet baskısına uğrayan haberlerin milyonlara ulaşmasını sağlar; ", "k": "premise", "tr": "attack", "c": 0.9},
            {"t": "örneğin polis şiddetinin ve seçim usulsüzlüklerinin vatandaşlarca kaydedilip yayılması, demokratik denetimi fiilen güçlendirmiştir. ", "k": "premise", "tr": "attack", "c": 0.92},
            {"t": "Üstelik 2020'de Journal of Communication'da yayımlanan deneysel bir araştırma, yalan habere yönelmenin platform yapısından çok bireysel ön yargılarla ilişkili olduğunu göstermiştir. ", "k": "premise", "tr": "attack", "c": 0.91},
            {"t": "Elbette her teknolojinin riski vardır; ", "k": "other", "tr": "none", "c": 0.71},
            {"t": "ne var ki riski aracın özüne atfetmek ile kullanım bağlamına atfetmek apayrı şeylerdir. ", "k": "other", "tr": "none", "c": 0.72},
            {"t": "Dolayısıyla sorunun kaynağı araç değil, medya okuryazarlığı eksikliği ve siyasi kutuplaşmadır.", "k": "claim", "tr": "attack", "c": 0.88},
        ],
    },
    {
        "side": "pro", "type": "rebuttal", "strength": 0.91,
        "spans": [
            {"t": "Rakibimin denetim örneklerini memnuniyetle kabul ediyorum, ", "k": "other", "tr": "none", "c": 0.7},
            {"t": "çünkü tam da bu örnekler asimetriyi kanıtlıyor. ", "k": "claim", "tr": "support", "c": 0.88},
            {"t": "Sosyal medyanın denetim aracı olabildiği doğrudur; ", "k": "claim", "tr": "support", "c": 0.86},
            {"t": "ancak platformlar doğrulanmış bilgi ile manipülasyonu eşit yarıştırmaz, ikincisini yapısal olarak kayırır. ", "k": "claim", "tr": "support", "c": 0.92},
            {"t": "2021'de Nature Human Behaviour'da yayımlanan bir araştırma, ahlaki öfke içeren paylaşımların erişiminin nötr paylaşımlara göre belirgin biçimde arttığını göstermiştir. ", "k": "premise", "tr": "support", "c": 0.99},
            {"t": "Arap Baharı gibi örnekler kısa vadeli istisnalardır; ", "k": "claim", "tr": "support", "c": 0.84},
            {"t": "zira aynı platformlar kısa sürede gözetim ve hedefli dezenformasyon araçlarına dönüşmüştür. ", "k": "premise", "tr": "support", "c": 0.9},
            {"t": "Sorumluluğu bütünüyle bireye yıkan okuryazarlık savı ise, öfkeyi kârlı gören bir dikkat ekonomisinde tek başına yetersizdir.", "k": "claim", "tr": "support", "c": 0.88},
        ],
    },
    {
        "side": "con", "type": "attack", "strength": 0.85,
        "spans": [
            {"t": "Bir an için tanımlara dönelim, ", "k": "other", "tr": "none", "c": 0.7},
            {"t": "çünkü tartışmanın düğümü burada. ", "k": "other", "tr": "none", "c": 0.71},
            {"t": "Rakibimin konumu, aracı kullanan aktör ile aracın kendisini özdeşleştiren bir kategori hatası içeriyor; ", "k": "claim", "tr": "attack", "c": 0.9},
            {"t": "bir bıçak hem cerrahın hem suçlunun elinde olabilir, ama biz bıçağı yasaklamayız. ", "k": "other", "tr": "none", "c": 0.73},
            {"t": "Otoriter rejimlerin sosyal medyayı kullanması, aracın içkin olarak yıkıcı değil bağlamsal olduğunu gösterir. ", "k": "claim", "tr": "attack", "c": 0.9},
            {"t": "Nitekim aynı araç bağımsız gazeteciler ve sivil toplum tarafından hesap sormak için de kullanılır. ", "k": "premise", "tr": "attack", "c": 0.89},
            {"t": "Kaldı ki 2022'de PNAS'ta yayımlanan saha çalışması, Finlandiya'da verilen medya okuryazarlığı eğitiminin yanlış bilgiyi ayırt etme başarısını anlamlı ölçüde artırdığını ortaya koymuştur. ", "k": "premise", "tr": "attack", "c": 0.93},
            {"t": "Yani sorun teknolojik determinizmle değil, eğitim ve kurumsal tasarımla çözülür.", "k": "claim", "tr": "attack", "c": 0.87},
        ],
    },
    {
        "side": "pro", "type": "rebuttal", "strength": 0.93,
        "spans": [
            {"t": "Bıçak benzetmesi ilgi çekici, ", "k": "other", "tr": "none", "c": 0.72},
            {"t": "ama yanıltıcı; çünkü bıçak kullanıcısını saatlerce elinde tutmak için davranışını optimize etmez. ", "k": "claim", "tr": "support", "c": 0.89},
            {"t": "Bağlamsallık argümanı esasen tezimi doğruluyor, ", "k": "claim", "tr": "support", "c": 0.9},
            {"t": "çünkü aynı aracın hem demokratik hareketler hem otoriter rejimlerce bu denli kolay silahlandırılabilmesi, ortak gerçeklik zeminini ne kadar kırılganlaştırdığını gösterir. ", "k": "premise", "tr": "support", "c": 0.95},
            {"t": "Finlandiya örneği kuralı değil istisnayı temsil eder; ", "k": "claim", "tr": "support", "c": 0.86},
            {"t": "zira küçük ve yüksek güvenli bir toplumun sonuçlarını küresel ölçekli bir dikkat ekonomisine genellemek metodolojik bir hatadır. ", "k": "premise", "tr": "support", "c": 0.9},
            {"t": "2023'te Nature'da yayımlanan bir araştırma, algoritmik sıralamanın bölücü paylaşımları kullanıcı tercihinden bağımsız olarak yapay biçimde yükselttiğini deneysel olarak kanıtlamıştır. ", "k": "premise", "tr": "support", "c": 0.99},
            {"t": "Yani sorun yalnızca bireyin neyi tıkladığı değil, sistemin önüne neyi koyduğudur.", "k": "claim", "tr": "support", "c": 0.91},
        ],
    },
    {
        "side": "con", "type": "attack", "strength": 0.82,
        "spans": [
            {"t": "Kapanışta pozisyonunuzdaki iç gerilimi vurgulayayım. ", "k": "other", "tr": "none", "c": 0.7},
            {"t": "En büyük açmazınız, algoritmik kaosu aynı anda hem otoriterlerin meşruiyet aracı hem de tüm kurumları çökerten evrensel bir zehir olarak tanımlamanızdır; ", "k": "claim", "tr": "attack", "c": 0.88},
            {"t": "bu iki iddia birbirini zayıflatır. ", "k": "claim", "tr": "attack", "c": 0.82},
            {"t": "Eğer araç bağlama göre hem güven hem güvensizlik üretebiliyorsa, belirleyici değişken aracın özü değil aktörlerin niyetidir. ", "k": "claim", "tr": "attack", "c": 0.9},
            {"t": "Nitekim Avrupa Komisyonu'nun 2023 raporu, Estonya, İrlanda ve Portekiz'de uygulanan okuryazarlık programlarının seçim dönemi dezenformasyon etkisini yüzde 30'a varan oranda azalttığını belgelemiştir. ", "k": "premise", "tr": "attack", "c": 0.94},
            {"t": "Yani çözüm, ", "k": "other", "tr": "none", "c": 0.7},
            {"t": "yangına su sıkmak değil, yangının kimyasını —kurumsal düzenleme ve eğitimi— değiştirmektir.", "k": "claim", "tr": "attack", "c": 0.85},
        ],
    },
]

# Inter-component relations (support/attack) referenced by (msg_index, span_index).
# Clause-level units → premise clauses support their message's main claim;
# rebuttal claims attack the opposing claim (cross-turn).
DEBATE_RELATIONS = [
    ((0, 3), (0, 2), "support", 0.99, 0),
    ((0, 4), (0, 2), "support", 0.97, 0),
    ((0, 6), (0, 2), "support", 0.92, 0),
    ((1, 2), (0, 2), "attack", 0.96, 1),
    ((1, 3), (1, 2), "support", 0.93, 0),
    ((1, 4), (1, 2), "support", 0.90, 0),
    ((1, 5), (1, 8), "support", 0.90, 0),
    ((2, 3), (1, 2), "attack", 0.95, 1),
    ((2, 4), (2, 3), "support", 0.99, 0),
    ((2, 1), (1, 2), "attack", 0.85, 1),
    ((3, 4), (2, 3), "attack", 0.90, 1),
    ((3, 6), (3, 4), "support", 0.93, 0),
    ((4, 2), (3, 4), "attack", 0.92, 1),
    ((4, 3), (4, 2), "support", 0.99, 0),
    ((4, 6), (4, 2), "support", 0.95, 0),
    ((5, 3), (4, 2), "attack", 0.90, 1),
    ((5, 4), (5, 3), "support", 0.94, 0),
    ((5, 1), (4, 2), "attack", 0.86, 1),
]


# ------------------------------------------------------------------
# FACT-CHECK — a low-credibility viral claim, fully analyzed.
# ------------------------------------------------------------------
FACT_ARTICLE_TITLE = "\"Elektrik faturaları bir yılda tam %400 arttı — vatandaş isyanda!\""
FACT_SPANS = [
    {"t": "Uzmanlara göre konut elektrik faturaları son bir yılda tam yüzde 400 arttı. ", "claim": True, "type": "claim", "verdict": "Yanıltıcı", "stat": True, "manip": 0.82,
     "exp": "EPDK konut tarife verisi (mock) yıllık ~%62 artış gösteriyor; iddia resmi rakamı yaklaşık 6,5 kat abartmaktadır."},
    {"t": "Yetkililer konuyu görmezden gelirken sosyal medyada paylaşımlar çığ gibi büyüyor. ", "claim": False},
    {"t": "Hiçbir resmi kurum bu rakamı yalanlamadı. ", "claim": True, "type": "claim", "verdict": "Yanlış", "stat": False, "manip": 0.74,
     "exp": "EPDK ve ilgili bakanlık tarife açıklamaları kamuya açıktır; 'yalanlanmadı' ifadesi kanıtlanamaz ve yanlıştır."},
    {"t": "Artan enerji maliyetinin hane bütçelerini zorladığı doğru olsa da, bu oranın gerçeği yansıtmadığı görülüyor. ", "claim": True, "type": "claim", "verdict": "Kısmen Doğru", "stat": False, "manip": 0.25,
     "exp": "Enerji maliyet artışı gerçektir; ancak haberdeki oran çarpıtılmıştır."},
    {"t": "Vatandaşlar bu akıl almaz zammın ülkeyi felakete sürüklediğini savunuyor.", "claim": False},
]
FACT_EVIDENCE = [
    {"claim_idx": 0, "domain": "teyit.org", "title": "Elektrik faturalarının bir yılda %400 arttığı iddiası", "url": "https://teyit.org/analiz/elektrik-faturasi-400-iddiasi",
     "snippet": "İncelememizde resmi EPDK tarifelerine göre konut elektrik birim fiyatındaki yıllık artışın %62 düzeyinde olduğu, %400 oranının gerçeği yansıtmadığı görüldü.",
     "verdict": "Yanıltıcı", "cred": 0.93, "bias": "merkez", "archive": True, "quality": "high"},
    {"claim_idx": 0, "domain": "epdk.gov.tr", "title": "EPDK Elektrik Tarifeleri — Konut Aboneliği", "url": "https://www.epdk.gov.tr/elektrik-tarifeler",
     "snippet": "Resmi tarife tablosuna göre konut grubu elektrik birim fiyatı son 12 ayda yaklaşık %62 oranında güncellenmiştir.",
     "verdict": "Resmi veri", "cred": 0.98, "bias": "resmi", "archive": False, "public": True, "quality": "high"},
    {"claim_idx": 2, "domain": "toolbox.google.com", "title": "Google Fact Check: misleading energy price claim", "url": "https://toolbox.google.com/factcheck/explorer",
     "snippet": "Independent reviewers rated similar 'electricity bills up 400%' claims as Misleading due to exaggerated figures.",
     "verdict": "Misleading", "cred": 0.85, "bias": "merkez", "archive": False, "quality": "high"},
]
FACT_BALANCED = {
    "main_thesis": "Konut elektrik faturalarının bir yılda %400 arttığı ve yetkililerin sustuğu iddiası.",
    "dialectic_tone": "Sansasyonel ve Alarmist",
    "objectivity_score": 22,
    "balance_score": 18,
    "source_diversity_count": 1,
    "findings": [
        "Tek taraflı anlatı: resmi kurum verisine veya karşı görüşe yer verilmemiş.",
        "Belirsiz otorite atfı: 'uzmanlara göre' ifadesi kaynak göstermiyor.",
        "Doğrulanabilir sayısal iddia (%400) resmi veriyle (%62) çelişiyor.",
    ],
}
FACT_MANIP = {
    "fallacies": [
        {"type": "Abartı / Sansasyon", "explanation": "'tam %400', 'akıl almaz zam', 'felaket' gibi abartılı ifadeler."},
        {"type": "Belirsiz Otorite", "explanation": "'Uzmanlara göre' — kaynak veya isim verilmeden otorite atfı."},
        {"type": "Kanıtlanamaz İddia", "explanation": "'Hiçbir resmi kurum yalanlamadı' ifadesi doğrulanamaz."},
        {"type": "Duygusal Dil", "explanation": "'vatandaş isyanda', 'çığ gibi' ile duygusal manipülasyon."},
    ],
}


async def _get_or_create_agents(db) -> tuple[Agent, Agent]:
    agents = (await db.execute(select(Agent).order_by(Agent.id))).scalars().all()
    pro = next((a for a in agents if a.role == "proponent"), None)
    con = next((a for a in agents if a.role == "opponent"), None)
    if pro and con:
        return pro, con
    pro = Agent(name="Savunucu Ajan", model_name="demo", role="proponent", system_prompt="demo", temperature=0.3)
    con = Agent(name="Karşıt Ajan", model_name="demo", role="opponent", system_prompt="demo", temperature=0.3)
    db.add_all([pro, con])
    await db.flush()
    return pro, con


async def _delete_debate(db, debate_id: int) -> None:
    """Remove a debate and all its dependents (FK-safe order)."""
    run_ids = (
        await db.execute(select(AnalysisRun.id).filter_by(debate_id=debate_id))
    ).scalars().all()
    if run_ids:
        await db.execute(delete(ArgumentRelation).where(ArgumentRelation.analysis_run_id.in_(run_ids)))
        await db.execute(delete(ArgumentComponent).where(ArgumentComponent.analysis_run_id.in_(run_ids)))
        await db.execute(delete(AnalysisRun).where(AnalysisRun.id.in_(run_ids)))
    msg_ids = (
        await db.execute(select(Message.id).filter_by(debate_id=debate_id))
    ).scalars().all()
    if msg_ids:
        await db.execute(delete(MessageVersion).where(MessageVersion.message_id.in_(msg_ids)))
        await db.execute(delete(Message).where(Message.debate_id == debate_id))
    await db.execute(delete(DebateParticipant).where(DebateParticipant.debate_id == debate_id))
    await db.execute(delete(Debate).where(Debate.id == debate_id))
    await db.flush()


async def _seed_debate(db, svc) -> int:
    # Always rebuild so content edits to the demo take effect on restart.
    existing = (
        await db.execute(select(Debate).filter_by(topic=DEMO_DEBATE_TOPIC).limit(1))
    ).scalars().first()
    if existing:
        await _delete_debate(db, existing.id)

    pro, con = await _get_or_create_agents(db)
    debate = Debate(topic=DEMO_DEBATE_TOPIC, status="resolved", max_rounds=3, current_round=3)
    db.add(debate)
    await db.flush()
    db.add_all([
        DebateParticipant(debate_id=debate.id, agent_id=pro.id, position=0, role_in_debate="proponent"),
        DebateParticipant(debate_id=debate.id, agent_id=con.id, position=1, role_in_debate="opponent"),
    ])

    run = AnalysisRun(
        debate_id=debate.id,
        status="completed",
        relation_threshold=svc.RELATION_GRAPH_THRESHOLD,
        attack_threshold=svc.ATTACK_RESPONSE_THRESHOLD,
        component_model=svc._component_model_version(),
        relation_model=svc._relation_model_version(),
        completed_at=datetime.datetime.utcnow(),
    )
    db.add(run)
    await db.flush()

    parent_id = None
    comp_index: dict[tuple[int, int], ArgumentComponent] = {}
    for mi, m in enumerate(DEBATE_MESSAGES):
        agent = pro if m["side"] == "pro" else con
        text, spans = _assemble(m["spans"])
        msg = Message(
            debate_id=debate.id, agent_id=agent.id, parent_id=parent_id,
            message_type=m["type"], position=mi,
        )
        db.add(msg)
        await db.flush()
        version = MessageVersion(
            message_id=msg.id, version_number=1, content=text,
            is_current=True, strength_score=m["strength"],
        )
        db.add(version)
        await db.flush()
        parent_id = msg.id

        for si, sp in enumerate(spans):
            comp = ArgumentComponent(
                analysis_run_id=run.id,
                message_version_id=version.id,
                message_id=msg.id,
                agent_id=agent.id,
                component_type=sp["k"],
                text=sp["t"].strip(),
                start_idx=sp["start"],
                end_idx=sp["end"],
                confidence=sp["c"],
                topic_relation_type=sp["tr"],
                topic_relation_confidence=sp["c"],
                topic_relation_probabilities_json=json.dumps({sp["tr"]: sp["c"]}),
            )
            db.add(comp)
            comp_index[(mi, si)] = comp
    await db.flush()

    for (src, tgt, rtype, conf, dist) in DEBATE_RELATIONS:
        s, t = comp_index.get(src), comp_index.get(tgt)
        if not s or not t:
            continue
        db.add(ArgumentRelation(
            analysis_run_id=run.id,
            source_component_id=s.id,
            target_component_id=t.id,
            relation_type=rtype,
            confidence=conf,
            probabilities_json=json.dumps({rtype: conf}),
            distance_turns=dist,
            is_long_range=dist >= 2,
        ))
    await db.flush()
    return debate.id


async def _seed_fact_check(db) -> str:
    existing = (
        await db.execute(select(FactCheckRun).filter_by(export_token=DEMO_FACT_TOKEN).limit(1))
    ).scalars().first()
    if existing:
        return DEMO_FACT_TOKEN

    raw_text, spans = _assemble([{"t": s["t"]} for s in FACT_SPANS])
    run = FactCheckRun(
        input_type="text",
        url=None,
        title=FACT_ARTICLE_TITLE,
        raw_text=raw_text,
        status="completed",
        source_metadata_json=json.dumps({
            "domain": "hizli-haber-merkezi.example",
            "credibility_score": 18,
            "source_bias": "sansasyonel",
            "published_at": "2026-06-14",
        }),
        summary="İddia, resmi EPDK tarife verisiyle (yıllık ~%62) çeliştiği için YANILTICI olarak değerlendirildi; metin sansasyonel dil ve belirsiz kaynak kullanıyor.",
        export_token=DEMO_FACT_TOKEN,
        balanced_reporting_json=json.dumps(FACT_BALANCED),
        manipulation_findings_json=json.dumps(FACT_MANIP),
        completed_at=datetime.datetime.utcnow(),
    )
    db.add(run)
    await db.flush()

    claim_rows: dict[int, FactClaim] = {}
    for i, (sp, meta) in enumerate(zip(spans, FACT_SPANS)):
        if not meta.get("claim"):
            continue
        claim = FactClaim(
            run_id=run.id,
            component_type="claim",
            text=meta["t"].strip(),
            start_idx=sp["start"],
            end_idx=sp["end"],
            confidence=0.95,
            claim_type="statistical" if meta.get("stat") else "factual",
            verdict_status=meta["verdict"],
            explanation=meta.get("exp"),
            is_statistical=bool(meta.get("stat")),
            manipulation_score=meta.get("manip", 0.0),
            extraction_reason="model_claim",
        )
        db.add(claim)
        await db.flush()
        claim_rows[i] = claim

    for ev in FACT_EVIDENCE:
        claim = claim_rows.get(ev["claim_idx"])
        evidence = FactEvidence(
            run_id=run.id,
            claim_id=claim.id if claim else None,
            search_query=FACT_ARTICLE_TITLE,
            url=ev["url"],
            title=ev["title"],
            source_domain=ev["domain"],
            snippet=ev["snippet"],
            published_at="2026",
            score=ev["cred"],
            credibility_score=ev["cred"],
            source_bias=ev.get("bias"),
            relevance_score=0.95,
            source_quality=ev.get("quality", "high"),
            accepted_for_verdict=True,
            is_turkish_archive=bool(ev.get("archive")),
            is_public_data_source=bool(ev.get("public")),
            archive_match_claim_text=claim.text if (claim and ev.get("archive")) else None,
        )
        db.add(evidence)
        await db.flush()
        db.add(FactRelation(
            run_id=run.id,
            source_claim_id=None,
            evidence_id=evidence.id,
            target_claim_id=claim.id if claim else list(claim_rows.values())[0].id,
            relation_scope="evidence",
            relation_type="attack" if ev["verdict"] in ("Yanıltıcı", "Misleading") else "support",
            confidence=ev["cred"],
            probabilities_json=json.dumps({"attack": ev["cred"]}),
        ))
    await db.flush()
    return DEMO_FACT_TOKEN


async def ensure_demo(db, svc) -> dict:
    """Create the demo debate + fact-check run if absent. Returns their handles."""
    debate_id = await _seed_debate(db, svc)
    fact_token = await _seed_fact_check(db)
    await db.commit()
    return {"debate_id": debate_id, "fact_token": fact_token}
