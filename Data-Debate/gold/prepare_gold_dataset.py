"""Create an independent gold seed dataset for argument mining.

This file intentionally does not read or transform the existing project
dataset. The examples below are hand-authored Turkish argument cases designed
to cover support, attack, neutral, same-topic neutral, connector-heavy, and
connector-free phrasing.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Literal, TypedDict


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
COMPONENT_PATH = DATA_DIR / "gold_component_dataset.json"
RELATION_PATH = DATA_DIR / "gold_relation_pairs.json"
SEED = 42

ComponentLabel = Literal["claim", "evidence", "background"]
RelationLabel = Literal["support", "attack", "neutral"]


class Segment(TypedDict):
    label: ComponentLabel
    text: str


class RelationPair(TypedDict):
    claim_text: str
    evidence_text: str
    label: RelationLabel
    topic: str


CASES = [
    {
        "topic": "uzaktan_calisma",
        "claim": "Uzaktan çalışma verimliliği artırır.",
        "support": [
            "Çünkü çalışanlar ofis içi kesintilerden uzaklaştığında derin çalışma süresi artar.",
            "Stanford kaynaklı bir saha deneyi, evden çalışan ekiplerde tamamlanan görev sayısının belirgin biçimde yükseldiğini göstermiştir.",
        ],
        "attack": [
            "Buna rağmen uzaktan çalışma yeni başlayan çalışanlarda mentorluk temasını azalttığı için öğrenme hızını düşürebilir.",
            "Takım içi koordinasyon zayıfladığında kararların gecikmesi verimlilik kazancını azaltır.",
        ],
        "neutral": [
            "Uzaktan çalışanların kullandığı masa sandalye modelleri şirketlerin satın alma politikalarına göre değişir.",
            "Kahve tüketimi büyük şehirlerde son yıllarda daha çeşitli hale gelmiştir.",
        ],
    },
    {
        "topic": "sosyal_medya",
        "claim": "Sosyal medya gençlerin ruh sağlığını iyileştirir.",
        "support": [
            "Destek gruplarına erişim bulan gençler yalnızlık duygusunu azaltan güvenli topluluklar kurabilir.",
            "Kriz anında yardım hatlarına yönlendiren kampanyalar bazı gençlerin profesyonel destek almasını kolaylaştırır.",
        ],
        "attack": [
            "Ancak yoğun sosyal medya kullanımı uyku düzenini bozarak kaygı belirtilerini artırabilir.",
            "Sürekli karşılaştırma kültürü beden algısı ve özsaygı üzerinde olumsuz baskı oluşturur.",
        ],
        "neutral": [
            "Sosyal medya uygulamalarının arayüz renkleri marka kimliğine göre sık sık güncellenir.",
            "Kış sporları turizmi yüksek rakımlı bölgelerde konaklama talebini artırır.",
        ],
    },
    {
        "topic": "elektrikli_arac",
        "claim": "Elektrikli araçlar çevre kirliliğini azaltır.",
        "support": [
            "Egzoz gazı üretmedikleri için yoğun şehir merkezlerinde yerel hava kalitesini iyileştirirler.",
            "Yenilenebilir elektrikle şarj edilen araçların kullanım aşamasındaki karbon salımı fosil yakıtlı araçlardan daha düşüktür.",
        ],
        "attack": [
            "Fakat batarya üretiminde kullanılan madencilik süreçleri su kaynakları ve ekosistemler üzerinde baskı yaratabilir.",
            "Elektrik üretimi kömüre dayalıysa toplam emisyon avantajı önemli ölçüde azalır.",
        ],
        "neutral": [
            "Elektrikli araçlarda ekran boyutu ve iç aydınlatma tercihleri marka segmentine göre değişir.",
            "Müze ziyaret saatleri bayram dönemlerinde yerel yönetimler tarafından yeniden düzenlenebilir.",
        ],
    },
    {
        "topic": "telefon_okul",
        "claim": "Okullarda telefon kullanımı akademik başarıyı düşürür.",
        "support": [
            "Ders sırasında gelen bildirimler öğrencinin dikkatini bölerek konu takibini zorlaştırır.",
            "Sınav haftalarında ekran süresi yükselen öğrencilerin çalışma süresi kısalabilir.",
        ],
        "attack": [
            "Buna karşın öğretmen denetiminde kullanılan telefonlar hızlı araştırma ve etkileşimli ölçme etkinliklerini destekleyebilir.",
            "Erişilebilirlik uygulamaları bazı öğrencilerin not alma ve okuma süreçlerini kolaylaştırır.",
        ],
        "neutral": [
            "Okul kantinlerinde satılan sandviç çeşitleri öğrencilerin öğle yemeği tercihlerini etkiler.",
            "Deniz fenerleri kıyı güvenliği açısından tarihsel olarak önemli yapılar arasında yer alır.",
        ],
    },
    {
        "topic": "plastik_poset",
        "claim": "Plastik poşet ücretleri çevre kirliliğini azaltır.",
        "support": [
            "Ücret uygulaması tüketicileri bez çanta kullanmaya yönelttiği için tek kullanımlık poşet talebi düşer.",
            "Market verileri poşet satışlarının ücretlendirme sonrasında belirgin biçimde azaldığını göstermektedir.",
        ],
        "attack": [
            "Ancak tüketiciler daha kalın çöp poşetleri satın almaya başlarsa toplam plastik tüketimi beklenenden az düşebilir.",
            "Denetim zayıf olduğunda küçük işletmeler ücretsiz poşet vermeyi sürdürerek politikanın etkisini azaltır.",
        ],
        "neutral": [
            "Plastik poşetlerin renkleri perakende markalarının tasarım tercihlerine göre değişebilir.",
            "Futbol kulüplerinin forma renkleri taraftar kimliğinde güçlü bir semboldür.",
        ],
    },
    {
        "topic": "yapay_zeka_egitim",
        "claim": "Yapay zeka eğitimde kaliteyi artırır.",
        "support": [
            "Öğrencinin eksiklerini hızlı tespit eden sistemler öğretmene kişiselleştirilmiş alıştırma önerileri sunar.",
            "Anında geri bildirim alan öğrenciler hatalarını sınav dönemini beklemeden düzeltebilir.",
        ],
        "attack": [
            "Yanlış veya önyargılı veriyle eğitilen sistemler öğrenciyi hatalı yönlendirebilir.",
            "Aşırı otomasyon öğretmenin pedagojik sezgisini geri plana iterse öğrenme deneyimi zayıflar.",
        ],
        "neutral": [
            "Yapay zeka araçlarının lisans ücretleri kurumların bütçe planına göre farklı ödeme dönemlerine ayrılır.",
            "Zeytinyağı üretiminde hasat zamanı ürün aromasını etkileyen faktörlerden biridir.",
        ],
    },
    {
        "topic": "dort_gun",
        "claim": "Haftada dört gün çalışma şirket performansını artırır.",
        "support": [
            "Daha kısa çalışma haftası tükenmişliği azalttığında çalışanların odaklanma kalitesi yükselir.",
            "Pilot uygulamalarda bazı ekipler aynı çıktıyı daha az toplantıyla üretebildiğini raporlamıştır.",
        ],
        "attack": [
            "Müşteri desteği kesintisiz hizmet gerektiriyorsa çalışma gününün azalması yanıt sürelerini uzatabilir.",
            "Üretim hattında vardiya planı değişmezse aynı hacmi korumak için ek maliyet doğabilir.",
        ],
        "neutral": [
            "Dört günlük modelde izin takvimi insan kaynakları yazılımında farklı renklerle gösterilebilir.",
            "Klasik gitar tellerinin malzemesi müzisyenin ton tercihine göre seçilir.",
        ],
    },
    {
        "topic": "toplu_tasima",
        "claim": "Toplu taşıma yatırımları trafik yoğunluğunu azaltır.",
        "support": [
            "Güvenilir metro ve otobüs hatları özel araç kullanma ihtiyacını azaltarak ana arterleri rahatlatır.",
            "Aktarma süreleri kısaldığında yolcular günlük ulaşımda otomobil yerine toplu taşımayı tercih eder.",
        ],
        "attack": [
            "Hatlar düşük talep bölgelerine plansız yapılırsa yolcu çekmeden kamu bütçesinde yük oluşturabilir.",
            "Park alanı ve ring bağlantısı kurulmadığında banliyö kullanıcıları aracı bırakmakta zorlanır.",
        ],
        "neutral": [
            "Toplu taşıma kartlarının tasarımı belediyenin görsel kimliğiyle uyumlu hazırlanır.",
            "Akdeniz mutfağında zeytinyağı birçok geleneksel tarifin temel unsurudur.",
        ],
    },
    {
        "topic": "geri_donusum",
        "claim": "Geri dönüşüm belediye atıklarını azaltır.",
        "support": [
            "Kaynağında ayrıştırma yapılan mahallelerde çöpe giden ambalaj miktarı düşer.",
            "Depozito sistemi şişe ve kutuların tekrar toplama zincirine dönmesini sağlar.",
        ],
        "attack": [
            "Ayrıştırma tesisi kapasitesi yetersizse toplanan malzemenin önemli bölümü yine depolama alanına gidebilir.",
            "Kirli veya karışık atıklar geri dönüşüm verimini düşürerek beklenen çevresel faydayı azaltır.",
        ],
        "neutral": [
            "Geri dönüşüm kutularının rengi kullanıcıların atık türünü hızlı ayırt etmesi için standartlaştırılır.",
            "Tiyatro festivalleri yerel sanat topluluklarının görünürlüğünü artırabilir.",
        ],
    },
    {
        "topic": "uyku",
        "claim": "Düzenli uyku akademik başarıyı artırır.",
        "support": [
            "Yeterli uyku dikkat süresini ve hafıza pekişmesini güçlendirdiği için öğrenilen bilgiler daha kalıcı olur.",
            "Her gün benzer saatte uyuyan öğrenciler sabah derslerinde daha uyanık kalabilir.",
        ],
        "attack": [
            "Tek başına uyku düzeni, düşük ders katılımı ve eksik çalışma alışkanlığı varsa başarıyı garanti etmez.",
            "Sınav kaygısı yüksek öğrenciler düzenli uyusalar bile performans düşüşü yaşayabilir.",
        ],
        "neutral": [
            "Uyku takip uygulamalarında grafiklerin renkleri kullanıcının seçtiği temaya göre değişir.",
            "Seramik atölyelerinde kullanılan sır teknikleri yüzey dokusunu belirler.",
        ],
    },
    {
        "topic": "organik_tarim",
        "claim": "Organik tarım toprak sağlığını korur.",
        "support": [
            "Sentetik pestisit kullanımının azalması topraktaki mikroorganizma çeşitliliğini destekler.",
            "Dönüşümlü ekim ve kompost uygulamaları organik madde oranını artırabilir.",
        ],
        "attack": [
            "Verim düşerse aynı miktarda ürün için daha geniş arazi kullanımı gerekebilir.",
            "Sertifika maliyetleri küçük üreticilerin sürdürülebilir uygulamalara erişimini zorlaştırabilir.",
        ],
        "neutral": [
            "Organik ürün etiketlerinde kullanılan logo boyutu yönetmeliklere göre belirlenir.",
            "Dağ bisikleti yarışları parkur eğimine göre farklı zorluk kategorilerine ayrılır.",
        ],
    },
    {
        "topic": "kentsel_yesil",
        "claim": "Kentlerde yeşil alanlar yaşam kalitesini yükseltir.",
        "support": [
            "Ağaç gölgesi yaz aylarında ısı adası etkisini azaltarak sokak konforunu artırır.",
            "Parklara yakın yaşayan kişiler açık havada daha fazla hareket ederek fiziksel sağlık kazanımı elde edebilir.",
        ],
        "attack": [
            "Bakım bütçesi ayrılmazsa parklar güvenlik ve temizlik sorunları nedeniyle kullanılmaz hale gelebilir.",
            "Yeşil alan yatırımı kira baskısını artırıp düşük gelirli sakinleri mahalleden uzaklaştırabilir.",
        ],
        "neutral": [
            "Park banklarının malzemesi belediyenin bakım stratejisine göre ahşap veya metal seçilebilir.",
            "Satranç turnuvalarında zaman kontrolü oyun temposunu belirleyen temel kurallardan biridir.",
        ],
    },
]


def span_item(sample_id: int, segments: list[Segment]) -> dict:
    """Build one component sample with exact offsets."""
    text_parts: list[str] = []
    claims: list[dict] = []
    evidences: list[dict] = []
    cursor = 0
    claim_count = 1
    evidence_count = 1

    for index, segment in enumerate(segments):
        if index:
            text_parts.append(" ")
            cursor += 1
        start = cursor
        text_parts.append(segment["text"])
        cursor += len(segment["text"])
        end = cursor

        if segment["label"] == "claim":
            claims.append({"id": f"c{claim_count}", "text": segment["text"], "start": start, "end": end})
            claim_count += 1
        elif segment["label"] == "evidence":
            evidences.append({"id": f"e{evidence_count}", "text": segment["text"], "start": start, "end": end})
            evidence_count += 1

    text = "".join(text_parts)
    for span in [*claims, *evidences]:
        assert text[span["start"] : span["end"]] == span["text"]

    return {
        "id": sample_id,
        "data": {
            "text": text,
            "claim": claims,
            "Evidence": evidences,
            "support": [],
            "attack": [],
        },
    }


def component_samples() -> list[dict]:
    """Create token-level component samples with mixed segment order."""
    samples: list[dict] = []
    sample_id = 1
    for case in CASES:
        claim = {"label": "claim", "text": case["claim"]}
        supports = [{"label": "evidence", "text": text} for text in case["support"]]
        attacks = [{"label": "evidence", "text": text} for text in case["attack"]]
        backgrounds = [{"label": "background", "text": text} for text in case["neutral"]]

        samples.append(span_item(sample_id, [claim, supports[0], supports[1], attacks[0], attacks[1], backgrounds[0]]))
        sample_id += 1
        samples.append(span_item(sample_id, [backgrounds[1], claim, attacks[0], supports[0]]))
        sample_id += 1
        samples.append(span_item(sample_id, [supports[0], claim, backgrounds[0], attacks[1]]))
        sample_id += 1
    return samples


def relation_pairs() -> list[RelationPair]:
    """Create balanced relation pairs from hand-authored cases."""
    pairs: list[RelationPair] = []
    for case in CASES:
        for text in case["support"]:
            pairs.append(
                {
                    "claim_text": case["claim"],
                    "evidence_text": text,
                    "label": "support",
                    "topic": case["topic"],
                }
            )
        for text in case["attack"]:
            pairs.append(
                {
                    "claim_text": case["claim"],
                    "evidence_text": text,
                    "label": "attack",
                    "topic": case["topic"],
                }
            )
        for text in case["neutral"]:
            pairs.append(
                {
                    "claim_text": case["claim"],
                    "evidence_text": text,
                    "label": "neutral",
                    "topic": case["topic"],
                }
            )

    rng = random.Random(SEED)
    claims = [(case["topic"], case["claim"]) for case in CASES]
    evidence_pool = [
        (case["topic"], text)
        for case in CASES
        for text in [*case["support"], *case["attack"]]
    ]
    added = 0
    while added < len(CASES):
        claim_topic, claim_text = rng.choice(claims)
        evidence_topic, evidence_text = rng.choice(evidence_pool)
        if claim_topic == evidence_topic:
            continue
        pairs.append(
            {
                "claim_text": claim_text,
                "evidence_text": evidence_text,
                "label": "neutral",
                "topic": f"{claim_topic}__cross__{evidence_topic}",
            }
        )
        added += 1

    rng.shuffle(pairs)
    return pairs


def main() -> None:
    """Write gold component and relation files."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    components = component_samples()
    relations = relation_pairs()

    COMPONENT_PATH.write_text(json.dumps(components, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    RELATION_PATH.write_text(json.dumps(relations, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"component_samples={len(components)} -> {COMPONENT_PATH}")
    print(f"relation_pairs={len(relations)} -> {RELATION_PATH}")
    print("relation_counts=", dict(Counter(pair["label"] for pair in relations)))


if __name__ == "__main__":
    main()
