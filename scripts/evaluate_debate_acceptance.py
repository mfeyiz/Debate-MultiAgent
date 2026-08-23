"""Evaluate two fixed five-turn debate transcripts with oracle labels."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.bert_service import ModernBERTPipeline


DEBATES = [
    {
        "name": "ai_education_five_turns",
        "components": [
            ("Yapay zeka araçları öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.", "claim"),
            ("Deneyde öğretmen geri bildirimiyle kullanılan yapay zeka araçları öğrencilerin kavram yanılgılarını azaltmıştır.", "premise"),
            ("Denetimsiz kullanımda öğrencilerin aracı kopyalama için kullandığı raporlandı.", "premise"),
            ("Bu bilgi denetimli kullanım iddiasını doğrudan çürütmez.", "claim"),
            ("Okul binalarındaki internet altyapısı ve lisans maliyetleri ayrıca listelendi.", "other"),
            ("Yapay zeka araçları tek başına kullanıldığında öğrenmeyi destekler.", "claim"),
            ("Deneyde öğretmen geri bildirimi olmadan kullanılan araçların öğrenme kazanımını artırmadığı görüldü.", "premise"),
            ("Öğrencilerin problem çözme becerileri gelişti.", "claim"),
            ("Michigan Üniversitesi çalışması, öğrencilerin bağımsız problem çözme becerilerinin gerilediğini gösterdi.", "premise"),
            ("Ana sayfada üyelik, çerez tercihleri ve sosyal medya bağlantıları yer alıyor.", "other"),
        ],
        "relations": [
            (
                "Yapay zeka araçları öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.",
                "Deneyde öğretmen geri bildirimiyle kullanılan yapay zeka araçları öğrencilerin kavram yanılgılarını azaltmıştır.",
                "support",
            ),
            (
                "Yapay zeka araçları öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.",
                "Denetimsiz kullanımda öğrencilerin aracı kopyalama için kullandığı raporlandı.",
                "none",
            ),
            (
                "Yapay zeka araçları tek başına kullanıldığında öğrenmeyi destekler.",
                "Deneyde öğretmen geri bildirimi olmadan kullanılan araçların öğrenme kazanımını artırmadığı görüldü.",
                "attack",
            ),
            (
                "Öğrencilerin problem çözme becerileri gelişti.",
                "Michigan Üniversitesi çalışması, öğrencilerin bağımsız problem çözme becerilerinin gerilediğini gösterdi.",
                "attack",
            ),
            (
                "Yapay zeka araçları öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.",
                "Okul binalarındaki internet altyapısı ve lisans maliyetleri ayrıca listelendi.",
                "none",
            ),
        ],
    },
    {
        "name": "economy_policy_five_turns",
        "components": [
            ("Enflasyon 2024 sonunda yüzde 44,38 oldu.", "claim"),
            ("Resmi bültende 2024 yıl sonu enflasyonunun yüzde 20 olduğu yazıyor.", "premise"),
            ("Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.", "claim"),
            ("Türkiye ekonomisi 2004 yılında yüksek büyüme oranı yakalamıştı.", "premise"),
            ("Bakan enflasyonun yıl sonunda tek haneye düşeceğini söyledi.", "claim"),
            ("Bakan konuşmasında yıl sonu için tek haneli enflasyon hedefi vermediğini açıkça belirtti.", "premise"),
            ("Konut fiyatları reel olarak yükseldi.", "claim"),
            ("Raporda nominal fiyatların arttığı, ancak enflasyondan arındırılmış reel fiyatların düştüğü belirtildi.", "premise"),
            ("TÜİK aynı dönemde toplam nüfus ve hane sayısı için ayrı istatistikler yayımladı.", "other"),
            ("Genç işsizlik oranı 2024'te azaldı.", "claim"),
            ("TÜİK tablosu genç işsizlik oranının 2024'te önceki yıla göre düştüğünü gösterdi.", "premise"),
        ],
        "relations": [
            (
                "Enflasyon 2024 sonunda yüzde 44,38 oldu.",
                "Resmi bültende 2024 yıl sonu enflasyonunun yüzde 20 olduğu yazıyor.",
                "attack",
            ),
            (
                "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
                "Türkiye ekonomisi 2004 yılında yüksek büyüme oranı yakalamıştı.",
                "none",
            ),
            (
                "Bakan enflasyonun yıl sonunda tek haneye düşeceğini söyledi.",
                "Bakan konuşmasında yıl sonu için tek haneli enflasyon hedefi vermediğini açıkça belirtti.",
                "attack",
            ),
            (
                "Konut fiyatları reel olarak yükseldi.",
                "Raporda nominal fiyatların arttığı, ancak enflasyondan arındırılmış reel fiyatların düştüğü belirtildi.",
                "attack",
            ),
            (
                "Genç işsizlik oranı 2024'te azaldı.",
                "TÜİK tablosu genç işsizlik oranının 2024'te önceki yıla göre düştüğünü gösterdi.",
                "support",
            ),
            (
                "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
                "TÜİK aynı dönemde toplam nüfus ve hane sayısı için ayrı istatistikler yayımladı.",
                "none",
            ),
        ],
    },
]


def pass_relation(expected: str, predicted: str, confidence: float) -> bool:
    if expected == predicted:
        return True
    return expected == "none" and predicted in {"support", "attack"} and confidence < 0.60


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--component-model-dir", type=Path, default=None)
    parser.add_argument("--relation-model-dir", type=Path, default=None)
    args = parser.parse_args()
    pipeline = ModernBERTPipeline(
        component_model_dir=args.component_model_dir,
        relation_model_dir=args.relation_model_dir,
    )
    debate_results = []
    for debate in DEBATES:
        component_results = []
        for text, expected in debate["components"]:
            predicted, confidence = pipeline._classify_component_unit(text)
            component_results.append(
                {
                    "text": text,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "passed": predicted == expected,
                }
            )

        relation_results = []
        for claim, evidence, expected in debate["relations"]:
            predicted, confidence, probabilities = pipeline.classify_relation(claim, evidence)
            relation_results.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "expected": expected,
                    "predicted": predicted,
                    "confidence": confidence,
                    "probabilities": probabilities,
                    "passed": pass_relation(expected, predicted, confidence),
                }
            )

        component_accuracy = sum(item["passed"] for item in component_results) / len(component_results)
        relation_accuracy = sum(item["passed"] for item in relation_results) / len(relation_results)
        debate_results.append(
            {
                "name": debate["name"],
                "component_accuracy": round(component_accuracy, 4),
                "relation_accuracy": round(relation_accuracy, 4),
                "passed": component_accuracy >= 0.90 and relation_accuracy >= 0.90,
                "components": component_results,
                "relations": relation_results,
            }
        )

    print(json.dumps({"debates": debate_results}, ensure_ascii=False, indent=2))
    if not all(result["passed"] for result in debate_results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
