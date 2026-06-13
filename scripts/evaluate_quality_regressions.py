"""Run fixed quality regressions against the local ModernBERT pipeline."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.bert_service import ModernBERTPipeline


CASES = [
    {
        "name": "support_remote_work",
        "target": "Uzaktan çalışma çalışan verimliliğini artırır.",
        "source": "Stanford Üniversitesi araştırması evden çalışan personelin verimliliğinin yüzde 13 arttığını göstermiştir.",
        "expected": "support",
    },
    {
        "name": "numeric_contradiction",
        "target": "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
        "source": "TÜİK verilerine göre Türkiye ekonomisi 2024 yılında yüzde 3,2 büyüdü.",
        "expected": "attack",
    },
    {
        "name": "irrelevant_source",
        "target": "Haberde uzman görüşü veya karşı görüş yer almıyor.",
        "source": "PolitiFact ana sayfasındaki menü, üyelik ve son haber bağlantıları listeleniyor.",
        "expected": "none",
    },
    {
        "name": "manipulative_unsupported",
        "target": "Aşılar toplum sağlığını korur.",
        "source": "Kaynaklara göre herkes bu gerçeği saklıyor ve medya bunu yayınlamıyor.",
        "expected": "none_or_weak",
    },
]


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--component-model-dir", type=Path, default=None)
    parser.add_argument("--relation-model-dir", type=Path, default=None)
    args = parser.parse_args()
    pipeline = ModernBERTPipeline(
        component_model_dir=args.component_model_dir,
        relation_model_dir=args.relation_model_dir,
    )
    results = []
    for case in CASES:
        result = pipeline.analyze(case["source"], case["target"])
        top_relation = result.relations[0].relation_type if result.relations else "none"
        results.append(
            {
                "name": case["name"],
                "expected": case["expected"],
                "predicted_relation": top_relation,
                "strength": result.overall_strength,
                "passed": (
                    top_relation == case["expected"]
                    or (case["expected"] == "neutral" and top_relation == "none")
                    or (case["expected"] == "none_or_weak" and result.overall_strength < 0.65)
                ),
            }
        )
    print(json.dumps({"results": results}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
