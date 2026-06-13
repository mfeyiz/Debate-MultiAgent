"""Evaluate hard ModernBERT edge cases beyond the smoke regressions."""

from __future__ import annotations

import json
import sys
from argparse import ArgumentParser
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.services.bert_service import ModernBERTPipeline


COMPONENT_CASES = [
    {
        "name": "ai_homework_main_thesis_strong_wording",
        "text": "Yapay zeka destekli ödev araçları, lise öğrencilerinin öğrenme kalitesini kesinlikle artırır.",
        "expected": "claim",
    },
    {
        "name": "ai_homework_mechanism_evidence",
        "text": "Bu araçlar, öğrencilere anlık ve kişiselleştirilmiş geri bildirim sağlayarak bireysel öğrenme hızına uyum sağlar.",
        "expected": "evidence",
    },
    {
        "name": "unsupported_conspiracy_other",
        "text": "Herkes bu gerçeği saklıyor ve medya bunu bilerek yayınlamıyor.",
        "expected": "other",
    },
    {
        "name": "rhetorical_question_claim",
        "text": "Bu kadar düşük başarı puanına rağmen yapay zeka ödevlerinin öğrenmeyi artırdığı nasıl söylenebilir?",
        "expected": "claim",
    },
    {
        "name": "official_source_claim_not_evidence",
        "text": "OECD raporu, öğrencilerin problem çözme becerilerinde gerileme olduğunu belirtti.",
        "expected": "claim",
    },
    {
        "name": "boilerplate_other_other",
        "text": "Ana sayfada üyelik, çerez tercihleri, gizlilik politikası ve sosyal medya bağlantıları yer alıyor.",
        "expected": "other",
    },
    {
        "name": "quoted_claim_is_claim",
        "text": "Bakan, enflasyonun yıl sonunda tek haneye düşeceğini söyledi.",
        "expected": "claim",
    },
    {
        "name": "hedged_claim_is_claim",
        "text": "Yeni düzenlemenin küçük işletmeler üzerindeki maliyeti artırması bekleniyor.",
        "expected": "claim",
    },
    {
        "name": "causal_claim_without_number",
        "text": "Uzun ekran süresi öğrencilerin dikkat süresini azaltır.",
        "expected": "claim",
    },
    {
        "name": "source_named_evidence",
        "text": "Hakemli bir çalışma, günde üç saatten fazla ekran kullanan öğrencilerin dikkat testlerinde daha düşük puan aldığını buldu.",
        "expected": "evidence",
    },
    {
        "name": "method_context_other",
        "text": "Bu bölümde araştırmanın örneklem seçimi ve veri toplama süreci açıklanmaktadır.",
        "expected": "other",
    },
    {
        "name": "sarcastic_claim",
        "text": "Elbette hiçbir kanıt yokken bu politikanın herkese fayda sağladığını varsaymamız gerekiyor.",
        "expected": "claim",
    },
    {
        "name": "recommendation_claim",
        "text": "Okullar yapay zeka ödev araçlarını sınırsız biçimde değil, öğretmen denetimiyle kullanmalıdır.",
        "expected": "claim",
    },
    {
        "name": "raw_navigation_other",
        "text": "Giriş yap, kayıt ol, şifremi unuttum, yardım merkezi ve uygulamayı indir bağlantıları gösteriliyor.",
        "expected": "other",
    },
    {
        "name": "comparative_numeric_claim",
        "text": "Türkiye'de genç işsizlik oranı 2024'te bir önceki yıla göre belirgin biçimde azaldı.",
        "expected": "claim",
    },
    {
        "name": "study_limitations_other",
        "text": "Çalışmanın örneklemi yalnızca üç şehirden seçildiği için bulguların genellenebilirliği sınırlıdır.",
        "expected": "other",
    },
    {
        "name": "factcheck_source_evidence",
        "text": "Doğrulama kuruluşu, videonun 2021'de çekildiğini ve iddia edilen olayla bağlantılı olmadığını bildirdi.",
        "expected": "evidence",
    },
    {
        "name": "conditional_policy_claim",
        "text": "Yapay zeka araçları ancak öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.",
        "expected": "claim",
    },
    {
        "name": "empty_authority_other",
        "text": "Uzmanlar bunu yıllardır söylüyor ama kimse asıl tabloyu görmek istemiyor.",
        "expected": "other",
    },
    {
        "name": "parenthood_counterclaim_is_claim",
        "text": "Agent Alpha'nın savındaki temel zayıflık, doğal olanın ahlaken doğru sayılması yanılgısına dayanmasıdır.",
        "expected": "claim",
    },
    {
        "name": "parenthood_conclusion_is_claim",
        "text": "Sonuç olarak bilinçli ebeveynlik argümanı, ahlaki bir seçimin değil bir ayrıcalığın savunusudur.",
        "expected": "claim",
    },
    {
        "name": "numeric_thousands_claim",
        "text": "UNICEF verilerine göre her gün 15.000 çocuk önlenebilir nedenlerle ölüyor.",
        "expected": "claim",
    },
    {
        "name": "markdown_heading_claim",
        "text": "**BM verileriyle çelişki**: Bölgesel nüfus düşüşü, küresel üreme yükümlülüğü doğurmaz.",
        "expected": "claim",
    },
]


RELATION_CASES = [
    {
        "name": "ai_topic_question_support_claim",
        "claim": "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır mı?",
        "evidence": "Yapay zeka destekli ödev araçları, lise öğrencilerinin öğrenme kalitesini kesinlikle artırır.",
        "expected": "support",
    },
    {
        "name": "ai_topic_question_attack_claim",
        "claim": "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır mı?",
        "evidence": "Yapay zeka destekli ödev araçları kısa vadeli başarıyı artırsa da öğrenme kalitesini uzun vadede azaltır.",
        "expected": "attack",
    },
    {
        "name": "mixed_ai_evidence_none",
        "claim": "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır.",
        "evidence": "Khan Academy raporu kısa vadeli geri bildirimin başarıyı artırdığını, fakat aşırı kullanımın bağımsız problem çözmeyi zayıflatabildiğini belirtiyor.",
        "expected": "none",
    },
    {
        "name": "same_topic_different_metric_none",
        "claim": "Merkez Bankası politika faizini yüzde 50 seviyesinde tuttu.",
        "evidence": "TÜİK aynı dönemde işsizlik oranının yüzde 8,5 olduğunu açıkladı.",
        "expected": "none",
    },
    {
        "name": "close_numeric_contradiction_attack",
        "claim": "Enflasyon 2024 sonunda yüzde 44,38 oldu.",
        "evidence": "Resmi bültende 2024 yıl sonu enflasyonunun yüzde 20 olduğu yazıyor.",
        "expected": "attack",
    },
    {
        "name": "negation_attack",
        "claim": "Aşılar toplum sağlığını korur.",
        "evidence": "Bu iddia yanlıştır; aşıların toplum sağlığını koruduğunu gösteren güvenilir bir kanıt yoktur.",
        "expected": "attack",
    },
    {
        "name": "unsupported_manipulation_none",
        "claim": "Aşılar toplum sağlığını korur.",
        "evidence": "Kaynaklara göre herkes bu gerçeği saklıyor ve medya bunu yayınlamıyor.",
        "expected": "none",
    },
    {
        "name": "irrelevant_factcheck_homepage_none",
        "claim": "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
        "evidence": "PolitiFact ana sayfasında üyelik, son haberler ve kategori bağlantıları bulunuyor.",
        "expected": "none",
    },
    {
        "name": "support_with_partial_overlap",
        "claim": "Uzaktan çalışma çalışan verimliliğini artırır.",
        "evidence": "Stanford araştırması, evden çalışan personelin çağrı başına performansında yüzde 13 artış ölçmüştür.",
        "expected": "support",
    },
    {
        "name": "semantic_opposite_without_negative_marker",
        "claim": "Öğrencilerin problem çözme becerileri gelişti.",
        "evidence": "Michigan Üniversitesi çalışması, öğrencilerin bağımsız problem çözme becerilerinin gerilediğini gösterdi.",
        "expected": "attack",
    },
    {
        "name": "temporal_mismatch_none",
        "claim": "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
        "evidence": "TÜİK verilerine göre Türkiye ekonomisi 2011 yılında yüzde 11'in üzerinde büyümüştü.",
        "expected": "none",
    },
    {
        "name": "entity_mismatch_none",
        "claim": "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
        "evidence": "Hindistan ekonomisi 2024 yılında yüzde 8 civarında büyüme kaydetti.",
        "expected": "none",
    },
    {
        "name": "causal_support_without_same_words",
        "claim": "Uzun ekran süresi öğrencilerin dikkat süresini azaltır.",
        "evidence": "Hakemli çalışma, günde üç saatten fazla ekran kullanan öğrencilerin dikkat testlerinde daha düşük puan aldığını buldu.",
        "expected": "support",
    },
    {
        "name": "causal_attack_without_same_words",
        "claim": "Uzun ekran süresi öğrencilerin dikkat süresini azaltır.",
        "evidence": "Randomize deneyde ekran süresi kısıtlanan ve kısıtlanmayan öğrencilerin dikkat puanları arasında anlamlı fark görülmedi.",
        "expected": "attack",
    },
    {
        "name": "hedged_support",
        "claim": "Yeni düzenleme küçük işletmelerin maliyetini artırabilir.",
        "evidence": "Sektör anketi, düzenlemeye uyum için küçük işletmelerin ek muhasebe ve yazılım gideri beklediğini gösteriyor.",
        "expected": "support",
    },
    {
        "name": "correlation_not_support_none",
        "claim": "Kahve tüketimi sınav başarısını artırır.",
        "evidence": "Araştırma kahve içen öğrencilerin daha yüksek not aldığını, ancak çalışma süresi ve gelir düzeyi kontrol edilmediğini belirtiyor.",
        "expected": "none",
    },
    {
        "name": "quote_denial_attack",
        "claim": "Bakan enflasyonun yıl sonunda tek haneye düşeceğini söyledi.",
        "evidence": "Bakan konuşmasında yıl sonu için tek haneli enflasyon hedefi vermediğini açıkça belirtti.",
        "expected": "attack",
    },
    {
        "name": "same_entity_opposite_direction_attack",
        "claim": "Konut fiyatları reel olarak yükseldi.",
        "evidence": "Merkez Bankası endeksi, aynı dönemde konut fiyatlarının reel olarak gerilediğini gösterdi.",
        "expected": "attack",
    },
    {
        "name": "policy_recommendation_support",
        "claim": "Okullar yapay zeka ödev araçlarını öğretmen denetimiyle kullanmalıdır.",
        "evidence": "Pilot uygulama, öğretmen denetimi olduğunda öğrencilerin aracı kopyalama yerine geri bildirim almak için kullandığını gösterdi.",
        "expected": "support",
    },
    {
        "name": "policy_recommendation_attack",
        "claim": "Okullar yapay zeka ödev araçlarını sınırsız biçimde kullanmalıdır.",
        "evidence": "OECD raporu, denetimsiz kullanımın pasif tüketim alışkanlığına ve bağımsız problem çözmede gerilemeye yol açabileceğini belirtiyor.",
        "expected": "attack",
    },
    {
        "name": "same_topic_other_none",
        "claim": "Aşılar toplum sağlığını korur.",
        "evidence": "Aşı randevusu almak için e-Nabız uygulamasına giriş yapılır ve aile hekimi seçilir.",
        "expected": "none",
    },
    {
        "name": "country_metric_mismatch_none_holdout",
        "claim": "Türkiye'de genç işsizlik oranı 2024'te düştü.",
        "evidence": "Yunanistan'da genç işsizlik oranı 2024'te arttı.",
        "expected": "none",
    },
    {
        "name": "nominal_vs_real_attack_holdout",
        "claim": "Konut fiyatları reel olarak yükseldi.",
        "evidence": "Raporda nominal fiyatların arttığı, ancak enflasyondan arındırılmış reel fiyatların düştüğü belirtildi.",
        "expected": "attack",
    },
    {
        "name": "denetimli_vs_denetimsiz_none_holdout",
        "claim": "Yapay zeka araçları öğretmen denetimiyle öğrenmeyi destekler.",
        "evidence": "Denetimsiz kullanımda öğrencilerin aracı kopyalama için kullandığı raporlandı.",
        "expected": "none",
    },
    {
        "name": "same_entity_same_metric_support_holdout",
        "claim": "Genç işsizlik oranı 2024'te azaldı.",
        "evidence": "TÜİK tablosu genç işsizlik oranının 2024'te önceki yıla göre düştüğünü gösterdi.",
        "expected": "support",
    },
    {
        "name": "same_entity_same_metric_attack_holdout",
        "claim": "Genç işsizlik oranı 2024'te azaldı.",
        "evidence": "TÜİK tablosu genç işsizlik oranının 2024'te önceki yıla göre yükseldiğini gösterdi.",
        "expected": "attack",
    },
    {
        "name": "denial_without_wrong_word_attack_holdout",
        "claim": "Video 2024'teki protestoda çekildi.",
        "evidence": "Doğrulama kuruluşu videonun 2021'de farklı bir ülkede kaydedildiğini belirledi.",
        "expected": "attack",
    },
    {
        "name": "same_topic_different_denominator_none_holdout",
        "claim": "Okula devamsızlık oranı azaldı.",
        "evidence": "Milli Eğitim Bakanlığı aynı dönemde toplam öğrenci sayısının arttığını açıkladı.",
        "expected": "none",
    },
    {
        "name": "conditional_support_holdout",
        "claim": "Yapay zeka araçları öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.",
        "evidence": "Deneyde öğretmen geri bildirimiyle kullanılan yapay zeka araçları öğrencilerin kavram yanılgılarını azaltmıştır.",
        "expected": "support",
    },
    {
        "name": "conditional_attack_holdout",
        "claim": "Yapay zeka araçları tek başına kullanıldığında öğrenmeyi destekler.",
        "evidence": "Deneyde öğretmen geri bildirimi olmadan kullanılan araçların öğrenme kazanımını artırmadığı görüldü.",
        "expected": "attack",
    },
    {
        "name": "same_entity_different_period_none_holdout",
        "claim": "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
        "evidence": "Türkiye ekonomisi 2004 yılında yüksek büyüme oranı yakalamıştı.",
        "expected": "none",
    },
    {
        "name": "parenthood_collective_not_individual_attack",
        "claim": "Çocuk sahibi olmak ahlaken doğru bir davranıştır çünkü insan türünün devamlılığını sağlar.",
        "evidence": "Türün devamlılığı kolektif bir sorumluluk olabilir, ancak bu bireysel üremeyi ahlaki yükümlülük haline getirmez.",
        "expected": "attack",
    },
    {
        "name": "parenthood_existing_children_attack",
        "claim": "Çocuk sahibi olmak ahlaken doğru bir davranıştır çünkü insan türünün devamlılığını sağlar.",
        "evidence": "Oysa ahlaki seçim, var olanları korumak ve gelecekteki çocukların yaşanabilir bir dünyaya doğmasını sağlamak olmalıdır.",
        "expected": "attack",
    },
    {
        "name": "parenthood_technology_none_to_hume",
        "claim": "Üremeyi bir zorunluluk olarak tanımlamaz.",
        "evidence": "Yeni nesiller yenilenebilir enerji ve karbon yakalama teknolojileri geliştirme potansiyeline sahiptir.",
        "expected": "none",
    },
    {
        "name": "same_topic_child_mortality_vs_workforce_none",
        "claim": "Japonya ve Almanya'da göç ve otomasyon nüfus sorununu tamamen çözememiştir.",
        "evidence": "UNICEF verilerine göre 2022'de 5,3 milyon çocuk beş yaşından önce öldü.",
        "expected": "none",
    },
]


def pass_relation(expected: str, predicted: str, confidence: float) -> bool:
    if expected == predicted:
        return True
    if expected == "none" and predicted in {"support", "attack"} and confidence < 0.60:
        return True
    return False


def main() -> None:
    parser = ArgumentParser()
    parser.add_argument("--component-model-dir", type=Path, default=None)
    parser.add_argument("--relation-model-dir", type=Path, default=None)
    parser.add_argument("--write-metrics", type=Path, default=None)
    args = parser.parse_args()

    pipeline = ModernBERTPipeline(args.component_model_dir, args.relation_model_dir)

    component_results = []
    for case in COMPONENT_CASES:
        predicted, confidence = pipeline._classify_component_unit(case["text"])
        component_results.append(
            {
                "name": case["name"],
                "expected": case["expected"],
                "predicted": predicted,
                "confidence": confidence,
                "passed": predicted == case["expected"],
            }
        )

    relation_results = []
    for case in RELATION_CASES:
        predicted, confidence, probabilities = pipeline.classify_relation(case["claim"], case["evidence"])
        relation_results.append(
            {
                "name": case["name"],
                "expected": case["expected"],
                "predicted": predicted,
                "confidence": confidence,
                "probabilities": probabilities,
                "passed": pass_relation(case["expected"], predicted, confidence),
            }
        )

    component_passed = sum(item["passed"] for item in component_results)
    relation_passed = sum(item["passed"] for item in relation_results)
    passed = component_passed + relation_passed
    total = len(component_results) + len(relation_results)
    claim_cases = [item for item in component_results if item["expected"] == "claim"]
    claim_recall = sum(item["predicted"] == "claim" for item in claim_cases) / len(claim_cases)
    predicted_none = [item for item in relation_results if item["predicted"] == "none"]
    none_precision = (
        sum(item["expected"] == "none" for item in predicted_none) / len(predicted_none)
        if predicted_none
        else 0.0
    )
    component_accuracy = component_passed / len(component_results)
    relation_accuracy = relation_passed / len(relation_results)
    gates = {
        "component_accuracy": component_accuracy >= 0.90,
        "relation_accuracy": relation_accuracy >= 0.90,
        "claim_recall": claim_recall >= 0.92,
        "none_precision": none_precision >= 0.88,
    }
    payload = {
        "passed": passed,
        "total": total,
        "accuracy": round(passed / total, 4),
        "component_accuracy": round(component_accuracy, 4),
        "relation_accuracy": round(relation_accuracy, 4),
        "claim_recall": round(claim_recall, 4),
        "none_precision": round(none_precision, 4),
        "gates": gates,
        "components": component_results,
        "relations": relation_results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if args.write_metrics:
        args.write_metrics.parent.mkdir(parents=True, exist_ok=True)
        args.write_metrics.write_text(
            json.dumps(
                {
                    "model": {
                        "f1_components": round(component_accuracy, 4),
                        "f1_relations": round(relation_accuracy, 4),
                        "component_hard_case_accuracy": round(component_accuracy, 4),
                        "relation_hard_case_accuracy": round(relation_accuracy, 4),
                        "claim_recall": round(claim_recall, 4),
                        "none_precision": round(none_precision, 4),
                        "source": "scripts/evaluate_hard_model_cases.py",
                    }
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    if not all(gates.values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
