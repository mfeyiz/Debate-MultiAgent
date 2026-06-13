import asyncio
from app.services.bert_service import ModernBERTPipeline

def test_opponent():
    pipeline = ModernBERTPipeline()
    topic = "Çocuk yapmak ahlaken doğru bir davranıştır."
    
    sentences = [
        "Çocuk yapmanın \"ahlaken doğru\" olduğu iddiası, en temel etik sorunu atlamaktadır: Rıza.",
        "Yeni bir birey, var olmayı tercih edemez;",
        "ona hayatın yükünü ve kaçınılmaz acılarını dayatmak, onu rızasız bir deneye tabi tutmaktır.",
        "Faydacı hesap, ebeveynin mutluluğu ile çocuğun potansiyel ıstırabını karşılaştırmakta yetersiz kalır.",
        "David Benatar'ın \"Doğmamış Olmak Daha İyidir\" tezinde savunduğu gibi, acının yokluğu bir kazançken, mutluluğun yokluğu bir kayıp değildir.",
        "Ayrıca, nüfus yenilenmesine yapılan vurgu, araçsallaştırmadır: Çocuk, toplumun sorunlarını çözmek için bir araç haline getirilir.",
        "Oysa ahlaki bir eylem, başka bir varlığın yaşamını kendi amaçlarına alet etmemelidir.",
        "UN verileri toplumsal krizleri gösteriyor olabilir;",
        "ancak bu veriler, sorunun çözümünün bireylerin yaşamları üzerinde spekülasyon yapmak değil, mevcut insanların refahını artırmak olduğu gerçeğini değiştirmez."
    ]
    
    print(f"Topic: {topic}\n")
    for s in sentences:
        rel, conf, probs = pipeline.classify_relation(topic, s)
        print(f"Sentence: \"{s}\"")
        print(f"Predicted Relation: {rel.upper()} (Confidence: {conf:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    test_opponent()
