import asyncio
from app.services.bert_service import ModernBERTPipeline

def test_variations():
    pipeline = ModernBERTPipeline()
    
    sentence = "Çocuk yapmak, insan türünün devamlılığını sağlamak ve toplumsal faydayı maksimize etmek açısından ahlaken doğrudur."
    
    topics = [
        "Çocuk yapmak ahlaken doğru bir davranış mı?",
        "Çocuk yapmak ahlaken doğru bir davranıştır.",
        "Çocuk yapmak ahlaken doğru bir davranış",
        "Çocuk yapmak ahlaken yanlış bir davranış mı?",
        "Çocuk sahibi olmak ahlaken doğru bir davranış mı?",
        "Çocuk sahibi olmak ahlaken doğru bir davranıştır."
    ]
    
    print(f"Sentence: \"{sentence}\"\n")
    for topic in topics:
        rel, conf, probs = pipeline.classify_relation(topic, sentence)
        print(f"Topic: \"{topic}\"")
        print(f"Predicted Relation: {rel.upper()} (Confidence: {conf:.3f})")
        print(f"Probabilities: {probs}")
        print("-" * 50)

if __name__ == "__main__":
    test_variations()
