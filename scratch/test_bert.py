import asyncio
import time
from app.services.bert_service import ModernBERTPipeline

def test_bert():
    print("Loading ModernBERT Pipeline...")
    t0 = time.time()
    pipeline = ModernBERTPipeline()
    t1 = time.time()
    print(f"Pipeline loaded in {t1 - t0:.2f} seconds.")
    
    print("Running analyze...")
    target = "Çocuk sahibi olmak ahlaken doğru bir davranıştır çünkü insan soyunun devamını sağlar."
    source = "Soyun devamını sağlamak kolektif bir sorumluluktur, bireysel bir ahlaki zorunluluk yaratmaz."
    
    result = pipeline.analyze(source, target)
    print("\nRESULTS:")
    print("Overall Strength:", result.overall_strength)
    print("Feedback:", result.feedback)
    
    print("\nComponents:")
    for c in result.components:
        print(f"- Text: '{c.text}' | Type: {c.component_type} | Conf: {c.confidence}")
        
    print("\nRelations:")
    for r in result.relations:
        print(f"- Source Component: '{r.source_component.text}'")
        print(f"  Target Component: '{r.target_component.text}'")
        print(f"  Relation: {r.relation_type} | Conf: {r.confidence}")

if __name__ == "__main__":
    test_bert()
