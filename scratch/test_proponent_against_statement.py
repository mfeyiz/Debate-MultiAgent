import asyncio
from app.services.bert_service import ModernBERTPipeline

def test_proponent():
    pipeline = ModernBERTPipeline()
    topic = "Çocuk yapmak ahlaken doğru bir davranıştır."
    
    sentences = [
        "Çocuk yapmak, insan türünün devamlılığını sağlamak ve toplumsal faydayı maksimize etmek açısından ahlaken doğrudur.",
        "Biyolojik bir zorunluluk olan üreme, aynı zamanda bireylere sorumluluk, fedakârlık ve empati gibi erdemleri kazandırarak toplumun etik temellerini güçlendirir.",
        "John Stuart Mill’in faydacı ahlak anlayışına göre, bir eylem en fazla sayıda insana en yüksek mutluluğu sağlıyorsa doğrudur;",
        "yeni bir hayatın varlığı, topluma katacağı potansiyel katkılar ve ebeveynin yaşamına kattığı anlam düşünüldüğünde, çocuk yapmanın net faydası olumsuzluklarından ağır basar.",
        "UN verilerine göre, nüfus yenilenme oranının altına düşen toplumlarda ekonomik ve sosyal krizler kaçınılmaz hale gelmektedir;",
        "bu nedenle, bilinçli ve sorumlu ebeveynlik koşuluyla çocuk yapmak, hem bireysel hem de kolektif ahlaki bir yükümlülükür."
    ]
    
    print(f"Topic: {topic}\n")
    for s in sentences:
        rel, conf, probs = pipeline.classify_relation(topic, s)
        print(f"Sentence: \"{s}\"")
        print(f"Predicted Relation: {rel.upper()} (Confidence: {conf:.3f})")
        print("-" * 50)

if __name__ == "__main__":
    test_proponent()
