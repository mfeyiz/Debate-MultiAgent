"""Deterministic quality examples for Logos ModernBERT fine-tuning.

The base gold topic file is broad, but it under-represents the failure modes
that made the product overconfident: numeric public-data claims, official-source
attribution, manipulative unsupported phrasing, irrelevant snippets, and
same-topic/different-metric evidence. These examples intentionally bias the
candidate models toward safer recall and stricter relation neutrality.
"""

from __future__ import annotations

from itertools import product


OFFICIAL_SOURCES = ["TÜİK", "TCMB", "OECD", "WHO", "Dünya Bankası", "Eurostat"]
ECONOMIC_METRICS = [
    ("enflasyon", "yüzde 44,38", "2024"),
    ("ekonomik büyüme", "yüzde 3,2", "2024"),
    ("politika faizi", "yüzde 50", "2024"),
    ("işsizlik oranı", "yüzde 8,5", "2025"),
    ("döviz kuru", "32,40 TL", "2024"),
]
MANIPULATIVE_PATTERNS = [
    "Kaynaklara göre herkes bu gerçeği saklıyor ve medya bunu yayınlamıyor.",
    "Bu konuda kesinlikle herkes kandırılıyor, gerçekler bilinçli olarak gizleniyor.",
    "Hiç kimse bunu konuşmuyor çünkü güçlü çevreler skandalı örtbas ediyor.",
    "Uzmanlar adını vermeden bunun büyük bir oyun olduğunu söylüyor.",
]
BOILERPLATE_SNIPPETS = [
    "Ana sayfa, hakkımızda, iletişim, üyelik ve son haberler bağlantıları listeleniyor.",
    "Çerez tercihleri, gizlilik politikası ve site kullanım şartları metni görüntüleniyor.",
    "Kategori sayfasında siyaset, ekonomi, spor ve kültür başlıkları yer alıyor.",
    "Bülten aboneliği kutusu, reklam alanı ve sosyal medya bağlantıları bulunuyor.",
]


def component_examples() -> list[dict[str, str]]:
    examples: list[dict[str, str]] = []

    for source, (metric, value, year) in product(OFFICIAL_SOURCES, ECONOMIC_METRICS):
        examples.append(
            {
                "text": f"{source} verilerine göre Türkiye'de {metric} {year} yılında {value} oldu.",
                "label": "claim",
                "topic": "quality_numeric_claim",
            }
        )

    for metric, value, year in ECONOMIC_METRICS:
        examples.extend(
            [
                {
                    "text": f"Türkiye'de {metric} {year} yılında {value} olarak gerçekleşti.",
                    "label": "claim",
                    "topic": "quality_numeric_claim",
                },
                {
                    "text": f"{year} verileri {metric} göstergesinin {value} düzeyinde olduğunu ortaya koydu.",
                    "label": "claim",
                    "topic": "quality_numeric_claim",
                },
                {
                    "text": f"Resmi açıklamaya göre {metric} {value} seviyesine ulaştı.",
                    "label": "claim",
                    "topic": "quality_numeric_claim",
                },
            ]
        )
        examples.append(
            {
                "text": f"Resmi bültende {metric} için {year} değeri {value} olarak raporlandı.",
                "label": "evidence",
                "topic": "quality_official_evidence",
            }
        )

    claim_templates = [
        "Yapay zeka destekli ödev araçları, lise öğrencilerinin öğrenme kalitesini kesinlikle artırır.",
        "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır.",
        "Yapay zeka araçları öğrencilerin öğrenme kalitesini artırır.",
        "Kişiselleştirilmiş geri bildirim sağlayan yapay zeka araçları öğrenme kalitesini artırabilir.",
        "Yapay zeka destekli ödev sistemleri öğrencilerin bireysel öğrenme hızına uyum sağlar.",
        "Bu kadar düşük başarı puanına rağmen yapay zeka ödevlerinin öğrenmeyi artırdığı nasıl söylenebilir?",
        "Bağımsız problem çözme becerileri gerilerken bu araçların öğrenme kalitesini artırdığı iddia edilemez.",
        "Kısa vadeli sınav başarısı artsa bile derin öğrenmenin geliştiğini söylemek mümkün değildir.",
        "Öğrenci hatasını anında düzeltmek gerçekten kalıcı öğrenme anlamına gelir mi?",
        "Kaynak rapor kurumun kendi değerlendirmesine dayanıyorsa bu sonuç tarafsız kabul edilemez.",
        "Yüzeysel ezberleme arttığında öğrenme kalitesinin yükseldiği söylenemez.",
        "Yeni düzenlemenin küçük işletmeler üzerindeki maliyeti artırması bekleniyor.",
        "Uzun ekran süresi öğrencilerin dikkat süresini azaltır.",
        "Okullar yapay zeka ödev araçlarını sınırsız biçimde değil, öğretmen denetimiyle kullanmalıdır.",
        "Kahve tüketimi sınav başarısını artırır.",
        "Konut fiyatları reel olarak yükseldi.",
        "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
        "Enflasyon 2024 sonunda yüzde 44,38 olarak gerçekleşti.",
        "Merkez Bankası politika faizini yüzde 50 seviyesinde tuttu.",
        "OECD raporu öğrencilerin matematik başarısında düşüş olduğunu belirtti.",
        "WHO verileri kızamık vakalarının son iki yılda arttığını gösterdi.",
        "TÜİK işsizlik oranının 2025 başında yüzde 8,5 olduğunu açıkladı.",
        "Kızamık vakaları son iki yılda yüzde 30 arttı.",
        "Öğrencilerin PISA matematik puanı 2022 döngüsünde düştü.",
        "Konut fiyat endeksi 2024 yılında reel olarak geriledi.",
        "İhracatın milli gelir içindeki payı 2024 yılında arttı.",
        "Aşılanma oranı düştüğünde salgın riski yükselir.",
        "Yapay zeka destekli ödev araçları öğrenme kalitesini her koşulda artırmaz.",
        "Yapay zeka araçları ancak öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.",
        "Yapay zeka araçları öğretmen denetimiyle kullanılırsa öğrenme kazanımlarını artırabilir.",
        "Okullar yapay zeka araçlarını koşulsuz değil, öğretmen rehberliğiyle kullanmalıdır.",
    ]
    examples.extend({"text": text, "label": "claim", "topic": "quality_claim"} for text in claim_templates)

    evidence_templates = [
        "Khan Academy raporu, yapay zeka destekli asistanın öğrencilerin hatalarını daha hızlı tespit ettiğini gösteriyor.",
        "Deneysel çalışma, yapay zeka destekli geri bildirimin öğrenme kazanımlarını yüzde 30'a varan oranlarda iyileştirdiğini bildirdi.",
        "Araştırmada kişiselleştirilmiş geri bildirim alan öğrencilerin kavramsal eksiklerini daha hızlı kapattığı raporlandı.",
        "TÜİK'in 2024 ulusal hesaplar bülteni büyümenin yüzde 3,2 olduğunu gösteriyor.",
        "TCMB karar metni politika faizinin yüzde 50 düzeyinde sabit tutulduğunu bildiriyor.",
        "OECD eğitim raporu PISA matematik puanlarında düşüş eğilimi olduğunu yazıyor.",
        "WHO haftalık bülteni kızamık vakalarındaki artışı ülke bazında listeliyor.",
        "Eurostat veri tabanı aynı dönemde işsizlik oranlarını karşılaştırmalı veriyor.",
    ]
    examples.extend({"text": text, "label": "evidence", "topic": "quality_evidence"} for text in evidence_templates)

    examples.extend({"text": text, "label": "other", "topic": "quality_manipulation"} for text in MANIPULATIVE_PATTERNS)
    examples.extend({"text": text, "label": "other", "topic": "quality_boilerplate"} for text in BOILERPLATE_SNIPPETS)

    neutral_templates = [
        "Bu metinde tartışmanın bağlamı ve önceki konuşmacıların sırası özetlenmektedir.",
        "Platform ayarlarında otomatik ilerleme ve bildirim tercihleri değiştirilebilir.",
        "Raporun giriş bölümünde yöntem ve kapsam sınırlılıkları açıklanmaktadır.",
        "Yazar, analizden önce kullanılan kavramların tanımını vermektedir.",
        "Çalışmanın örneklemi yalnızca üç şehirden seçildiği için bulguların genellenebilirliği sınırlıdır.",
        "Araştırma tasarımı gözlemsel olduğu için sonuçlar nedensellik kanıtı olarak yorumlanmamalıdır.",
        "Bu bölümde veri setinin kapsamı, örneklem seçimi ve ölçüm sınırlılıkları açıklanmaktadır.",
    ]
    examples.extend({"text": text, "label": "other", "topic": "quality_background"} for text in neutral_templates)
    return examples


def relation_examples() -> list[dict[str, str]]:
    pairs: list[dict[str, str]] = []

    contradiction_pairs = [
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "TÜİK verilerine göre Türkiye ekonomisi 2024 yılında yüzde 3,2 büyüdü.",
        ),
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "TÜİK aynı yıl büyümenin yüzde 3,2 olduğunu açıkladı.",
        ),
        (
            "Enflasyon 2024 sonunda yüzde 20 oldu.",
            "TÜİK bülteni 2024 sonunda yıllık enflasyonu yüzde 44,38 olarak açıkladı.",
        ),
        (
            "Enflasyon 2024 sonunda yüzde 44,38 oldu.",
            "Resmi bültende 2024 yıl sonu enflasyonunun yüzde 20 olduğu yazıyor.",
        ),
        (
            "Enflasyon yıl sonunda yüzde 44,38 olarak gerçekleşti.",
            "Aynı resmi tabloda yıl sonu enflasyonu yüzde 20 olarak verildi.",
        ),
        (
            "Merkez Bankası politika faizini yüzde 15'e indirdi.",
            "TCMB karar metni politika faizinin yüzde 50 seviyesinde sabit tutulduğunu gösteriyor.",
        ),
    ]
    for claim, evidence in contradiction_pairs:
        pairs.append({"claim_text": claim, "evidence_text": evidence, "label": "attack", "topic": "quality_numeric_attack"})

    support_pairs = [
        (
            "Okullar yapay zeka ödev araçlarını öğretmen denetimiyle kullanmalıdır.",
            "Pilot uygulama, öğretmen denetimi olduğunda öğrencilerin aracı kopyalama yerine geri bildirim almak için kullandığını gösterdi.",
        ),
        (
            "Okullar yapay zeka ödev araçlarını kontrollü biçimde kullanmalıdır.",
            "Denetimli sınıf deneyinde öğretmen rehberliği olan grupta öğrenciler aracı geri bildirim almak için kullandı ve ödev kalitesi arttı.",
        ),
        (
            "Yapay zeka ödev araçları öğretmen gözetimiyle öğrenmeyi destekleyebilir.",
            "Araştırma, öğretmen gözetimi ve açık yönerge olduğunda öğrencilerin aracı kopyalama yerine hatalarını anlamak için kullandığını raporladı.",
        ),
        (
            "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır mı?",
            "Yapay zeka destekli ödev araçları, lise öğrencilerinin öğrenme kalitesini kesinlikle artırır.",
        ),
        (
            "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır mı?",
            "Bu araçlar, öğrencilere anlık, kişiselleştirilmiş geri bildirim sağlayarak geleneksel sınıf ortamında mümkün olmayan bireysel öğrenme hızına uyum sağlar.",
        ),
        (
            "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır.",
            "Yapay zeka destekli ödev araçları, lise öğrencilerinin öğrenme kalitesini kesinlikle artırır.",
        ),
        (
            "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır.",
            "Örneğin, Khan Academy'nin yapay zeka destekli asistanı öğrencilerin hatalarını anında tespit edip kavramsal eksikliklerini gidermelerine yardımcı olarak öğrenme kazanımlarını yüzde 30'a varan oranlarda iyileştirdiğini gösteren deneysel çalışmalar mevcuttur.",
        ),
        (
            "Enflasyon 2024 sonunda yüzde 44,38 oldu.",
            "TÜİK verileri 2024 yıl sonu enflasyonunu yüzde 44,38 olarak raporladı.",
        ),
        (
            "Merkez Bankası politika faizini yüzde 50 seviyesinde tuttu.",
            "TCMB karar metninde politika faizinin yüzde 50'de sabit bırakıldığı açıklandı.",
        ),
        (
            "OECD raporu öğrencilerin matematik başarısında düşüş olduğunu belirtti.",
            "OECD eğitim raporu matematik performansında önceki döngüye göre düşüş kaydedildiğini yazdı.",
        ),
    ]
    for claim, evidence in support_pairs:
        pairs.append({"claim_text": claim, "evidence_text": evidence, "label": "support", "topic": "quality_support"})

    neutral_pairs = [
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "TÜİK verilerine göre Türkiye ekonomisi 2011 yılında yüzde 11'in üzerinde büyümüştü.",
        ),
        (
            "Enflasyon 2024 sonunda yüzde 44,38 oldu.",
            "TÜİK verilerine göre enflasyon 2022 yılında yüzde 64,27 olarak gerçekleşmişti.",
        ),
        (
            "Konut fiyatları reel olarak yükseldi.",
            "Merkez Bankası endeksi farklı bir dönemde konut fiyatlarının reel olarak gerilediğini göstermişti.",
        ),
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "Hindistan ekonomisi 2024 yılında yüzde 8 civarında büyüme kaydetti.",
        ),
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "Hindistan ekonomisi 2024 yılında yüzde 8 civarında büyüme kaydetti.",
        ),
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "Brezilya ekonomisi 2024 yılında yüzde 3 civarında büyüdü.",
        ),
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "Almanya ekonomisi aynı yıl daralma riskiyle karşı karşıya kaldı.",
        ),
        (
            "TCMB politika faizini yüzde 50 seviyesinde tuttu.",
            "ABD Merkez Bankası aynı dönemde federal fon faizini farklı bir aralıkta tuttu.",
        ),
        (
            "Apple 2024 gelirini artırdı.",
            "Samsung aynı yıl yarı iletken gelirinde farklı bir değişim raporladı.",
        ),
        (
            "Türkiye'de enflasyon 2024 sonunda yüzde 44,38 oldu.",
            "Arjantin'de enflasyon aynı yıl çok daha yüksek bir seviyede gerçekleşti.",
        ),
        (
            "Kahve tüketimi sınav başarısını artırır.",
            "Araştırma kahve içen öğrencilerin daha yüksek not aldığını, ancak çalışma süresi ve gelir düzeyi kontrol edilmediğini belirtiyor.",
        ),
        (
            "Yapay zeka destekli ödev araçları lise öğrencilerinin öğrenme kalitesini artırır mı?",
            "Ana sayfada üyelik, yardım, gizlilik ve son haber bağlantıları yer alıyor.",
        ),
        (
            "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
            "PolitiFact ana sayfasında üyelik, son haberler ve kategori bağlantıları bulunuyor.",
        ),
        (
            "Enflasyon 2024 sonunda yüzde 44,38 oldu.",
            "Kaynaklara göre herkes bu gerçeği saklıyor ve medya bunu yayınlamıyor.",
        ),
        (
            "Merkez Bankası politika faizini yüzde 50 seviyesinde tuttu.",
            "TÜİK bülteni aynı dönemde işsizlik oranının yüzde 8,5 olduğunu açıkladı.",
        ),
        (
            "OECD raporu öğrencilerin matematik başarısında düşüş olduğunu belirtti.",
            "Sitede çerez tercihleri, gizlilik politikası ve bülten aboneliği bağlantıları yer alıyor.",
        ),
        (
            "Yapay zeka araçları öğretmen denetimiyle öğrenmeyi destekler.",
            "Denetimsiz kullanımda öğrencilerin aracı kopyalama için kullandığı raporlandı.",
        ),
        (
            "Yapay zeka araçları öğretmen rehberliğiyle faydalıdır.",
            "Öğretmen rehberliği olmadan yapılan uygulamada öğrencilerin kopyalama eğilimi arttı.",
        ),
        (
            "Okula devamsızlık oranı azaldı.",
            "Milli Eğitim Bakanlığı aynı dönemde toplam öğrenci sayısının arttığını açıkladı.",
        ),
        (
            "Genç işsizlik oranı düştü.",
            "İŞKUR aynı dönemde başvuru yapan genç sayısının arttığını bildirdi.",
        ),
    ]
    for claim, evidence in neutral_pairs:
        pairs.append({"claim_text": claim, "evidence_text": evidence, "label": "none", "topic": "quality_neutral"})

    attack_pairs = [
        (
            "Uzun ekran süresi öğrencilerin dikkat süresini azaltır.",
            "Randomize deneyde ekran süresi kısıtlanan ve kısıtlanmayan öğrencilerin dikkat puanları arasında anlamlı fark görülmedi.",
        ),
        (
            "Bakan enflasyonun yıl sonunda tek haneye düşeceğini söyledi.",
            "Bakan konuşmasında yıl sonu için tek haneli enflasyon hedefi vermediğini açıkça belirtti.",
        ),
        (
            "Konut fiyatları reel olarak yükseldi.",
            "Merkez Bankası endeksi, aynı dönemde konut fiyatlarının reel olarak gerilediğini gösterdi.",
        ),
        (
            "Konut fiyatları reel olarak arttı.",
            "Aynı dönem için Merkez Bankası reel konut fiyat endeksinde düşüş raporladı.",
        ),
        (
            "Konut fiyatları reel olarak düşmedi, yükseldi.",
            "Aynı döneme ait endeks konut fiyatlarının reel olarak gerilediğini gösteriyor.",
        ),
        (
            "Video 2024'teki protestoda çekildi.",
            "Doğrulama kuruluşu videonun 2021'de farklı bir ülkede kaydedildiğini belirledi.",
        ),
        (
            "Fotoğraf bu haftaki depremden sonra çekildi.",
            "Doğrulama ekibi fotoğrafın 2019'da başka bir ülkede yayımlandığını tespit etti.",
        ),
        (
            "Görüntüler Ankara'daki son eyleme aittir.",
            "Arşiv taraması görüntülerin 2020'de İstanbul'da çekildiğini gösterdi.",
        ),
        (
            "Okullar yapay zeka ödev araçlarını sınırsız biçimde kullanmalıdır.",
            "OECD raporu, denetimsiz kullanımın pasif tüketim alışkanlığına ve bağımsız problem çözmede gerilemeye yol açabileceğini belirtiyor.",
        ),
        (
            "Okullar yapay zeka ödev araçlarını sınırsız biçimde kullanmalıdır.",
            "Pilot uygulama, öğretmen denetimi olmadan öğrencilerin aracı kopyalama için kullandığını ve öğrenme kazanımlarının zayıfladığını gösterdi.",
        ),
        (
            "Yeni düzenleme küçük işletmelerin maliyetini düşürebilir.",
            "Sektör anketi, düzenlemeye uyum için küçük işletmelerin ek muhasebe ve yazılım gideri beklediğini gösteriyor.",
        ),
    ]
    for claim, evidence in attack_pairs:
        pairs.append({"claim_text": claim, "evidence_text": evidence, "label": "attack", "topic": "quality_attack"})

    extra_support_pairs = [
        (
            "Yeni düzenleme küçük işletmelerin maliyetini artırabilir.",
            "Sektör anketi, düzenlemeye uyum için küçük işletmelerin ek muhasebe ve yazılım gideri beklediğini gösteriyor.",
        ),
    ]
    for claim, evidence in extra_support_pairs:
        pairs.append({"claim_text": claim, "evidence_text": evidence, "label": "support", "topic": "quality_support"})

    mixed_pairs = [
        (
            "Yapay zeka araçları öğrencilerin öğrenme kalitesini artırır.",
            "Khan Academy raporu kısa vadeli geri bildirimin başarıyı artırdığını, fakat aşırı kullanımın bağımsız problem çözmeyi zayıflatabildiğini belirtiyor.",
            "neutral",
        ),
        (
            "Uzaktan çalışma çalışan verimliliğini artırır.",
            "Stanford çalışması verimlilik artışı bulurken, ekip içi yaratıcılık ve yeni çalışan uyumunda sınırlılıklar raporlamıştır.",
            "neutral",
        ),
    ]
    for claim, evidence, label in mixed_pairs:
        pairs.append({"claim_text": claim, "evidence_text": evidence, "label": label, "topic": "quality_mixed"})

    return pairs
