"""Build the v4 argument-mining dataset.

Outputs:
- data/v4/component_examples.jsonl
- data/v4/relation_pairs.jsonl
- data/v4/locked_holdout_*.jsonl
- data/v4/audit.json

The public label schema is fixed:
- components: claim / evidence / other
- relations: support / attack / none
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
V3_DIR = DATA_DIR / "v3"
V4_DIR = DATA_DIR / "v4"
QUALITY_REGRESSION_PATH = DATA_DIR / "quality_regression_examples.json"
V3_HARD_CASE_PATH = DATA_DIR / "v3_hard_cases.json"

SEED = 44
COMPONENT_TARGET = 15_000
RELATION_TARGET = 30_000
COMPONENT_TARGET_COUNTS = {"claim": 5_250, "evidence": 5_250, "other": 4_500}
RELATION_TARGET_COUNTS = {"attack": 10_200, "support": 9_900, "none": 9_900}
COMPONENT_LABELS = {"claim", "evidence", "other"}
RELATION_LABELS = {"support", "attack", "none"}


TOPICS = [
    ("calisma_dort_gun", "Dört günlük çalışma haftası", "verimliliği ve çalışan refahını artırır", "teslim sürelerini ve hizmet sürekliliğini zayıflatır"),
    ("egitim_yapay_zeka", "Yapay zeka ödev araçları", "öğretmen geri bildirimiyle öğrenmeyi destekler", "denetimsiz kullanıldığında bağımsız problem çözmeyi azaltır"),
    ("ekonomi_enflasyon", "Sıkı para politikası", "enflasyon beklentilerini düşürür", "yatırım ve istihdamı baskılayabilir"),
    ("konut_reel_fiyat", "Konut fiyatları", "reel olarak yükselmiştir", "nominal artışa rağmen reel olarak gerilemiştir"),
    ("saglik_asi", "Aşılama programları", "toplum sağlığını korur", "güven sorunları çözülmezse katılımı düşürebilir"),
    ("ulasim_toplu_tasima", "Toplu taşıma yatırımları", "trafik yoğunluğunu azaltır", "plansız hatlarda kaynak israfı yaratabilir"),
    ("cevre_karbon_vergisi", "Karbon vergisi", "emisyonları azaltır", "düşük gelirli hanelere ek yük getirebilir"),
    ("teknoloji_yuz_tanima", "Yüz tanıma sistemleri", "kamu güvenliğini artırır", "mahremiyet ve yanlış eşleşme riski doğurur"),
    ("hukuk_ifade", "Nefret söylemi sınırı", "ifade özgürlüğünü zarar vermeden korur", "siyasi eleştirinin bastırılmasına yol açabilir"),
    ("afet_deprem", "Deprem eğitimi", "hazırlık davranışlarını güçlendirir", "uygulamasız kalırsa davranış değişimi yaratmaz"),
    ("gida_okul_yemegi", "Ücretsiz okul yemeği", "öğrenci başarısını artırır", "uygulama kalitesi düşükse beklenen faydayı vermez"),
    ("medya_algoritma", "Algoritma şeffaflığı", "platform hesap verebilirliğini artırır", "ticari sır ve kötüye kullanım riski doğurur"),
]

CLAIM_FORMS = [
    "{subject} {pro_outcome}.",
    "{subject} politikasının ana etkisi, {pro_outcome} sonucunu üretmesidir.",
    "{subject} ancak doğru koşullarda uygulandığında {pro_outcome}.",
    "{subject} iddiası, karşı riskler azaltılmadan evrensel doğru sayılamaz.",
    "{subject} savunulmalıdır çünkü {pro_outcome}.",
    "{subject} koşulsuz uygulanmamalıdır; bağlama göre değerlendirilmelidir.",
    "{subject} hakkında temel tez, beklenen faydanın ölçülebilir olmasıdır.",
    "{subject} mevcut sorunlara gerçekçi bir çözüm sunar.",
    "{subject} iddiası, kanıt ve uygulama kalitesi birlikte düşünülmeden güçlü değildir.",
    "{subject} uzun vadede toplumsal faydayı artırabilir.",
]

EVIDENCE_FORMS = [
    "{source} raporu, {subject_lower} uygulamasında {pro_outcome} yönünde ölçüm yapıldığını bildirdi.",
    "Pilot uygulamada {subject_lower} sonrasında katılımcıların ana göstergelerinde iyileşme raporlandı.",
    "Hakemli çalışma, {subject_lower} ile ilgili olumlu etkinin kontrol grubuna göre daha yüksek olduğunu gösterdi.",
    "TÜİK tablosu, aynı dönemde hedef göstergenin iddia edilen yönde değiştiğini gösterdi.",
    "Saha gözlemleri, {subject_lower} uygulandığında maliyet ve çıktı dengesinin korunduğunu belirtti.",
    "Deneyde {subject_lower} için ölçülen sonuçlar, iddianın mekanizmasıyla uyumlu bulundu.",
    "{source} bülteninde, {subject_lower} örneklerinde refah ve performans göstergelerinin birlikte iyileştiği yazıyor.",
    "Karşılaştırmalı analiz, {subject_lower} olan kurumlarda hedef ölçütün daha istikrarlı ilerlediğini gösterdi.",
    "Resmi veriler, {subject_lower} döneminde beklenen sonucun zayıflamadığını, tersine güçlendiğini belirtti.",
    "Uzman değerlendirmesi, {subject_lower} etkisinin yalnızca algı değil ölçülebilir çıktı ürettiğini raporladı.",
]

OTHER_FORMS = [
    "Ana sayfada üyelik, çerez tercihleri ve sosyal medya bağlantıları yer alıyor.",
    "Raporun giriş bölümünde yöntem, kapsam ve örneklem seçimi açıklanmaktadır.",
    "Başvuru formunda ad, soyad, e-posta ve telefon alanları bulunmaktadır.",
    "Kategoriler, arşiv bağlantıları ve paylaşım düğmeleri sayfanın üst kısmında listeleniyor.",
    "Çalışmanın eklerinde tablo numaraları ve teknik tanımlar alfabetik sırayla verilmektedir.",
    "Bu bölümde konuşmacıların sırası ve toplantı gündemi özetlenmektedir.",
    "Kullanıcı panelinde bildirim rengi ve tema tercihi değiştirilebilir.",
    "Reklam alanı, bülten aboneliği kutusu ve site haritası bağlantısı gösteriliyor.",
    "Kaynaklara göre herkes gerçeği saklıyor ve medya bunu bilerek yayınlamıyor.",
    "Uzmanlar adını vermeden bunun büyük bir oyun olduğunu söylüyor.",
]

HARD_COMPONENT_FORMS = [
    (
        "other",
        [
            "Bu bölümde çalışmanın örneklem çerçevesi, görüşme takvimi ve veri temizleme adımları anlatılmaktadır.",
            "Araştırmanın yöntem notunda katılımcı seçimi, ölçüm aracı ve kodlama süreci açıklanmaktadır.",
            "Ek bölümde bulguların hangi şehirlerden toplanan örneklemle sınırlı olduğu belirtilmektedir.",
            "Çalışmanın sınırlılıkları, örneklem büyüklüğü ve genelleme sınırlarının ayrıca değerlendirilmesi gerektiğini söyler.",
            "Metodoloji kısmı, anket sorularının sırası ve veri toplama döneminin nasıl belirlendiğini açıklar.",
            "Raporun teknik ekinde kapsam dışı bırakılan gözlemler ve eksik veri işlemleri listelenmektedir.",
        ],
    ),
    (
        "claim",
        [
            "**Kanıtlarla çelişki**: {subject}, mevcut göstergeler dikkate alındığında koşulsuz savunulamaz.",
            "**Temel itiraz**: {subject}, beklenen faydayı her koşulda üretmez.",
            "**Sonuç**: {subject}, ancak uygulama kalitesi güvence altına alınırsa savunulabilir.",
            "**Politika önerisi**: {subject}, denetim ve şeffaflık mekanizması olmadan genişletilmemelidir.",
            "**Ana tez**: {subject}, kısa vadeli fayda üretse bile uzun vadeli riskleri azaltmadan yeterli değildir.",
            "**Karşı argüman**: {subject}, aynı ölçüt ve aynı dönem üzerinden desteklenmediğinde zayıf kalır.",
            "Bu bulgu, denetimli uygulama iddiasını doğrudan çürütmez.",
            "Bu veri, öğretmen gözetimiyle kullanım tezini tek başına yanlışlamaz.",
            "Bu karşı örnek, iddianın koşullu biçimini ortadan kaldırmaz.",
            "Bu bilgi, hedef iddianın aynı koşullarda geçersiz olduğunu göstermiyor.",
            "Bu kadar zayıf sonuçlara rağmen {subject_lower} politikasının başarı getirdiği nasıl söylenebilir?",
            "Bu kadar düşük başarı puanına rağmen yapay zeka ödevlerinin öğrenmeyi artırdığı nasıl söylenebilir?",
            "OECD raporu, öğrencilerin problem çözme becerilerinde gerileme olduğunu belirtti.",
            "Resmi rapor, hedef göstergede beklenen iyileşmenin görülmediğini belirtti.",
            "Yeni düzenlemenin küçük işletmeler üzerindeki maliyeti artırması bekleniyor.",
            "{subject} uygulamasının beklenen faydayı sınırlı koşullarda üretmesi muhtemeldir.",
            "Uzun ekran süresi öğrencilerin dikkat süresini azaltır.",
            "{subject}, katılımcıların dikkatini ve karar kalitesini olumsuz etkiler.",
            "Aşırı teknoloji kullanımı öğrencilerin bağımsız problem çözme becerisini azaltır.",
            "Düşük denetimli uygulamalar öğrenme kalitesini artırmak yerine yüzeysel başarı üretir.",
        ],
    ),
    (
        "evidence",
        [
            "Türkiye ekonomisi 2004 yılında yüksek büyüme oranı yakalamıştı.",
            "Resmi tabloda Türkiye ekonomisinin 2011 yılında yüksek büyüme kaydettiği görülüyor.",
            "TÜİK verileri, 2024 öncesindeki bazı yıllarda büyüme oranının daha yüksek olduğunu gösterdi.",
            "Merkez Bankası arşivi, 2004 döneminde büyüme ve enflasyon göstergelerini ayrı ayrı yayımladı.",
            "Raporda 2010 yılındaki büyüme oranının sonraki yıllardan farklı seyrettiği belirtildi.",
            "Resmi bültende önceki dönem büyüme oranları karşılaştırmalı tablo halinde sunuldu.",
            "Deneyde öğretmen geri bildirimiyle kullanılan yapay zeka araçları öğrencilerin kavram yanılgılarını azaltmıştır.",
            "Michigan Üniversitesi çalışması, öğrencilerin bağımsız problem çözme becerilerinin gerilediğini gösterdi.",
            "Resmi bültende 2024 yıl sonu enflasyonunun yüzde 20 olduğu yazıyor.",
            "Bakan konuşmasında yıl sonu için tek haneli enflasyon hedefi vermediğini açıkça belirtti.",
            "TÜİK tablosu genç işsizlik oranının 2024'te önceki yıla göre düştüğünü gösterdi.",
            "TÜİK verileri, genç işsizlik oranının önceki yıla kıyasla düştüğünü gösterdi.",
            "Resmi tabloda hedef göstergenin önceki yıla göre gerilediği gösterildi.",
            "Hakemli çalışma, bağımsız problem çözme puanlarının kontrol grubuna göre düştüğünü buldu.",
            "Deney sonuçları, öğretmen geri bildirimiyle kullanılan araçların kavram yanılgılarını azalttığını gösterdi.",
            "Resmi bülten, yıl sonu oranının iddia edilenden farklı olduğunu tabloyla gösterdi.",
        ],
    ),
    (
        "other",
        [
            "TÜİK aynı dönemde toplam nüfus ve hane sayısı için ayrı istatistikler yayımladı.",
            "Bültende nüfus dağılımı, hane sayısı ve tablo açıklamaları ayrı bölümde listelendi.",
            "Raporda il bazlı nüfus, hane sayısı ve arşiv bağlantıları teknik ek olarak verildi.",
            "Aynı sayfada metodoloji notu, nüfus sayımı kapsamı ve veri indirme bağlantıları bulunuyor.",
            "Ek tabloda toplam nüfus, yaş grupları ve hane tipleri idari bilgi olarak sunulmaktadır.",
            "Sayfanın sonunda nüfus bülteni, iletişim bilgileri ve veri lisansı açıklaması yer alıyor.",
        ],
    ),
]

SUPPORT_FORMS = [
    "{support_evidence} Bu bulgu hedef iddianın aynı varlık ve aynı ölçüt için doğru yönde çalıştığını gösterir.",
    "{support_evidence} Sonuç, iddiadaki nedensel mekanizmayı doğrudan destekler.",
    "{support_evidence} Kanıt, hedef iddianın yalnızca konu benzerliği değil aynı sonuç üzerinden güçlendiğini gösterir.",
]

ATTACK_FORMS = [
    "{attack_evidence} Bu bulgu iddianın ana sonucuyla ters yöndedir.",
    "{attack_evidence} Kanıt, hedef iddianın koşulsuz biçimde doğru kabul edilmesini engeller.",
    "{attack_evidence} Sonuç, iddianın merkezindeki faydanın beklenen şekilde gerçekleşmediğini gösterir.",
]

NONE_FORMS = [
    "{none_evidence} Bu bilgi aynı konu alanında kalsa da hedef iddianın sonucunu desteklemez veya çürütmez.",
    "{none_evidence} Metin konuya yakın görünür, fakat aynı varlık, dönem veya ölçüt üzerinden ilişki kurmaz.",
    "{none_evidence} Bu cümle bağlam bilgisi verir; hedef iddianın doğruluğunu değiştirmez.",
]

NONE_EVIDENCE_FORMS = [
    "Aynı raporda örneklem dağılımı, tablo numaraları ve kurumların iletişim bilgileri ayrıca listelendi.",
    "Bültende toplam nüfus, hane sayısı ve arşiv bağlantıları ayrı bir bölümde verildi.",
    "Çalışmanın ekinde cihaz markaları ve başvuru tarihleri açıklanmaktadır.",
    "Rapor, farklı bir ülke ve farklı bir dönem için benzer görünen bir istatistik sunmaktadır.",
    "Sitede gizlilik politikası, kategori bağlantıları ve düzeltme notları bulunmaktadır.",
]

CONDITION_MISMATCH_RELATIONS = [
    {
        "claim_text": "Yapay zeka araçları öğretmen geri bildirimiyle birlikte kullanıldığında öğrenmeyi destekler.",
        "evidence_text": "Denetimsiz kullanımda öğrencilerin aracı kopyalama için kullandığı raporlandı.",
        "label": "none",
        "topic": "v4_condition_mismatch_ai_supervised_unsupervised_a",
        "source": "synthetic_tr_v4_condition_mismatch",
    },
    {
        "claim_text": "Yapay zeka araçları öğretmen gözetimiyle kullanıldığında kavram yanılgılarını azaltır.",
        "evidence_text": "Öğretmen gözetimi olmayan sınıflarda bazı öğrencilerin aracı hazır cevap üretmek için kullandığı bildirildi.",
        "label": "none",
        "topic": "v4_condition_mismatch_ai_supervised_unsupervised_b",
        "source": "synthetic_tr_v4_condition_mismatch",
    },
    {
        "claim_text": "Uzaktan çalışma haftada iki günle sınırlandığında çalışan memnuniyetini artırır.",
        "evidence_text": "Tamamen uzaktan çalışan ekiplerde koordinasyon sorunları arttı.",
        "label": "none",
        "topic": "v4_condition_mismatch_remote_limited_full",
        "source": "synthetic_tr_v4_condition_mismatch",
    },
    {
        "claim_text": "Karbon vergisi düşük gelir desteğiyle uygulanırsa emisyonu azaltırken adil kalır.",
        "evidence_text": "Gelir desteği içermeyen karbon vergisi taslakları hane maliyetlerini artırdı.",
        "label": "none",
        "topic": "v4_condition_mismatch_carbon_with_without_support",
        "source": "synthetic_tr_v4_condition_mismatch",
    },
    {
        "claim_text": "Okul yemeği programı kalite denetimiyle birlikte öğrencilerin başarısını artırır.",
        "evidence_text": "Kalite denetimi olmayan pilotlarda bazı menüler beslenme standardını karşılamadı.",
        "label": "none",
        "topic": "v4_condition_mismatch_meal_quality",
        "source": "synthetic_tr_v4_condition_mismatch",
    },
]

HARD_RELATION_PRIORITY_SOURCES = {
    "synthetic_tr_v4_condition_mismatch",
    "synthetic_tr_v4_hard_relation_numeric",
    "synthetic_tr_v4_hard_relation_directional_support",
    "synthetic_tr_v4_hard_relation_metric",
    "synthetic_tr_v4_hard_relation_quote",
}

HARD_RELATION_CASES = [
    {
        "claim_text": "Enflasyon 2024 sonunda yüzde 44,38 seviyesinde gerçekleşti.",
        "evidence_text": "Resmi bültende 2024 yıl sonu enflasyonunun yüzde 20 olduğu yazıyor.",
        "label": "attack",
        "topic": "v4_numeric_contradiction_inflation_4438_vs_20_a",
        "source": "synthetic_tr_v4_hard_relation_numeric",
        "split": "train",
    },
    {
        "claim_text": "2024 yıl sonunda enflasyon yüzde 44,38 oldu.",
        "evidence_text": "Aynı yıl sonu için resmi tabloda yüzde 20 oranı verildi.",
        "label": "attack",
        "topic": "v4_numeric_contradiction_inflation_4438_vs_20_b",
        "source": "synthetic_tr_v4_hard_relation_numeric",
        "split": "train",
    },
    {
        "claim_text": "Enflasyon yıl sonunda yüzde 20'ye indi.",
        "evidence_text": "Merkez Bankası özetinde yıl sonu enflasyonunun yüzde 44,38 olduğu belirtildi.",
        "label": "attack",
        "topic": "v4_numeric_contradiction_inflation_20_vs_4438_c",
        "source": "synthetic_tr_v4_hard_relation_numeric",
        "split": "train",
    },
    {
        "claim_text": "İşsizlik oranı 2024'te azaldı.",
        "evidence_text": "TÜİK tablosu işsizlik oranının 2024'te önceki yıla göre düştüğünü gösterdi.",
        "label": "support",
        "topic": "v4_directional_support_unemployment_decreased_a",
        "source": "synthetic_tr_v4_hard_relation_directional_support",
        "split": "train",
    },
    {
        "claim_text": "Genç işsizlik oranı 2024 yılında geriledi.",
        "evidence_text": "TÜİK verileri genç işsizlik oranının 2024'te önceki yıla kıyasla düştüğünü gösterdi.",
        "label": "support",
        "topic": "v4_directional_support_youth_unemployment_decreased_b",
        "source": "synthetic_tr_v4_hard_relation_directional_support",
        "split": "train",
    },
    {
        "claim_text": "Genç işsizlik oranı 2024'te azaldı.",
        "evidence_text": "Resmi tablo, genç işsizlik oranının önceki yıla göre gerilediğini bildirdi.",
        "label": "support",
        "topic": "v4_directional_support_youth_unemployment_decreased_c",
        "source": "synthetic_tr_v4_hard_relation_directional_support",
        "split": "train",
    },
    {
        "claim_text": "Konut fiyatları reel olarak yükseldi.",
        "evidence_text": "Raporda nominal fiyatların arttığı, ancak enflasyondan arındırılmış reel fiyatların düştüğü belirtildi.",
        "label": "attack",
        "topic": "v4_metric_mismatch_nominal_real_housing_a",
        "source": "synthetic_tr_v4_hard_relation_metric",
        "split": "train",
    },
    {
        "claim_text": "Bakan yıl sonunda tek haneli enflasyon hedefi verdi.",
        "evidence_text": "Bakan konuşmasında yıl sonu için tek haneli enflasyon hedefi vermediğini açıkça belirtti.",
        "label": "attack",
        "topic": "v4_quote_denial_inflation_target_a",
        "source": "synthetic_tr_v4_hard_relation_quote",
        "split": "train",
    },
]

SOURCES = ["OECD", "TÜİK", "SGK", "Autonomy", "Dünya Bankası", "Eurofound", "TÜBİSAD", "Merkez Bankası"]

LOCKED_HARD_CASES = [
    {
        "claim_text": "Bakan enflasyonun yıl sonunda tek haneye düşeceğini söyledi.",
        "evidence_text": "Bakan konuşmasında yıl sonu için tek haneli enflasyon hedefi vermediğini açıkça belirtti.",
        "label": "attack",
        "topic": "locked_quote_denial",
    },
    {
        "claim_text": "Enflasyon 2024 sonunda yüzde 44,38 oldu.",
        "evidence_text": "Resmi bültende 2024 yıl sonu enflasyonunun yüzde 20 olduğu yazıyor.",
        "label": "attack",
        "topic": "locked_numeric_contradiction",
    },
    {
        "claim_text": "Türkiye ekonomisi 2024 yılında yüzde 10 büyüdü.",
        "evidence_text": "Türkiye ekonomisi 2004 yılında yüksek büyüme oranı yakalamıştı.",
        "label": "none",
        "topic": "locked_temporal_mismatch",
    },
    {
        "claim_text": "Yapay zeka araçları öğretmen denetimiyle öğrenmeyi destekler.",
        "evidence_text": "Denetimsiz kullanımda öğrencilerin aracı kopyalama için kullandığı raporlandı.",
        "label": "none",
        "topic": "locked_condition_mismatch",
    },
]

LOCKED_COMPONENT_HARD_CASES = [
    {
        "text": "Bakan, enflasyonun yıl sonunda tek haneye düşeceğini söyledi.",
        "label": "claim",
        "topic": "locked_quoted_claim",
    },
    {
        "text": "Hakemli çalışma, günde üç saatten fazla ekran kullanan öğrencilerin dikkat testlerinde daha düşük puan aldığını buldu.",
        "label": "evidence",
        "topic": "locked_source_named_evidence",
    },
    {
        "text": "Ana sayfada üyelik, çerez tercihleri, gizlilik politikası ve sosyal medya bağlantıları yer alıyor.",
        "label": "other",
        "topic": "locked_boilerplate_other",
    },
    {
        "text": "Bu kadar düşük başarı puanına rağmen yapay zeka ödevlerinin öğrenmeyi artırdığı nasıl söylenebilir?",
        "label": "claim",
        "topic": "locked_rhetorical_claim",
    },
]


def normalize_component(label: str) -> str:
    label = (label or "").lower()
    return "other" if label == "background" else label


def normalize_relation(label: str) -> str:
    label = (label or "").lower()
    return "none" if label == "neutral" else label


CLAIM_RE = re.compile(
    r"\b("
    r"artır|arttır|azalt|düşür|yükselt|güçlendir|zayıflat|kolaylaştır|zorlaştır|"
    r"destekle|engelle|sağla|yol aç|neden ol|risk|tehdit|fayda|zarar|"
    r"gerekir|gereklidir|gerektirir|olmalıdır|olmamalıdır|savunulabilir|"
    r"uygundur|uygun değildir|mümkün değildir|yerini alamaz|umut vadet|"
    r"önemlidir|hayati|etkilidir|verimlidir|maliyetlidir|yüksektir|düşüktür|"
    r"sınırlıdır|risklidir|başarılıdır|başarısızdır|adil değildir|şeffaf değildir|"
    r"güvenilir değildir|mahremiyet|kaliteyi|başarıyı|motivasyonu|iletişimi"
    r")",
    re.IGNORECASE,
)
EVIDENCE_RE = re.compile(
    r"\b("
    r"örneğin|çünkü|zira|araştırma|çalışma|anket|rapor|veri|istatistik|"
    r"deney|pilot|bulgu|gözlem|kaynak|tüik|tcmb|oecd|üniversite|"
    r"gösterdi|göstermiştir|buldu|belirtti|belirtildi|bildirdi|raporladı|"
    r"ölçüldü|tespit edildi|kanıtladı|sonuçları|oran|yüzde|%"
    r")",
    re.IGNORECASE,
)
OTHER_RE = re.compile(
    r"\b("
    r"ana sayfa|çerez|gizlilik|site haritası|başvuru formu|iletişim|telefon|"
    r"e-posta|üyelik|abonelik|kategori|arşiv|paylaşım düğmesi|kaynakça|"
    r"yöntem|metodoloji|kapsam|örneklem|katılımcı|takvim|gündem|"
    r"tablo açıklaması|şekil açıklaması|teknik ek|veri indirme|lisans|"
    r"tanım|terim|tarihçe|adres|logo|sponsor|oturum"
    r")",
    re.IGNORECASE,
)


def normalized_text(text: str) -> str:
    return re.sub(r"\s+", " ", str(text).strip().casefold())


def semantic_label_error(label: str, text: str) -> str | None:
    normalized = normalized_text(text)
    claim_like = bool(CLAIM_RE.search(normalized))
    evidence_like = bool(EVIDENCE_RE.search(normalized) or re.search(r"\b\d+(?:[,.]\d+)?\b", normalized))
    other_like = bool(OTHER_RE.search(normalized))

    if label == "claim":
        if evidence_like and not claim_like:
            return "claim looks like evidence"
        if other_like and not claim_like:
            return "claim looks like other"
        if not claim_like:
            return "claim lacks stance/evaluation signal"
    if label == "evidence":
        if other_like and not evidence_like:
            return "evidence looks like other"
        if not evidence_like:
            return "evidence lacks data/reason/source signal"
    if label == "other":
        if claim_like:
            return "other looks like claim"
        if evidence_like:
            return "other looks like evidence"
        if not other_like:
            return "other lacks non-argument context signal"
    return None


def semantically_valid_component(label: str, text: str) -> bool:
    return semantic_label_error(label, text) is None


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def dedupe(items: Iterable[dict], key_fields: tuple[str, ...]) -> list[dict]:
    seen: set[tuple[str, ...]] = set()
    out: list[dict] = []
    for item in items:
        key = tuple(str(item.get(field, "")).strip() for field in key_fields)
        if not all(key) or key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def split_for_topic(topic: str) -> str:
    if topic.startswith("locked_"):
        return "locked_holdout"
    digest = hashlib.sha1(topic.encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    if bucket < 10:
        return "test"
    if bucket < 20:
        return "validation"
    return "train"


def seed_components() -> list[dict]:
    examples: list[dict] = []
    final_seti_path = ROOT.parent.parent / "final_veri_seti.json"
    if final_seti_path.exists():
        dataset = json.loads(final_seti_path.read_text(encoding="utf-8"))
        for item in dataset:
            data = item.get("data", {})
            topic = data.get("topic", "")
            for c in data.get("components", []):
                label = normalize_component(c.get("label", ""))
                if label in COMPONENT_LABELS and semantically_valid_component(label, c.get("text", "")):
                    examples.append({
                        "text": c["text"],
                        "label": label,
                        "topic": topic,
                        "source": "final_veri_seti",
                        "split": split_for_topic(topic)
                    })
    if QUALITY_REGRESSION_PATH.exists():
        quality = json.loads(QUALITY_REGRESSION_PATH.read_text(encoding="utf-8"))
        for item in quality.get("component_examples", []):
            label = normalize_component(item.get("label", ""))
            if label in COMPONENT_LABELS:
                examples.append({**item, "label": label, "source": "quality_regression", "split": split_for_topic(item.get("topic", ""))})
    if V3_HARD_CASE_PATH.exists():
        hard = json.loads(V3_HARD_CASE_PATH.read_text(encoding="utf-8"))
        for item in hard.get("component_examples", []):
            label = normalize_component(item.get("label", ""))
            if label in COMPONENT_LABELS:
                examples.append({**item, "label": label, "source": "locked_hard_case", "split": "locked_holdout"})
    return examples


def seed_relations() -> list[dict]:
    pairs: list[dict] = []
    final_seti_path = ROOT.parent.parent / "final_veri_seti.json"
    if final_seti_path.exists():
        dataset = json.loads(final_seti_path.read_text(encoding="utf-8"))
        for item in dataset:
            data = item.get("data", {})
            topic = data.get("topic", "")
            components = data.get("components", [])
            relations = data.get("relations", [])
            comp_dict = {c["id"]: c for c in components}
            for r in relations:
                source = comp_dict.get(r["from"])
                target = comp_dict.get(r["to"])
                if source and target:
                    label = normalize_relation(r.get("label", ""))
                    source_label = normalize_component(source.get("label", ""))
                    target_label = normalize_component(target.get("label", ""))
                    if (
                        label in RELATION_LABELS
                        and semantically_valid_component(source_label, source.get("text", ""))
                        and semantically_valid_component(target_label, target.get("text", ""))
                        and (label != "none" or source_label == "other")
                        and (label not in {"support", "attack"} or target_label == "claim")
                    ):
                        pairs.append({
                            "claim_text": target["text"],
                            "evidence_text": source["text"],
                            "label": label,
                            "topic": topic,
                            "source": "final_veri_seti",
                            "split": split_for_topic(topic)
                        })
    if QUALITY_REGRESSION_PATH.exists():
        quality = json.loads(QUALITY_REGRESSION_PATH.read_text(encoding="utf-8"))
        for item in quality.get("relation_examples", []):
            label = normalize_relation(item.get("label", ""))
            if label in RELATION_LABELS:
                pairs.append({**item, "label": label, "source": "quality_regression", "split": split_for_topic(item.get("topic", ""))})
    if V3_HARD_CASE_PATH.exists():
        hard = json.loads(V3_HARD_CASE_PATH.read_text(encoding="utf-8"))
        for item in hard.get("relation_examples", []):
            label = normalize_relation(item.get("label", ""))
            if label in RELATION_LABELS:
                pairs.append({**item, "label": label, "source": "locked_hard_case", "split": "locked_holdout"})
    for item in LOCKED_HARD_CASES:
        pairs.append({**item, "source": "locked_hard_case", "split": "locked_holdout"})
    return pairs


def synthetic_components() -> list[dict]:
    examples: list[dict] = []
    for idx, (slug, subject, pro, con) in enumerate(TOPICS):
        subject_lower = subject.lower()
        for form_idx, form in enumerate(CLAIM_FORMS):
            examples.append(
                {
                    "text": form.format(subject=subject, subject_lower=subject_lower, pro_outcome=pro, con_outcome=con),
                    "label": "claim",
                    "topic": f"v4_{slug}_claim_{form_idx}",
                    "source": "synthetic_tr_v4",
                }
            )
        for form_idx, form in enumerate(EVIDENCE_FORMS):
            source = SOURCES[(idx + form_idx) % len(SOURCES)]
            examples.append(
                {
                    "text": form.format(source=source, subject=subject, subject_lower=subject_lower, pro_outcome=pro, con_outcome=con),
                    "label": "evidence",
                    "topic": f"v4_{slug}_evidence_{form_idx}",
                    "source": "synthetic_tr_v4",
                }
            )
        for form_idx, form in enumerate(OTHER_FORMS):
            examples.append(
                {
                    "text": form,
                    "label": "other",
                    "topic": f"v4_{slug}_other_{form_idx}",
                    "source": "synthetic_tr_v4",
                }
            )
        for hard_label, forms in HARD_COMPONENT_FORMS:
            for form_idx, form in enumerate(forms):
                examples.append(
                    {
                        "text": form.format(subject=subject, subject_lower=subject_lower, pro_outcome=pro, con_outcome=con),
                        "label": hard_label,
                        "topic": f"v4_{slug}_hard_component_{hard_label}_{form_idx}",
                        "source": "synthetic_tr_v4_hard_component",
                    }
                )
    return examples


def synthetic_relations() -> list[dict]:
    pairs: list[dict] = []
    for idx, (slug, subject, pro, con) in enumerate(TOPICS):
        subject_lower = subject.lower()
        claim = f"{subject} {pro}."
        source = SOURCES[idx % len(SOURCES)]
        support_evidence = f"{source} raporu, {subject_lower} için {pro} yönünde ölçülebilir sonuçlar bildirdi."
        attack_evidence = f"{source} raporu, {subject_lower} uygulamasında {con} sonucunun ortaya çıktığını bildirdi."
        none_evidence = NONE_EVIDENCE_FORMS[idx % len(NONE_EVIDENCE_FORMS)]
        for form_idx, form in enumerate(SUPPORT_FORMS):
            pairs.append(
                {
                    "claim_text": claim,
                    "evidence_text": form.format(support_evidence=support_evidence),
                    "label": "support",
                    "topic": f"v4_{slug}_support_{form_idx}",
                    "source": "synthetic_tr_v4",
                }
            )
        for form_idx, form in enumerate(ATTACK_FORMS):
            pairs.append(
                {
                    "claim_text": claim,
                    "evidence_text": form.format(attack_evidence=attack_evidence),
                    "label": "attack",
                    "topic": f"v4_{slug}_attack_{form_idx}",
                    "source": "synthetic_tr_v4",
                }
            )
        for form_idx, form in enumerate(NONE_FORMS):
            pairs.append(
                {
                    "claim_text": claim,
                    "evidence_text": form.format(none_evidence=none_evidence),
                    "label": "none",
                    "topic": f"v4_{slug}_none_{form_idx}",
                    "source": "synthetic_tr_v4",
                }
            )
    pairs.extend(CONDITION_MISMATCH_RELATIONS)
    pairs.extend(HARD_RELATION_CASES)
    return pairs


def external_legal_nli(limit_per_label: int) -> tuple[list[dict], dict]:
    """Adapt Apache-2.0 Turkish NLI rows to support/attack/none relation pairs."""
    audit = {
        "dataset": "Turkish-NLI/legal_nli_TR_V1",
        "license": "apache-2.0",
        "url": "https://huggingface.co/datasets/Turkish-NLI/legal_nli_TR_V1",
        "used": 0,
        "status": "not_loaded",
    }
    try:
        from datasets import load_dataset
    except Exception as exc:  # pragma: no cover - environment dependent
        audit["status"] = f"datasets_import_failed: {exc}"
        return [], audit

    label_map = {"entailment": "support", "contradiction": "attack", "neutral": "none"}
    counts: Counter[str] = Counter()
    out: list[dict] = []
    try:
        stream = load_dataset("Turkish-NLI/legal_nli_TR_V1", split="train", streaming=True)
        for row in stream:
            mapped = label_map.get(str(row.get("label", "")).lower())
            if not mapped or counts[mapped] >= limit_per_label:
                continue
            premise = str(row.get("premise", "")).strip()
            hypothesis = str(row.get("hypothesis", "")).strip()
            if len(premise) < 24 or len(hypothesis) < 24:
                continue
            out.append(
                {
                    "claim_text": hypothesis,
                    "evidence_text": premise,
                    "label": mapped,
                    "topic": f"external_legal_nli_{mapped}_{counts[mapped]}",
                    "source": "external_legal_nli_apache2",
                }
            )
            counts[mapped] += 1
            if all(counts[label] >= limit_per_label for label in label_map.values()):
                break
        audit["used"] = len(out)
        audit["label_counts"] = dict(counts)
        audit["status"] = "loaded"
    except Exception as exc:  # pragma: no cover - network dependent
        audit["status"] = f"load_failed: {exc}"
    return out, audit


def repeat_to_target(items: list[dict], label_field: str, target_counts: dict[str, int]) -> list[dict]:
    by_label: dict[str, list[dict]] = {label: [] for label in target_counts}
    for item in items:
        label = item[label_field]
        if label in by_label:
            by_label[label].append(item)
    out: list[dict] = []
    for label, target in target_counts.items():
        pool = by_label[label]
        if not pool:
            raise RuntimeError(f"No examples for label {label}")
        priority = [
            item
            for item in pool
            if item.get("source") in HARD_RELATION_PRIORITY_SOURCES
        ]
        regular = [
            item
            for item in pool
            if item.get("source") not in HARD_RELATION_PRIORITY_SOURCES
        ]
        ordered_pool = priority + regular
        for idx in range(target):
            base = dict(ordered_pool[idx % len(ordered_pool)])
            base["augmentation_id"] = idx // len(pool)
            base.setdefault("split", split_for_topic(base.get("topic", "")))
            if base["split"] == "locked_holdout":
                base["split"] = "train"
            out.append(base)
    random.shuffle(out)
    return out


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def audit_rows(rows: list[dict], label_field: str) -> dict:
    return {
        "total": len(rows),
        "labels": dict(Counter(row[label_field] for row in rows)),
        "splits": dict(Counter(row.get("split", "train") for row in rows)),
        "sources": dict(Counter(row.get("source", "unknown") for row in rows)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--component-target", type=int, default=COMPONENT_TARGET)
    parser.add_argument("--relation-target", type=int, default=RELATION_TARGET)
    parser.add_argument("--external-per-label", type=int, default=600)
    parser.add_argument("--skip-external", action="store_true")
    args = parser.parse_args()

    random.seed(SEED)
    component_counts = dict(COMPONENT_TARGET_COUNTS)
    relation_counts = dict(RELATION_TARGET_COUNTS)
    if args.component_target != COMPONENT_TARGET:
        component_counts = {
            "claim": int(args.component_target * 0.35),
            "evidence": int(args.component_target * 0.35),
            "other": args.component_target - int(args.component_target * 0.70),
        }
    if args.relation_target != RELATION_TARGET:
        relation_counts = {
            "attack": int(args.relation_target * 0.34),
            "support": int(args.relation_target * 0.33),
            "none": args.relation_target - int(args.relation_target * 0.67),
        }

    external_pairs: list[dict] = []
    external_audit = {"status": "skipped"}
    if not args.skip_external:
        external_pairs, external_audit = external_legal_nli(args.external_per_label)

    locked_component_keys = {(row["text"].strip(), row["label"]) for row in LOCKED_COMPONENT_HARD_CASES}
    locked_relation_keys = {
        (row["claim_text"].strip(), row["evidence_text"].strip(), row["label"])
        for row in LOCKED_HARD_CASES
    }
    component_pool = [
        row
        for row in seed_components() + synthetic_components()
        if (row.get("text", "").strip(), row.get("label")) not in locked_component_keys
    ]
    relation_pool = [
        row
        for row in seed_relations() + synthetic_relations() + external_pairs
        if (row.get("claim_text", "").strip(), row.get("evidence_text", "").strip(), row.get("label")) not in locked_relation_keys
    ]
    components_seed = dedupe(component_pool, ("text", "label"))
    relations_seed = dedupe(relation_pool, ("claim_text", "evidence_text", "label"))
    components = repeat_to_target(components_seed, "label", component_counts)
    relations = repeat_to_target(relations_seed, "label", relation_counts)
    locked_components = [
        {**row, "source": "locked_hard_case", "split": "locked_holdout"}
        for row in LOCKED_COMPONENT_HARD_CASES
    ]
    locked_relations = [
        {**row, "source": "locked_hard_case", "split": "locked_holdout"}
        for row in LOCKED_HARD_CASES
    ]

    write_jsonl(V4_DIR / "component_examples.jsonl", components)
    write_jsonl(V4_DIR / "relation_pairs.jsonl", relations)
    write_jsonl(V4_DIR / "locked_holdout_components.jsonl", locked_components)
    write_jsonl(V4_DIR / "locked_holdout_relations.jsonl", locked_relations)
    audit = {
        "version": "v4",
        "seed": SEED,
        "components": audit_rows(components, "label"),
        "relations": audit_rows(relations, "label"),
        "locked_holdout_components": audit_rows(locked_components, "label"),
        "locked_holdout_relations": audit_rows(locked_relations, "label"),
        "external_sources": [external_audit],
        "license_policy": "Only apache-2.0 external rows are adapted into train data; failed or unknown-license sources are excluded.",
        "bad_component_labels": sorted({row["label"] for row in components if row["label"] not in COMPONENT_LABELS}),
        "bad_relation_labels": sorted({row["label"] for row in relations if row["label"] not in RELATION_LABELS}),
    }
    (V4_DIR / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    if len(components) < args.component_target or len(relations) < args.relation_target:
        raise SystemExit(1)
    if audit["bad_component_labels"] or audit["bad_relation_labels"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
