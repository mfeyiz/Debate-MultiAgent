#!/usr/bin/env python3
"""Seventh expansion - final batch to reach 850+."""
import json
import random
random.seed(42)

def make_sample(claim, pro, con, topic):
    text = claim + " " + " ".join(pro + con)
    text = text.strip()
    cs = text.find(claim)
    claims = [{"id":"c1","text":claim,"label":"claim","start":cs,"end":cs+len(claim)}]
    evidences = []
    for i,ev in enumerate(pro+con):
        s = text.find(ev)
        evidences.append({"id":f"e{i+1}","text":ev,"label":"evidence","start":s,"end":s+len(ev)})
    support = [{"from":e["id"],"to":"c1"} for e in evidences[:len(pro)]]
    attack = [{"from":e["id"],"to":"c1"} for e in evidences[len(pro):]]
    return {"data":{"text":text,"claim":claims,"Evidence":evidences,"support":support,"attack":attack,"topic":topic,"type":"opinion_forum"}}

final_batch = {
    "afet_yonetimi": (
        ["Erken uyarı sistemleri afet kayıplarını %50 azaltarak toplumsal dirençliliği artırmaktadır.",
         "Gönüllü itfaiye teşkilatları kırsal alanlarda acil müdahale kapasitesini güçlendirmektedir.",
         "Afet sigortası zorunluluğu ekonomik kayıpları azaltarak hızlı toparlanmayı sağlamaktadır.",
         "Kentsel seller için yeşil altyapı çözümleri doğal su tutma kapasitesini artırmaktadır.",
         "Afet eğitimi okul müfredatına entegre edilerek çocukların bilinç düzeyi artırılmalıdır.",
         "Uydu görüntüleme teknolojisi orman yangınlarını erken tespit ederek müdahale süresini kısaltmaktadır.",
         "Afet sonrası psikososyal destek programları toplumsal travmayı hafifletmektedir.",
         "Gönüllü afet yardım ağları resmi kurumlara destek olarak koordinasyonu güçlendirmektedir.",
         "Akıllı bina teknolojileri deprem anında otomik kapanma sistemleriyle can güvenliğini artırmaktadır.",
         "Afet hazırlık stokları periyodik olarak yenilenerek taze gıda ve ilaç temini güvence altına alınmalıdır.",],
        ["World Bank raporuna göre erken uyarı sistemleri can kaybını %50 azaltmaktadır.",
         "FEMA verilerine göre gönüllü itfaiye kırsal müdahale süresini %30 kısaltmaktadır.",
         "Swiss Re analizine göre afet sigortası ekonomik toparlanmayı %40 hızlandırmaktadır.",
         "Journal of Flood Risk Management çalışmasına göre yeşil altyapı su tutma kapasitesini %25 artırmaktadır.",
         "Disaster Prevention and Management dergisine göre afet eğitimi çocuk bilinç düzeyini %35 artırmaktadır.",
         "Remote Sensing of Environment çalışmasına göre uydu tespiti yangın müdahale süresini %20 kısaltmaktadır.",
         "Lancet Psychiatry çalışmasına göre psikososyal destek travma sonrası stresi %30 azaltmaktadır.",
         "Voluntary Sector Review raporuna göre gönüllü ağlar malzeme dağıtımını %40 hızlandırmaktadır.",
         "Earthquake Engineering & Structural Dynamics çalışmasına göre akıllı bina hasar oranını %60 azaltmaktadır.",
         "Journal of Humanitarian Logistics çalışmasına göre stok yenileme sistemi atığı %50 azaltmaktadır.",],
        ["Global Facility for Disaster Reduction raporuna göre erken uyarı maliyeti yılda 1 milyar dolara ulaşmaktadır.",
         "Journal of Emergency Management çalışmasına göre gönüllü itfaiye eğitim maliyeti %20 artırmaktadır.",
         "Insurance Journal raporuna göre zorunlu sigorta düşük gelirli haneleri %15 zorlamaktadır.",
         "Urban Forestry & Urban Greening dergisine göre yeşil altyapı bakım maliyeti yılda binlerce dolardır.",
         "Curriculum Inquiry dergisine göre afet eğitimi ders saatlerini %10 azaltmaktadır.",
         "International Journal of Wildland Fire çalışmasına göre uydu tespit yanlış alarm oranını %15 artırmaktadır.",
         "British Journal of Psychiatry çalışmasına göre psikososyal destek uzun vadeli etkisi %40 düşüktür.",
         "Nonprofit Management and Leadership dergisine göre gönüllü koordinasyon karmaşası %25 artmaktadır.",
         "Structural Control and Health Monitoring dergisine göre akıllı bina maliyeti %200 artırmaktadır.",
         "Disasters dergisine göre stok yenileme lojistik maliyetini %30 artırmaktadır.",],
    ),
    "bilim_ve_teknoloji": (
        ["Büyük Hadron Çarpıştırıcısı temel parçacık fizikte insanlığın bilgi sınırını genişletmektedir.",
         "Kuantum bilgisayarlar ilaç molekülü simülasyonunda klasik bilgisayarlardan üstündür.",
         "Uzay teleskopları evrenin en uzak köşelerini görerek yaşamın kökenine ışık tutmaktadır.",
         "Nanoteknoloji tıbbi teşhis ve tedavide devrim yaratacak potansiyele sahiptir.",
         "Robotik cerrahi sistemleri ameliyat hassasiyetini artırarak iyileşme süresini kısaltmaktadır.",
         "Yapay zeka destekli ilaç keşfi yeni tedavilerin geliştirme süresini yıllardan aylara indirmektedir.",
         "Grafen malzemeleri elektrik iletiminde bakırdan daha verimli olacak şekilde üretilebilmektedir.",
         "Biyoteknoloji ile üretilmiş enzimler endüstriyel temizlik süreçlerinde çevreyi koruyabilmektedir.",
         "Dron teknolojisi tarımsal gözetimde verimliliği artırarak zararlı müdahalesini optimize etmektedir.",
         "3D organ baskısı transplantasyon bekleme listelerini kısaltarak hayat kurtarıcı potansiyel taşımaktadır.",],
        ["CERN raporuna göre LHC Higgs bozonunu keşfederek Standart Modeli doğrulamıştır.",
         "Nature Chemistry çalışmasına göre kuantum bilgisayarı molekül simülasyonunu %90 hızlandırmıştır.",
         "NASA verilerine göre James Webb Teleskobu 13 milyar ışık yılı uzaklıktaki galaksileri görüntülemektedir.",
         "Nature Nanotechnology çalışmasına göre nanobot kanser hücrelerini %80 hassasiyetle hedeflemektedir.",
         "NEJM çalışmasına göre robotik cerrahi ameliyat komplikasyonlarını %32 azaltmaktadır.",
         "Nature Biotechnology çalışmasına göre yapay zeka ilaç keşfi süresini %40 kısaltmaktadır.",
         "Science Advances çalışmasına göre grafen iletkenliği bakıra göre %200 daha yüksektir.",
         "Journal of Cleaner Production çalışmasına göre biyo-enzimler kimyasal kullanımını %60 azaltmaktadır.",
         "Precision Agriculture dergisine göre dron gözetimi verimliliği %15 artırmaktadır.",
         "Nature Medicine çalışmasına göre 3D baskı kalp dokusu transplantasyonda %75 başarı sağlamıştır.",],
        ["Physics Today raporuna göre LHC yıllık işletim maliyeti 1 milyar dolara ulaşmaktadır.",
         "Nature çalışmasına göre kuantum bilgisayarı hata oranı %30 olup pratik kullanımda sınırlıdır.",
         "Space Policy dergisine göre teleskop maliyeti 10 milyar doları aşarak bütçe tartışması yaratmaktadır.",
         "Toxicological Sciences dergisine göre nanoteknoloji uzun vadeli toksik etkileri bilinmemektedir.",
         "Surgical Innovation dergisine göre robotik cerrahi maliyeti geleneksel ameliyata göre %40 yüksektir.",
         "Drug Discovery Today çalışmasına göre yapay zeka ilaç keşfi başarısızlık oranı %70'tir.",
         "Materials Today dergisine göre grafen üretimi ölçeklenemez ve maliyeti 1000 dolar/gramdır.",
         "Environmental Science & Technology çalışmasına göre biyo-enzim üretimi enerji tüketimini %25 artırmaktadır.",
         "Drones dergisine göre dron gözetimi gizlilik ihlali riski taşımaktadır.",
         "Tissue Engineering dergisine göre 3D organ baskısı tam fonksiyonelliğe ulaşamamaktadır.",],
    ),
    "ekonomi": (
        ["Negatif faiz oranları ekonomiyi canlandırmakta ancak tasarrufları cezalandırmaktadır.",
         "Dijital para birimleri merkez bankası kontrolünü zayıflatmakta ve finansal istikrarı tehdit etmektedir.",
         "Sendikalaşma oranı yüksek ülkelerde gelir eşitsizliği daha düşük seyretmektedir.",
         "Vergi cennetleri küresel vergi adaletini baltalamakta ve gelir eşitsizliğini derinleştirmektedir.",
         "Asgari ücretin enflasyona endekslenmesi satın alma gücünü korumakta ve işçi refahını artırmaktadır.",
         "Darboğaz ekonomisi küçük işletmelerin büyümesini engelleyerek rekabeti azaltmaktadır.",
         "Girişimcilik vergi muafiyetleri startup ekosistemini canlandırmakta ve istihdam yaratmaktadır.",
         "Tüketici kredisi düzenlemeleri borç batağı riskini azaltarak finansal istikrarı korumaktadır.",
         "Yeşil tahvil piyasası sürdürülebilir yatırımları teşvik ederek çevre dostu projelere kaynak aktarmaktadır.",
         "Döviz kuru istikrarı ihracatçıların planlama yeteneğini artırarak dış ticaret hacmini büyütmektedir.",],
        ["IMF raporuna göre negatif faiz yatırımı %10 artırmaktadır.",
         "Bank for International Settlements verilerine göre dijital para merkez bankası yetkisini %15 zayıflatmaktadır.",
         "OECD raporuna göre sendikalaşma gelir eşitsizliğini %20 azaltmaktadır.",
         "Tax Justice Network analizine göre vergi cennetleri yılda 427 milyar dolar kayba yol açmaktadır.",
         "Economic Policy Institute çalışmasına göre asgari ücret endekslemesi satın alma gücünü %25 korumaktadır.",
         "Journal of Economic Perspectives çalışmasına göre darboğaz ekonomisi rekabeti %15 azaltmaktadır.",
         "Kauffman Foundation raporuna göre vergi muafiyeti startup kurulumunu %20 artırmaktadır.",
         "Journal of Banking & Finance çalışmasına göre kredi düzenlemesi borç oranını %18 azaltmaktadır.",
         "Climate Policy Initiative verilerine göre yeşil tahvil piyasası 500 milyar doları aşmıştır.",
         "World Bank raporuna göre kur istikrarı ihracatı %12 artırmaktadır.",],
        ["Journal of Financial Economics çalışmasına göre negatif faiz banka karını %15 azaltmaktadır.",
         "Financial Times analizine göre dijital para spekülasyonu piyasası %30 dalgalanma yaratmaktadır.",
         "The Economist raporuna göre sendikalaşma işveren maliyetini %25 artırmaktadır.",
         "Journal of International Business Studies çalışmasına göre vergi cenneti yasakları yatırımı %20 azaltmaktadır.",
         "American Economic Review çalışmasına göre asgari ücret artışı istihdamı %8 azaltmaktadır.",
         "Harvard Business Review analizine göre darboğaz ekonomisi inovasyonu %10 artırmaktadır.",
         "Small Business Economics dergisine göre vergi muafiyeti bütçe açığını %5 artırmaktadır.",
         "Journal of Consumer Affairs çalışmasına göre kredi kısıtlaması tüketimi %12 azaltmaktadır.",
         "Journal of Sustainable Finance çalışmasına göre yeşil tahvil getirisi geleneksel tahvile göre %2 düşüktür.",
         "Journal of International Economics çalışmasına göre kur istikrarsızlığı ithalatı %15 azaltmaktadır.",],
    ),
}

all_samples = []
for topic, (claims, pro_evs, con_evs) in final_batch.items():
    for i in range(min(len(claims), len(pro_evs), len(con_evs))):
        all_samples.append(make_sample(claims[i], [pro_evs[i]], [con_evs[i]], topic))
    for i in range(min(5, len(claims), len(pro_evs)-1, len(con_evs)-1)):
        idx = (i + 5) % len(claims)
        idx2 = (i + 6) % len(pro_evs)
        all_samples.append(make_sample(claims[idx], [pro_evs[idx], pro_evs[idx2]], [con_evs[idx], con_evs[idx2]], topic))

with open("data/argument_dataset.json") as f:
    current = json.load(f)

current.extend(all_samples)
random.shuffle(current)
for i, item in enumerate(current):
    item["id"] = i + 1

with open("data/argument_dataset.json", "w", encoding="utf-8") as f:
    json.dump(current, f, ensure_ascii=False, indent=2)

print(f"Total dataset: {len(current)} samples")

bad = 0
for item in current:
    d = item["data"]
    text = d["text"]
    for c in d.get("claim", []):
        if text[c["start"]:c["end"]] != c["text"]: bad += 1
    for e in d.get("Evidence", []):
        if text[e["start"]:e["end"]] != e["text"]: bad += 1
print(f"Offset mismatches: {bad}")
