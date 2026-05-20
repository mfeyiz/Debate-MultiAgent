#!/usr/bin/env python3
"""Fifth expansion - diverse short and long form arguments."""
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

final_data = {
    "enerji_verimliligi": (
        ["Bina yalıtımına yapılan yatırım ısıtma maliyetlerini %40 azaltarak hızlı geri dönüş sağlamaktadır.",
         "LED aydınlatma teknolojisi geleneksel ampullere göre %80 daha az enerji tüketmektedir.",
         "Akıllı termostatlar ev enerji verimliliğini %15 artırarak konforu koruyabilmektedir.",
         "Güneş panelleri evsel elektrik ihtiyacının %100'ünü karşılayarak faturaları sıfırlayabilmektedir.",
         "Enerji etiketleme sistemi tüketicileri bilinçli seçim yapmaya yönlendirmektedir.",
         "Isı pompası teknolojisi elektrikli ısıtmanın verimliliğini 4 kat artırmaktadır.",
         "Enerji verimli cihazlar pahalı olmakla birlikte ömür boyu tasarruf sağlamaktadır.",
         "Yeşil bina sertifikasyonu emlak değerini %10 artırarak yatırım getirisi sunmaktadır.",
         "Sanayi enerji denetimleri atık ısıyı geri kazanarak üretim maliyetlerini düşürmektedir.",
         "Kamusal binalarda enerji verimliliği standartları vergi mükelleflerine doğrudan tasarruf sağlamaktadır.",],
        ["International Energy Agency raporuna göre yalıtım yatırımları ısıtma maliyetlerini %40 azaltmaktadır.",
         "DOE verilerine göre LED ampuller 25 bin saat ömürle geleneksel ampullerden %80 daha az enerji harcamaktadır.",
         "Nest Labs araştırmasına göre akıllı termostat yıllık enerji faturalarını %15 düşürmektedir.",
         "National Renewable Energy Laboratory verilerine göre güneş paneli maliyeti son 10 yılda %90 düşmüştür.",
         "European Commission raporuna göre enerji etiketleme sistemi tüketici bilincini %30 artırmaktadır.",
         "Heat Pump Centre verilerine göre ısı pompası COP değeri 4'e ulaşarak verimliliği 4 kat artırmaktadır.",
         "Consumer Reports analizine göre enerji verimli cihazlar 5 yılda kendini amorti etmektedir.",
         "Journal of Sustainable Real Estate çalışmasına göre LEED sertifikası emlak değerini %10 artırmaktadır.",
         "Department of Energy raporuna göre sanayi enerji denetimleri atık ısı geri kazanımını %30 sağlamaktadır.",
         "National Association of State Energy Officials verilerine göre kamusal bina verimliliği yılda 2 milyar dolar tasarruf sağlamaktadır.",],
        ["Building and Environment dergisine göre yalıtım maliyeti bin metrekare başına 10 bin dolara ulaşmaktadır.",
         "Environmental Science & Technology çalışmasına göre LED üretimi nadir toprak elementi tüketmektedir.",
         "Privacy Journal raporuna göre akıllı termostat verileri üçüncü taraflarla paylaşılmaktadır.",
         "Energy Policy dergisine göre güneş paneli depolama maliyeti üretim maliyetinin %40'ına ulaşmaktadır.",
         "Journal of Consumer Affairs çalışmasına göre enerji etiketleri %20 oranında yanıltıcı bilgi içermektedir.",
         "Renewable Energy dergisine göre ısı pompası soğuk iklimlerde verimliliği %30 düşürmektedir.",
         "The Guardian haberine göre enerji verimli cihazlar onarım maliyeti geleneksel cihazlara göre %50 yüksektir.",
         "Journal of Real Estate Finance çalışmasına göre yeşil sertifika alım maliyeti yatırım getirisini %5 azaltmaktadır.",
         "Industrial Ecology dergisine göre enerji denetimleri üretim süreçlerini %10 yavaşlatmaktadır.",
         "Public Administration Review raporuna göre kamusal enerji projeleri bütçe aşım oranı %25'tir.",],
    ),
    "gumruk_ve_ticaret": (
        ["Serbest ticaret anlaşmaları tüketicilere daha düşük fiyat ve daha fazla seçenek sunmaktadır.",
         "İthalat vergileri yerel üretimi korur ancak tüketicilere yüksek maliyet getirmektedir.",
         "Dijital gümrük sistemleri ticaret süreçlerini hızlandırarak yolsuzluğu azaltmaktadır.",
         "Yerel üretim teşvikleri istihdamı artırırken ithalat bağımlılığını azaltmaktadır.",
         "E-ticaret düzenlemeleri tüketici haklarını korurken küçük işletmelerin büyümesine olanak tanımaktadır.",
         "Tedarik zinciri şeffaflığı ürün güvenliğini artırarak güveni tesis etmektedir.",
         "Ticari arabuluculuk anlaşmazlıkları hızlı çözerek mahkeme yükünü azaltmaktadır.",
         "Yeşil gümrük politikaları sürdürülebilir ticareti teşvik ederek çevre korumasını güçlendirmektedir.",
         "Küçük ölçekli üreticiler ihracat destekleriyle küresel pazarlara erişebilmektedir.",
         "Blockchain tabanlı ticaret finansmanı işlem maliyetlerini %50 azaltmaktadır.",],
        ["World Trade Organization raporuna göre serbest ticaret tükenci fiyatlarını ortalama %15 düşürmektedir.",
         "Peterson Institute analizine göre ithalat vergisi kaldırılması yıllık 1,4 trilyon dolar kazanç sağlamaktadır.",
         "World Customs Organization verilerine göre dijital gümrük işlem süresini %60 kısaltmaktadır.",
         "OECD raporuna göre yerel üretim teşvikleri istihdamı %10 artırmaktadır.",
         "eBay analizine göre e-ticaret düzenlemeleri küçük işletme satışlarını %25 artırmaktadır.",
         "GS1 raporuna göre tedarik zinciri şeffaflığı ürün izlenebilirliğini %80 artırmaktadır.",
         "Harvard Negotiation Project çalışmasına göre arabuluculuk mahkeme maliyetlerini %70 azaltmaktadır.",
         "Journal of Environmental Economics çalışmasına göre yeşil gümrük politikaları emisyonları %20 azaltmaktadır.",
         "International Trade Centre verilerine göre ihracat destekleri KOBİ ihracatını %30 artırmaktadır.",
         "McKinsey raporuna göre blockchain ticaret finansmanı maliyetini %50 azaltmaktadır.",],
        ["Economic Policy Institute raporuna göre serbest ticaret yerli üretim istihdamını %15 azaltmaktadır.",
         "Journal of International Economics çalışmasına göre ithalat vergisi kaldırılması vergi gelirlerini %20 düşürmektedir.",
         "Transparency International raporuna göre dijital gümrük sistemleri siber saldırı riski taşımaktadır.",
         "The Economist analizine göre yerel üretim teşvikleri verimliliği %10 düşürmektedir.",
         "Journal of Small Business Management çalışmasına göre e-ticaret uyumu KOBİ maliyetini %40 artırmaktadır.",
         "Supply Chain Management dergisine göre tedarik şeffaflığı rekabet bilgisi sızdırmaktadır.",
         "Stanford Law Review raporuna göre arabuluculuk zorlayıcı uygulamalar adaletsizliğe yol açmaktadır.",
         "Energy Economics dergisine göre yeşil gümrük politikaları ticaret maliyetini %15 artırmaktadır.",
         "Journal of Development Economics çalışmasına göre ihracat destekleri yerel pazarı %20 ihmal etmektedir.",
         "Journal of Financial Economics dergisine göre blockchain sistemi enerji tüketimi yüksek olmaktadır.",],
    ),
    "devlet_yonetimi": (
        ["E-devlet hizmetleri bürokratik süreçleri azaltarak vatandaş memnuniyetini artırmaktadır.",
         "Açık bütçe uygulamaları kamu harcamalarının hesap verebilirliğini güçlendirmektedir.",
         "Katılımcı bütçeleme yerel toplulukların ihtiyaçlarına doğrudan yanıt vermektedir.",
         "Kamuda yapay zeka destekli karar sistemleri verimliliği artırarak insan hatalarını azaltmaktadır.",
         "Veri tabanlı politika oluşturma kanıta dayalı yönetim anlayışını yerleştirmektedir.",
         "Şeffaf ihale süreçleri yolsuzluk riskini azaltarak kamu kaynaklarını korumaktadır.",
         "Dijital kimlik kartı hizmet erişimini kolaylaştırarak vatandaş deneyimini iyileştirmektedir.",
         "Kamu çalışanlarına uzaktan çalışma imkanı verimliliği artırmakta ve memnuniyeti yükseltmektedir.",
         "Ombudsman kurumu vatandaş şikayetlerini bağımsız olarak çözerek hukuk devletini güçlendirmektedir.",
         "Belediye hizmetlerinde mobil uygulamalar vatandaş geri bildirimini anlık olarak almaktadır.",],
        ["United Nations e-Government Survey raporuna göre e-devlet endeksi yüksek ülkelerde memnuniyet %30 artmaktadır.",
         "International Budget Partnership verilerine göre açık bütçe yolsuzluk algısını %25 azaltmaktadır.",
         "World Bank raporuna göre katılımcı bütçeleme yoksul mahallelerin kaynak payını %15 artırmaktadır.",
         "Harvard Kennedy School çalışmasına göre yapay zeka kamu kararlarında hata oranını %40 azaltmaktadır.",
         "Brookings Enstitüsü analizine göre veri tabanlı politika hedef başarısını %20 artırmaktadır.",
         "Transparency International verilerine göre şeffaf ihale yolsuzluk riskini %35 azaltmaktadır.",
         "European Commission raporuna göre dijital kimlik hizmet erişim süresini %50 kısaltmaktadır.",
         "Gallup anketi sonuçlarına göre uzaktan çalışma memnuniyeti kamu sektöründe %25 artırmıştır.",
         "Journal of Public Administration çalışmasına göre ombudsman şikayet çözüm oranını %60 artırmaktadır.",
         "Urban Studies dergisine göre mobil belediye uygulamaları şikayet çözüm süresini %40 kısaltmaktadır.",],
        ["Digital Government Review raporuna göre e-devlet projeleri bütçenin %30'unu aşabilmektedir.",
         "Public Administration dergisine göre açık bütçe kamu kurumlarının rekabet gücünü %10 azaltmaktadır.",
         "Local Government Studies çalışmasına göre katılımcı bütçeleme süreç yavaşlığına yol açmaktadır.",
         "AI & Society dergisine göre yapay zeka karar sistemleri algoritmik önyargı riski taşımaktadır.",
         "Policy Studies Journal çalışmasına göre veri tabanlı politika veri toplama maliyetini %50 artırmaktadır.",
         "Journal of Public Procurement çalışmasına göre şeffaf ihale maliyeti geleneksel sürece göre %20 yüksektir.",
         "Information Polity dergisine göre dijital kimlik sistemleri veri ihlali riski taşımaktadır.",
         "Public Personnel Management çalışmasına göre uzaktan çalışma kamu hizmet kalitesini %15 düşürmektedir.",
         "Ombudsman Journal raporuna göre ombudsman yetki sınırları karmaşık davaları %40 çözememektedir.",
         "Government Information Quarterly çalışmasına göre mobil uygulamalar sadece %25 vatandaş tarafından kullanılmaktadır.",],
    ),
}

all_samples = []
for topic, (claims, pro_evs, con_evs) in final_data.items():
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
