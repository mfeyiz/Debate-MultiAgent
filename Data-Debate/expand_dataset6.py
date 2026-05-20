#!/usr/bin/env python3
"""Sixth expansion - final push to 800+ samples."""
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

last_data = {
    "kuresel_saglik": (
        ["Küresel aşı eşitsizliği salgın hastalıkların kontrolsüz yayılmasına yol açmaktadır.",
         "Dijital sağlık pasaportları uluslararası seyahati güvenli hale getirmektedir.",
         "Genetik veri paylaşımı küresel sağlık araştırmalarını hızlandırmaktadır.",
         "Sağlık turizmi gelişmekte olan ülkelerin ekonomik büyümesini desteklemektedir.",
         "Tele-sağlık hizmetleri kırsal alanlarda sağlık erişim eşitsizliğini ortadan kaldırmaktadır.",
         "Antibiyotik direnci küresel bir tehdit oluşturmakta ve uluslararası iş birliği gerektirmektedir.",
         "Küresel sağlık finansmanı düşük gelirli ülkelerin temel sağlık altyapısını güçlendirmektedir.",
         "Mobil sağlık klinikleri afet bölgelerinde acil müdahale kapasitesini artırmaktadır.",
         "Büyük veri analizi salgın erken uyarı sistemlerinin doğruluğunu %85 artırmaktadır.",
         "Küresel tıbbi cihaz standardizasyonu sağlık hizmeti kalitesini eşitlemektedir.",],
        ["WHO raporuna göre aşı eşitsizliği her yıl 1,5 milyon ölülebilir ölüme yol açmaktadır.",
         "IATA verilerine göre dijital sağlık pasaportları sınır geçiş süresini %40 kısaltmaktadır.",
         "Nature Genetics çalışmasına göre genetik veri paylaşımı araştırma hızını %30 artırmaktadır.",
         "World Medical Association raporuna göre sağlık turizmi gelişmekte olan ülkelere 100 milyar dolar getirmektedir.",
         "Journal of Telemedicine çalışmasına göre tele-sağlık kırsal erişimi %45 artırmaktadır.",
         "CDC verilerine göre antibiyotik direnci yılda 35 bin ölüme neden olmaktadır.",
         "Global Fund raporuna göre küresel sağlık finansmanı AIDS, verem ve sıtmada 50 milyon hayat kurtarmıştır.",
         "Lancet Global Health çalışmasına göre mobil klinikler afet müdahale süresini %30 kısaltmaktadır.",
         "Health Affairs dergisine göre büyük veri salgın tahmin doğruluğunu %85 artırmaktadır.",
         "WHO verilerine göre cihaz standardizasyonu sağlık hatası oranını %20 azaltmaktadır.",],
        ["Journal of Human Rights çalışmasına göre aşı zorunluluğu bireysel özgürlükleri %15 kısıtlamaktadır.",
         "Privacy International raporuna göre dijital sağlık pasaportları veri gizliliğini tehdit etmektedir.",
         "Nature Medicine dergisine göre genetik veri paylaşımı ayrımcılık riskini %25 artırmaktadır.",
         "Developing World Bioethics çalışmasına göre sağlık turizmi yerel hasta kapasitesini %20 azaltmaktadır.",
         "Journal of Rural Health çalışmasına göre tele-sağlık teşhis doğruluğu yüz yüze muayeneye göre %10 düşüktür.",
         "Microbiology Spectrum dergisine göre antibiyotik kısıtlamaları hayvancılık verimliliğini %12 düşürmektedir.",
         "Health Policy and Planning çalışmasına göre küresel fon bağımlılığı yerel sistemleri %30 zayıflatmaktadır.",
         "Disaster Medicine çalışmasına göre mobil klinik maliyeti sabit kliniklere göre %50 yüksektir.",
         "Big Data & Society dergisine göre veri analizi yanlış alarm oranını %20 artırmaktadır.",
         "Medical Device Regulation raporuna göre standardizasyon inovasyonu %15 yavaşlatmaktadır.",],
    ),
    "medya_ve_haberlesme": (
        ["Kamu yayıncılığı demokratik toplumlarda çeşitli görüşlerin duyulmasını garanti altına almaktadır.",
         "Yerel gazetelerin dijital dönüşümü topluluk haberleşmesini güçlendirmektedir.",
         "Radyo frekansları acil durum iletişiminde hayati rol oynamaktadır.",
         "Medya okuryazarlığı eğitimi dezenformasyona karşı direnci artırmaktadır.",
         "Bağımsız gazetecilik yolsuzlukları ortaya çıkararak hesap verebilirliği sağlamaktadır.",
         "Podcast platformları uzun form haberciliği destekleyerek derinlemesine analizi teşvik etmektedir.",
         "Veri gazeteciliği karmaşık konuları görselleştirerek kamuoyunu bilgilendirmektedir.",
         "Sosyal medya platformları vatandaş gazeteciliği aracılığıyla haber üretimini çeşitlendirmektedir.",
         "Medyanın çoğulculuğu toplumsal fikir çeşitliliğini koruyarak demokratik tartışmayı zenginleştirmektedir.",
         "Haber ajansları uluslararası haber akışını standartlaştırarak bilgi güvenilirliğini artırmaktadır.",],
        ["Reuters Institute raporuna göre kamu yayıncılığı haber güvenilirliğini %35 artırmaktadır.",
         "Pew Research Center verilerine göre yerel dijital haber aboneliği %20 artmıştır.",
         "FEMA raporuna göre acil durum radyo iletişimi hayat kurtarma oranını %25 artırmaktadır.",
         "Stanford History Education Group çalışmasına göre medya okuryazarlığı dezenformasyonu %30 azaltmaktadır.",
         "Center for Public Integrity verilerine göre bağımsız gazetecilik yolsuzluk vakalarının %40'ını açığa çıkarmaktadır.",
         "Edison Research raporuna göre podcast dinleyicileri haber konularında %60 daha bilgilidir.",
         "Data Journalism Handbook çalışmasına göre veri gazeteciliği okuyucu etkileşimini %45 artırmaktadır.",
         "Reuters Institute verilerine göre sosyal medya haber kaynağı olarak %68 kullanılmaktadır.",
         "Journalism Studies dergisine göre medya çoğulculuğu fikir çeşitliliğini %30 artırmaktadır.",
         "Reuters Agency raporuna göre standart haber formatı bilgi tutarlılığını %25 artırmaktadır.",],
        ["Media Culture & Society dergisine göre kamu yayıncılığı vergi yükünü %10 artırmaktadır.",
         "Journalism Practice çalışmasına göre yerel dijital dönüşüm bütçe açığını %30 oluşturmaktadır.",
         "Radio Science dergisine göre frekans kısıtlaması radyo erişimini %20 azaltmaktadır.",
         "Computers in Human Behavior çalışmasına göre medya okuryazarlığı etkisi 6 ayda %50 azalmaktadır.",
         "Columbia Journalism Review haberine göre bağımsız gazetecilik finansmanı zorlukla karşı karşıyadır.",
         "Nieman Lab raporuna göre podcast üretim maliyeti geleneksel habere göre %40 yüksektir.",
         "Digital Journalism dergisine göre veri gazeteciliği teknik yetersizlik nedeniyle %25 hatalıdır.",
         "Science Advances çalışmasına göre sosyal medya haberleri %30 daha fazla yanlış bilgi içermektedir.",
         "Political Communication dergisine göre medya çoğulculuğu kutuplaşmayı %15 artırmaktadır.",
         "Global Media Journal çalışmasına göre haber ajansı tekeli bilgi çeşitliliğini %20 azaltmaktadır.",],
    ),
    "kulturel_miras": (
        ["UNESCO Dünya Miras Listesi kültürel varlıkların korunmasında uluslararası iş birliğini sağlamaktadır.",
         "Dijital arşivleme teknolojileri tarihi belgelerin gelecek nesillere aktarımını garantilemektedir.",
         "Geleneksel el sanatları kültürel kimliğin sürekliliğini sağlayarak yerel ekonomiye katkı sunmaktadır.",
         "Tarihi kent merkezlerinin korunması turizmi teşvik ederek bölgesel kalkınmayı desteklemektedir.",
         "Müze eğitim programları öğrencilerde kültürel farkındalık ve tarihsel empati geliştirmektedir.",
         "Somut olmayan kültürel mirasın belgelenmesi nesilden nesile aktarımı güvence altına almaktadır.",
         "Arkeolojik kazılar geçmiş uygarlıkların sırlarını çözerek insanlık tarihine ışık tutmaktadır.",
         "Restorasyon uzmanlığı yapı kimliğini koruyarak tarihi eserlerin ömrünü uzatmaktadır.",
         "Kültürel miras eğitimi azınlık gruplarının tarihsel katkılarının tanınmasını sağlamaktadır.",
         "Tarihi peyzaj koruma alanları biyoçeşitlilik ve kültürel değerleri birlikte korumaktadır.",],
        ["UNESCO verilerine göre Dünya Miras Listesi 1000'den fazla alanı korumaya almıştır.",
         "Digital Preservation Coalition raporuna göre dijital arşivleme belge ömrünü %200 uzatmaktadır.",
         "Journal of Cultural Economics çalışmasına göre el sanatları yerel geliri %20 artırmaktadır.",
         "Tourism Management dergisine göre tarihi merkez koruma turizmi %30 artırmaktadır.",
         "Journal of Museum Education çalışmasına göre müze programları öğrenci farkındalığını %25 artırmaktadır.",
         "Intangible Cultural Heritage raporuna göre belgeleme projeleri 500'den fazla geleneği kaydetmiştir.",
         "Antiquity dergisine göre arkeolojik kazılar her yıl 100'den fazla yeni keşif yapmaktadır.",
         "Studies in Conservation dergisine göre uzman restorasyon yapı ömrünü %40 uzatmaktadır.",
         "Multicultural Education Review çalışmasına göre miras eğitimi empati skorlarını %30 artırmaktadır.",
         "Landscape Research dergisine göre tarihi peyzaj koruma biyoçeşitliliği %15 artırmaktadır.",],
        ["Journal of Cultural Heritage çalışmasına göre UNESCO listesi turizm baskısını %25 artırmaktadır.",
         "Archival Science dergisine göre dijital format değişikliği belgelerin %10'unu kaybettirmektedir.",
         "Handcraft Business Journal çalışmasına göre el sanatları üretimi makine üretimine göre %300 daha pahalıdır.",
         "Urban Studies dergisine göre tarihi merkez koruma konut maliyetini %20 artırmaktadır.",
         "Curator dergisine göre müze eğitim maliyeti öğrenci başına 50 dolara ulaşmaktadır.",
         "Cultural Anthropology dergisine göre belgeleme geleneksel aktarımı %15 zayıflatmaktadır.",
         "Archaeological Prospection çalışmasına göre kazı maliyeti proje başına 1 milyon doları aşmaktadır.",
         "Journal of Architectural Conservation dergisine göre restorasyon bütçe aşımı %30'dur.",
         "Social Studies Research çalışmasına göre miras eğitimi matematik saatlerini %10 azaltmaktadır.",
         "Land Use Policy dergisine göre tarihi peyzaj koruma tarım arazilerini %12 kısıtlamaktadır.",],
    ),
    "girisimcilik": (
        ["Girişimcilik eğitimi üniversite müfredatının zorunlu bir parçası olmalıdır.",
         "Melek yatırım ağları startup ekosistemine erken aşama finansman sağlayarak inovasyonu desteklemektedir.",
         "Kitle fonlaması platformları demokratik yatırım olanağı tanıyarak sermaye erişim eşitsizliğini azaltmaktadır.",
         "Startup merkezleri fiziksel altyapı ve mentorluk desteğiyle girişimci başarı oranını artırmaktadır.",
         "Serbest çalışan ekonomisi bireylere zaman ve mekan bağımsızlığı sunmaktadır.",
         "Girişimci vizeleri uluslararası yetenek rekabetinde ülkelerin çekiciliğini artırmaktadır.",
         "Hızlandırıcı programlar startup'ların pazara çıkış süresini 6 aydan 3 aya indirmektedir.",
         "Kadın girişimcilik destekleri cinsiyet eşitsizliğini azaltarak ekonomik kalkınmayı güçlendirmektedir.",
         "Sosyal girişimcilik kâr amacı gütmeyen modellerle toplumsal sorunlara sürdürülebilir çözümler üretmektedir.",
         "Patent koruma sistemi inovasyonu teşvik ederek Ar-Ge yatırımlarını artırmaktadır.",],
        ["Kauffman Foundation raporuna göre girişimcilik eğitimi mezunların startup kurma oranını %25 artırmaktadır.",
         "Crunchbase verilerine göre melek yatırım ağları yılda 25 milyar dolar erken aşama finansman sağlamaktadır.",
         "Journal of Business Venturing çalışmasına göre kitle fonlaması kadın girişimcilerin başarı oranını %30 artırmaktadır.",
         "Global Entrepreneurship Monitor raporuna göre startup merkezleri başarı oranını %40 artırmaktadır.",
         "Freelancers Union verilerine göre serbest çalışan sayısı 60 milyona ulaşmıştır.",
         "OECD raporuna göre girişimci vizesi uygulayan ülkeler startup sayısını %35 artırmıştır.",
         "Y Combinator verilerine göre hızlandırıcı programlar pazara çıkış süresini %50 kısaltmaktadır.",
         "World Bank raporuna göre kadın girişimcilik destekleri GSYH'ye %10 katkı sağlamaktadır.",
         "Stanford Social Innovation Review çalışmasına göre sosyal girişimcilik 100 milyon insana hizmet sunmaktadır.",
         "WIPO verilerine göre patent başvurusu Ar-Ge yatırımlarını %20 artırmaktadır.",],
        ["Harvard Business Review analizine göre zorunlu girişimcilik dersi akademik yükü %15 artırmaktadır.",
         "Venture Capital Journal raporuna göre melek yatırım başarı oranı sadece %10'dur.",
         "Small Business Economics dergisine göre kitle fonlaması projelerinin %60'ı hedefini tutturamamaktadır.",
         "Real Estate Economics çalışmasına göre startup merkezi kira maliyeti bütçeyi %25 aşmaktadır.",
         "Economic Policy Institute raporuna göre serbest çalışanlar sağlık sigortasından %70 yoksundur.",
         "Migration Policy Institute çalışmasına göre girişimci vizesi yerli istihdamını %5 azaltmaktadır.",
         "Forbes haberine göre hızlandırıcı programlar startup'ların %80'ini erken aşamada terk etmeye zorlamaktadır.",
         "Journal of Developmental Entrepreneurship çalışmasına göre cinsiyet destekleri erkek girişimcileri %15 dışlamaktadır.",
         "Nonprofit and Voluntary Sector Quarterly dergisine göre sosyal girişimcilik sürdürülebilirlik oranı %35'tir.",
         "Research Policy dergisine göre patent başvuru maliyeti küçük girişimciler için 10 bin doları aşmaktadır.",],
    ),
    "kadin_haklari": (
        ["Kadın kotaları yönetim kurullarında cinsiyet eşitliğini hızlandırmakta ve karar kalitesini artırmaktadır.",
         "Eşit ücret yasaları toplumsal cinsiyet adaletini sağlamakta ve ekonomik kalkınmayı desteklemektedir.",
         "Kreş ve anaokulu hizmetlerinin yaygınlaştırılması kadın iş gücüne katılımını artırmaktadır.",
         "Kadın sağlık araştırmalarına ayrılan fon artırılmalı çünkü tarihsel olarak yetersiz kalmıştır.",
         "Kadın liderlik programları iş dünyasında cinsiyet dengesini sağlamakta ve yenilikçiliği teşvik etmektedir.",
         "Kadın hakları savunucuları uluslararası arenada barış ve kalkınma hedeflerini güçlendirmektedir.",
         "Toplumsal cinsiyet eğitimi okul müfredatına entegre edilerek genç nesillerde farkındalık artırılmalıdır.",
         "Kadın girişimcilere yönelik mikro kredi programları ekonomik bağımsızlığı desteklemektedir.",
         "Şiddet mağduru kadınlara yönelik sığın evleri hizmetleri acilen genişletilmelidir.",
         "Kadınların siyasi temsil oranı demokratik meşruiyet ve politika kalitesini doğrudan etkilemektedir.",],
        ["McKinsey raporuna göre kadın kotaları yönetim kurulu çeşitliliğini %30 artırmaktadır.",
         "ILO verilerine göre eşit ücret uygulaması kadın gelirini %20 artırmaktadır.",
         "OECD raporuna göre kreş hizmetleri kadın istihdamını %15 artırmaktadır.",
         "Journal of Women's Health çalışmasına göre kadın sağlık araştırma fonu hastalık tespitini %25 hızlandırmaktadır.",
         "Catalyst raporuna göre kadın liderlik programları şirket performansını %10 artırmaktadır.",
         "UN Women verilerine göre kadın hakları savunuculuğu barış anlaşmaları dayanıklılığını %35 artırmaktadır.",
         "Gender and Education dergisine göre toplumsal cinsiyet eğitimi önyargıları %20 azaltmaktadır.",
         "World Bank raporuna göre mikro kredi programları kadın gelirini %40 artırmaktadır.",
         "UNDP verilerine göre sığın evleri şiddet tekrar oranını %50 azaltmaktadır.",
         "Inter-Parliamentary Union verilerine göre kadın temsil oranı yasama kalitesini %25 artırmaktadır.",],
        ["Journal of Management çalışmasına göre kota uygulamaları yetenek yerine cinsiyeti öne çıkarmaktadır.",
         "Economic Journal çalışmasına göre eşit ücret yasaları işveren maliyetini %15 artırmaktadır.",
         "Journal of Population Economics dergisine göre kreş hizmetleri vergi yükünü %10 artırmaktadır.",
         "Health Affairs dergisine göre kadın sağlık araştırma fonu erkek sağlık çalışmalarını %20 azaltmaktadır.",
         "Harvard Business Review analizine göre liderlik programları kota dışı adayları %10 dışlamaktadır.",
         "Foreign Policy dergisine göre kadın hakları savunuculuğu ulusal egemenlik tartışmaları yaratmaktadır.",
         "British Journal of Educational Psychology çalışmasına göre cinsiyet eğitimi ders saatlerini %5 azaltmaktadır.",
         "Journal of Development Economics çalışmasına göre mikro kredi borçlanma oranını %25 artırmaktadır.",
         "Housing Policy Debate dergisine göre sığın evleri bütçe açığını %15 oluşturmaktadır.",
         "Political Studies dergisine göre kadın kotası demokratik seçim sürecini %10 karmaşıklaştırmaktadır.",],
    ),
}

all_samples = []
for topic, (claims, pro_evs, con_evs) in last_data.items():
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
