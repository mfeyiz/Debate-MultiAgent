#!/usr/bin/env python3
"""Generate expanded argument dataset with exact offsets (hand-crafted templates)."""

import json
import random
from pathlib import Path

random.seed(42)


# Templates for different domains - each produces a debate paragraph
def make_sample(claim, pro_evidences, con_evidences, topic):
    """Build a structured sample with exact text and offsets."""
    text_parts = [claim + " "]
    for ev in pro_evidences:
        text_parts.append(ev + " ")
    for ev in con_evidences:
        text_parts.append(ev + " ")
    text = "".join(text_parts).strip()

    # Build exact spans with character offsets
    claims = []
    evidences = []

    c_start = text.find(claim)
    if c_start != -1:
        claims.append({
            "id": "c1",
            "text": claim,
            "label": "claim",
            "start": c_start,
            "end": c_start + len(claim),
        })

    e_idx = 1
    for ev in pro_evidences + con_evidences:
        e_start = text.find(ev)
        if e_start != -1:
            evidences.append({
                "id": f"e{e_idx}",
                "text": ev,
                "label": "evidence",
                "start": e_start,
                "end": e_start + len(ev),
            })
            e_idx += 1

    support = []
    attack = []
    pro_count = len(pro_evidences)

    for i, ev in enumerate(evidences):
        if i < pro_count:
            support.append({"from": ev["id"], "to": "c1"})
        else:
            attack.append({"from": ev["id"], "to": "c1"})

    return {
        "data": {
            "text": text,
            "claim": claims,
            "Evidence": evidences,
            "support": support,
            "attack": attack,
            "topic": topic,
            "type": "opinion_forum",
        }
    }


# ---- YAPAY ZEKA ----
ai_claims = [
    "Yapay zeka teknolojilerinin iş hayatına entegrasyonu ekonomik büyümeyi hızlandırsa da ciddi riskler barındırmaktadır.",
    "Yapay zeka destekli otomasyon sistemleri gelecekte milyonlarca insanı işsiz bırakma potansiyeline sahiptir.",
    "Yapay zeka tabanlı teşhis sistemleri sağlık sektöründe hata oranlarını önemli ölçüde azaltmaktadır.",
    "Özerk yapay zeka sistemleri askeri alanda insan kontrolüne gerek kalmadan karar alabilmelidir.",
    "Yapay zeka eğitim sistemlerinde kullanılmalıdır çünkü öğrencilere kişiselleştirilmiş deneyimler sunar.",
    "Yapay zeka sanat eserleri üretebilir ancak bu eserler gerçek sanatsal değere sahip değildir.",
    "Yapay zeka destekli finansal danışmanlık insan uzmanların yerini tamamen almalıdır.",
    "Yapay zeka algoritmaları yargı sisteminde karar verme süreçlerinde kullanılmamalıdır.",
    "Yapay zeka destekli tarım teknolojileri gıda güvenliğini artırarak küresel açlığı sonlandırabilir.",
    "Yapay zeka çevre kirliliğini izlemede ve önlemede kritik bir araçtır.",
]

ai_pro = [
    "McKinsey Enstitüsü'nün 2024 verilerine göre yapay zeka otomasyonu şirketlerin üretim kapasitesini %25 oranında artırmaktadır.",
    "Dünya Ekonomik Forumu raporunda yapay zeka sektörünün önümüzdeki on yılda 12 milyon yeni istihdam yaratacağı öngörülmektedir.",
    "Nature dergisinde yayınlanan çalışmaya göre yapay zeka destekli teşhis sistemi meme kanseri tespitinde %94 doğruluk oranına ulaşmıştır.",
    "Google DeepMind'in AlphaFold sistemi 200 milyon protein yapısını çözerek ilaç geliştirme sürecini yıllarca kısaltmıştır.",
    "UNESCO raporuna göre yapay zeka destekli eğitim platformları öğrencilerin öğrenme hızını %30 oranında artırmaktadır.",
    "MIT Media Lab araştırması yapay zeka algoritmalarının iklim modellerinde tahmin doğruluğunu %40 iyileştirdiğini göstermektedir.",
    "Stanford HAI raporuna göre yapay zeka destekli tarım sistemleri su kullanımını %50 oranında azaltabilmektedir.",
    "Gartner analizine göre 2025 yılına kadar şirketlerin %80'i yapay zeka destekli müşteri hizmeti kullanacaktır.",
    "IBM Watson Health verilerine göre yapay zeka kanser tedavisinde tedavi planlarını %60 daha hızlı optimize edebilmektedir.",
    "PWC raporuna göre yapay zeka küresel ekonomiye 2030 yılına kadar 15 trilyon dolar katkı sağlayacaktır.",
]

ai_con = [
    "Uluslararası Çalışma Örgütü ofis çalışanlarının %30'unun önümüzdeki beş yıl içinde işsiz kalma riskiyle karşı karşıya olduğunu belirtmektedir.",
    "Oxford Üniversitesi araştırması alt gelir grubundaki işçilerin %47'sinin yapay zeka nedeniyle maaş kesintisi yaşayabileceğini kanıtlamaktadır.",
    "Amnesty International raporuna göre yapay zeka yüz tanıma sistemleri etnik azınlıklara karşı %34 daha fazla hata yapmaktadır.",
    "MIT Technology Review makalesine göre büyük dil modelleri eğitim verilerindeki önyargıları sistematik olarak pekiştirmektedir.",
    "Harvard Business Review analizine göre yapay zeka destekli işe alım algoritmaları kadın adaylara karşı %20 daha az olumlu sonuç vermektedir.",
    "Electronic Frontier Foundation raporuna göre yapay zeka destekli gözetim sistemleri kişisel mahremiyeti ciddi şekilde tehdit etmektedir.",
    "Science dergisindeki araştırmaya göre yapay zeka modelleri eğitiminde kullanılan elektrik miktarı binlerce hanenin yıllık tüketimine eşittir.",
    "Brookings Enstitüsü analizine göre yapay zeka otomasyonu en çok orta gelirli işçileri etkileyerek gelir eşitsizliğini artırmaktadır.",
    "The Lancet çalışmasına göre yapay zeka destekli teşhis sistemleri az gelişmiş bölgelerde sağlık eşitsizliğini derinleştirmektedir.",
    "Yale Law School araştırmasına göre yapay zeka destekli yargı algoritmaları siyah sanıklara daha yüksek risk skorları atamaktadır.",
]

# ---- ÇALIŞMA MODELLERİ ----
work_claims = [
    "Haftada dört gün çalışma modeli hem çalışanlar hem de şirketler için avantajlı bir sistemdir.",
    "Uzaktan çalışma kalıcı olarak benimsenmeli ve ofis tabanlı çalışma modelleri terk edilmelidir.",
    "Esnek çalışma saatleri çalışan verimliliğini artırır ve iş-yaşam dengesini olumlu yönde etkiler.",
    "Gig ekonomisi modern iş gücü piyasasının kaçınılmaz bir parçasıdır ve desteklenmelidir.",
    "Asgari ücretin saatlik olarak belirlenmesi yerine evrensel temel gelir sistemi uygulanmalıdır.",
    "Yıllık izin süresi en az 30 gün olmalı ve zorunlu kullanım şartı getirilmelidir.",
    "Kamuda hibrit çalışma modeli kalıcı hale getirilmeli ve çalışanlara seçim hakkı tanınmalıdır.",
    "İş yerinde psikolojik baskı yasalarla açıkça tanımlanmalı ve cezai yaptırımlar uygulanmalıdır.",
    "Kreş desteği zorunlu hale getirilmeli ve işverenler çocuk bakım maliyetlerinin %50'sini karşılamalıdır.",
    "Emeklilik yaşı kademeli olarak 70'e çıkarılmalıdır çünkü ortalama yaşam süresi artmaktadır.",
]

work_pro = [
    "İzlanda Hükümeti'nin 2021 pilot uygulama sonuçlarına göre dört günlük çalışmada verimlilik %15 artmıştır.",
    "Cambridge Üniversitesi araştırması esnek çalışma sisteminde tükenmişlik vakalarının %71 azaldığını göstermektedir.",
    "Microsoft Work Trend Index raporu uzaktan çalışmanın %73 çalışanı daha memnun ettiğini ortaya koymaktadır.",
    "Stanford Ekonomi Bölümü çalışmasına göre uzaktan çalışanlar ofis çalışanlarına göre günlük %13 daha fazla iş üretmektedir.",
    "Gallup anketi sonuçlarına göre esnek çalışma saatleri olan şirketlerde çalışan bağlılığı %25 daha yüksektir.",
    "Deloitte araştırmasına göre hibrit çalışma modeli kullanan şirketlerde işe alım maliyetleri %30 azalmaktadır.",
    "Harvard Business School analizine göre dört günlük çalışma haftası hastalık izni kullanımını %40 azaltmaktadır.",
    "OECD verilerine göre kreş desteği olan ülkelerde kadın iş gücüne katılım oranı %20 daha yüksektir.",
    "McKinsey raporuna göre evrensel temel gelir pilot programlarında girişimcilik oranı %17 artmıştır.",
    "Forbes dergisi araştırmasına göre psikolojik güvenliği yüksek ekiplerde yenilikçilik %35 daha fazladır.",
]

work_con = [
    "Avrupa Merkez Bankası verileri üretim sektöründe mesai azalmasının toplam üretim hacminde %8 daralmaya yol açabileceğine işaret etmektedir.",
    "Forbes sektörel analizine göre KOBİ'lerin %40'ı dört günlük sistemin getirdiği ek personel maliyetlerini karşılayamamaktadır.",
    "Goldman Sachs raporuna göre tam uzaktan çalışma şirket kültürünü zayıflatmakta ve bilgi transferini %20 azaltmaktadır.",
    "Bloomberg analizine göre gig ekonomisi çalışanlarının %58'i sağlık sigortası ve emeklilik gibi temel haklardan yoksun kalmaktadır.",
    "IMF çalışmasına göre evrensel temel gelir uygulaması kamu bütçesine yılda GSYH'nin %10-35'i arasında maliyet getirmektedir.",
    "Journal of Applied Psychology araştırmasına göre uzaktan çalışma yeni çalışanların mentorluk almasını %45 zorlaştırmaktadır.",
    "The Economist analizine göre ofis sosyalleşmesinin azalması çalışan yalnızlığını %28 artırmaktadır.",
    "World Bank raporuna göre emeklilik yaşının yükseltilmesi genç işsizlik oranlarını %15 artırmaktadır.",
    "Society for Human Resource Management verilerine göre hibrit modelde yönetim maliyetleri %22 artmaktadır.",
    "National Bureau of Economic Research çalışmasına göre uzaktan çalışma kariyer ilerleme hızını %18 yavaşlatmaktadır.",
]

# ---- ELEKTRİKLİ ARAÇLAR ----
ev_claims = [
    "Elektrikli araçların yaygınlaşması çevre kirliliğini önlemede kritik bir rol oynamaktadır.",
    "Hidrojen yakıt hücreli araçlar elektrikli araçlardan daha sürdürülebilir bir gelecek vaat etmektedir.",
    "Şehir içi ulaşımda bisiklet ve yürüyüş yollarına yapılan yatırım otomobil bağımlılığını azaltmalıdır.",
    "Otonom sürüş teknolojisi trafik kazalarını %90 oranında azaltma potansiyeline sahiptir.",
    "Kamusal ulaşım sistemleri ücretsiz hale getirilmeli ve özel araç kullanımı kısıtlanmalıdır.",
    "Uçak seyahatleri karbon vergisi uygulanarak pahalılaştırılmalı ve tren taşımacılığı teşvik edilmelidir.",
    "Lityum iyon batarya üretimi çevre kirliliğine yol açtığı için farklı enerji depolama teknolojilerine geçilmelidir.",
    "Akıllı trafik yönetim sistemleri şehirlerdeki trafik sıkışıklığını %30 oranında azaltabilmektedir.",
    "Paylaşımlı mobilite uygulamaları şehirlerdeki araç sayısını %60 oranında azaltabilmektedir.",
    "Yüksek hızlı tren hatları kısa mesafe uçuşların yerini tamamen almalıdır.",
]

ev_pro = [
    "Uluslararası Enerji Ajansı 2023 raporu elektrikli araç kullanımının kentsel karbon emisyonlarını yıllık %18 düşürdüğünü belgelemektedir.",
    "Greenpeace veri setleri elektrikli araç geçişinin solunum yolu hastalıklarını %22 azalttığını ortaya koymaktadır.",
    "BloombergNEF analizine göre 2027 yılında elektrikli araçların maliyeti içten yanmalı motorlu araçlarla aynı seviyeye inecektir.",
    "Waymo verilerine göre otonom araçlar 2024 yılında milyonlarca mil sürüşte insan sürücülere göre %85 daha az kaza yapmıştır.",
    "ITF raporuna göre ücretsiz kamusal ulaşım uygulayan şehirlerde trafik yoğunluğu %12 azalmaktadır.",
    "Eurostat verilerine göre yüksek hızlı trenler uçaklara göre km başına %90 daha az karbon salımı gerçekleştirmektedir.",
    "Lloyd's Register araştırmasına göre hidrojen yakıt hücreli araçlar sıfır emisyonla bin kilometre menzil sunabilmektedir.",
    "Siemens akıllı trafik projesi Münih'te seyahat sürelerini %25 kısaltmıştır.",
    "McKinsey raporuna göre paylaşımlı mobilite uygulamaları talep üzerine hizmet sunarak araç başına kullanımı %40 artırmaktadır.",
    "Transport & Environment çalışmasına göre Avrupa'daki bisiklet yolu yatırımları bisiklet kullanımını %300 artırmıştır.",
]

ev_con = [
    "MIT yaşam döngüsü analizine göre lityum iyon batarya üretimi geleneksel araç üretiminden %40 daha fazla sera gazı açığa çıkarmaktadır.",
    "WWF raporuna göre batarya geri dönüşüm tesislerinin yetersizliği sebebiyle her yıl 2 milyon ton zehirli atık toprağa karışmaktadır.",
    "Reuters haberine göre elektrikli araç bataryalarında kullanılan kobaltın %70'i çocuk işçiliği sorunu yaşayan Kongo'dan çıkarılmaktadır.",
    "Nature Climate Change çalışmasına göre hidrojen üretiminin %95'i hâlâ fosil yakıtlardan elde edilmektedir.",
    "The Guardian araştırmasına göre otonom araçların yazılım hataları 2023 yılında ciddi kazalara yol açmıştır.",
    "OECD raporuna göre ücretsiz ulaşım belediye bütçelerine yılda milyonlarca dolar ek maliyet getirmektedir.",
    "Energy Policy dergisine göre elektrikli araç şarj altyapısı inşaatı beton ve çelik kullanımını artırarak inşaat sektörü emisyonlarını yükseltmektedir.",
    "Financial Times analizine göre havayolu şirketleri karbon vergisini bilet fiyatlarına yansıtarak orta sınıf seyahatini %30 pahalılaştırmaktadır.",
    "Science Advances araştırmasına göre akıllı trafik sistemleri yatırım maliyeti geleneksel sistemlere göre 5 kat daha yüksektir.",
    "Consumer Reports anketine göre şarj istasyonu yetersizliği elektrikli araç sahiplerinin %45'inde menzil kaygısı yaratmaktadır.",
]

# ---- SAĞLIK TEKNOLOJİLERİ ----
health_claims = [
    "Tele-tıp uygulamaları sağlık hizmetlerine erişimi demokratikleştirmeli ve kalıcı hale getirilmelidir.",
    "Genetik mühendisliği ile tasarlanmış bebekler etik sınırları zorladığı için yasaklanmalıdır.",
    "Biyometrik veriler sağlık sigortası primlerinin kişiselleştirilmesinde kullanılabilmelidir.",
    "Klonlama teknolojisi organ yetmezliği tedavisinde etik bir çözüm sunmaktadır.",
    "Aşı karşıtlığı toplum sağlığını tehdit ettiği için sosyal medyada kısıtlanmalıdır.",
    "Robotik cerrahi standart cerrahi yöntemlerden daha güvenli ve daha az invazivdir.",
    "Büyük veri analizi salgın hastalıkları önceden tespit etmede kritik bir rol oynamaktadır.",
    "Kişiselleştirilmiş ilaç tedavisi genetik profil analizi gerektirdiği için güvenli değildir.",
    "Sanal gerçeklik terapisi psikolojik tedavilerde geleneksel yöntemlerin yerini almalıdır.",
    "İmplant teknolojisi insan yeteneklerini artırarak toplumsal eşitsizliği derinleştirecektir.",
]

health_pro = [
    "American Medical Association raporuna göre tele-tıp uygulamaları kırsal bölgelerde sağlık erişimini %45 artırmıştır.",
    "Johns Hopkins Üniversitesi araştırmasına göre robotik cerrahi ameliyat komplikasyonlarını %32 azaltmaktadır.",
    "Lancet Digital Health çalışmasına göre yapay zeka destekli teşhis sistemleri deri kanseri tespitinde dermatologları geride bırakmıştır.",
    "CDC verilerine göre büyük veri analizi COVID-19 salgınının yayılma hızını tahmin etmede %85 başarı sağlamıştır.",
    "Nature Medicine makalesine göre kişiselleştirilmiş kanser ilacı tedavisi standart kemoterapiden %40 daha etkili olmaktadır.",
    "JAMA Psychiatry araştırmasına göre sanal gerçeklik terapisi PTSD semptomlarını geleneksel terapiden %25 daha hızlı azaltmaktadır.",
    "WHO raporuna göre aşılama programları çocukluk çağı ölümlerini yılda 4 milyon azaltmaktadır.",
    "NEJM çalışmasına göre biyometrik izleme sistemleri kronik hastalıklarda erken uyarı başarısını %60 artırmaktadır.",
    "Stanford Medicine raporuna göre CRISPR gen düzenleme teknobi orak hücre anemisinde %93 tedavi başarısı sağlamıştır.",
    "PLOS ONE araştırmasına göre 3D baskı organ modelleri cerrahi planlama hatalarını %50 azaltmaktadır.",
]

health_con = [
    "Nature Biotechnology çalışmasına göre CRISPR düzenlemelerinde istenmeyen gen mutasyonları ortaya çıkabilmektedir.",
    "Brookings Enstitüsü raporuna göre tele-tıp uygulamaları düşük gelirli hastaları daha az kaliteli muayeneye yönlendirmektedir.",
    "Science Translational Medicine araştırmasına göre kişiselleştirilmiş ilaç maliyetleri standart tedaviye göre 10 kat daha yüksektir.",
    "The Hastings Center raporuna göre biyometrik veri kullanımı sigorta şirketlerinin ayrımcılık yapmasına olanak tanımaktadır.",
    "Reuters araştırmasına göre aşı karşıtı içerikler sosyal medyada doğru bilgiden %300 daha fazla etkileşim almaktadır.",
    "BMJ analizine göre robotik cerrahi maliyetleri geleneksel cerrahiden %40 daha yüksektir.",
    "The Guardian haberine göre genetik test şirketleri müşteri verilerini üçüncü taraflara satmaktadır.",
    "Health Affairs dergisine göre sanal terapi uygulamalarında hasta-terapist bağlantısı yüz yüze tedaviye göre %30 daha zayıftır.",
    "AP News raporuna göre beyin implantları hacklenme riski taşımakta ve kişisel düşüncelerin sızdırılmasına yol açabilmektedir.",
    "RAND Corporation çalışmasına göre sağlık verisi platformları siber saldırılara karşı finansal kurumlardan %200 daha savunmasızdır.",
]

# ---- EĞİTİM SİSTEMİ ----
edu_claims = [
    "Uzaktan eğitim geleneksel yüz yüze eğitimin yerini tamamen almalıdır.",
    "Sınav merkezli değerlendirme sistemi öğrenci yaratıcılığını baskılamaktadır.",
    "STEM eğitimine yapılan yatırımlar sanat ve beşeri bilimlerin bütçesinden karşılanmamalıdır.",
    "Yapay zeka destekli öğretmen asistanları öğrenci başarısını artıracak etkili bir araçtır.",
    "Okul öncesi eğitim zorunlu hale getirilmeli ve ücretsiz sunulmalıdır.",
    "Üniversite eğitimi herkes için ücretsiz olmalı ve devlet tarafından finanse edilmelidir.",
    "Eğitimde oyunlaştırma öğrenci motivasyonunu artırır ancak uzun vadede öğrenmeyi zayıflatır.",
    "Öğretmenlerin maaşları doktor ve mühendislerle aynı seviyeye çıkarılmalıdır.",
    "Birebir öğretmen desteği alan öğrenciler sınıf ortamına göre %40 daha fazla akademik gelişme göstermektedir.",
    "Müfredat ulusal değil küresel standardizasyonla belirlenmelidir.",
]

edu_pro = [
    "UNESCO raporuna göre uzaktan eğitim platformları kırsal kesimdeki öğrenci erişimini %35 artırmıştır.",
    "PISA verilerine göre proje tabanlı değerlendirme sistemleri problem çözme becerilerini %28 geliştirmektedir.",
    "Stanford HAI araştırmasına göre yapay zeka asistanları öğretmenlerin idari iş yükünü %40 azaltmaktadır.",
    "OECD verilerine göre okul öncesi eğitim alan çocukların yetişkinlikte geliri %25 daha yüksektir.",
    "Brookings Enstitüsü analizine göre ücretsiz üniversite uygulamaları sosyal hareketliliği %18 artırmaktadır.",
    "Nature Human Behaviour çalışmasına göre oyunlaştırılmış öğrenme kısa vadeli bilgi tutma oranını %45 artırmaktadır.",
    "Economic Policy Institute raporuna göre öğretmen maaşlarındaki %10 artış öğrenci başarısını %5-10 yükseltmektedir.",
    "Journal of Educational Psychology araştırmasına göre küçük sınıf boyutu öğrenci başarısını %15 artırmaktadır.",
    "World Bank raporuna göre dijital okuryazarlık eğitimi gelişmekte olan ülkelerde istihdamı %20 artırmaktadır.",
    "Harvard Graduate School analizine göre uluslararası müfredat standartları üniversite hazırlığını %22 iyileştirmektedir.",
]

edu_con = [
    "MIT Technology Review raporuna göre uzaktan eğitimde öğrenci katılım oranı yüz yüze eğitime göre %30 daha düşüktür.",
    "The Atlantic makalesine göre sınavsız değerlendirme sistemleri öğrencilerin temel becerilerini %20 geriletmiştir.",
    "Forbes analizine göre sanat ve beşeri bilimler bütçelerinin azaltılması yaratıcı sektör istihdamını %15 düşürmüştür.",
    "Education Week araştırmasına göre yapay zeka asistanları öğretmen-öğrenci ilişkisini %25 zayıflatmaktadır.",
    "The Economist raporuna göre ücretsiz üniversite uygulamaları vergi yükünü GSYH'nin %2'sine çıkarmaktadır.",
    "Psychological Science çalışmasına göre oyunlaştırma ödül bağımlılığı yaratarak içsel motivasyonu %18 azaltmaktadır.",
    "Wall Street Journal haberine göre öğretmen maaş artışları okul bütçelerinin %60'ını tüketmektedir.",
    "Hechinger Report analizine göre birebir öğretmen desteği maliyeti grup eğitiminden 8 kat daha yüksektir.",
    "UNESCO uyarısına göre küresel müfredat standardizasyonu yerel kültürel değerlerin %40'ını kaybettirmektedir.",
    "Nature çalışmasına göre uzaktan eğitimde öğrenci yalnızlığı ve depresyon oranları %35 artmıştır.",
]

# ---- İKLİM DEĞİŞİKLİĞİ ----
climate_claims = [
    "Karbon vergisi en etkili iklim değişikliği çözümüdür ve tüm ülkelerde uygulanmalıdır.",
    "Nükleer enerji fosil yakıtlardan tamamen geçiş yapana kadar geçici bir çözüm olarak benimsenmelidir.",
    "Güneş enerjisi kömürden daha ucuz hale geldiği için devlet sübvansiyonlarına artık gerek yoktur.",
    "Orman yangınlarını önlemek için kontrollü yanma uygulamaları artırılmalıdır.",
    "Deniz seviyesi yükselmesi tehdidi altındaki kıyı şehirleri terk edilmeli ve içeri taşınmalıdır.",
    "Hayvancılık sektörü sera gazı salımının %14'ünü oluşturduğu için bitkisel alternatiflere geçiş zorunludur.",
    "Karbon yakalama ve depolama teknolojileri enerji sektöründe zorunlu hale getirilmelidir.",
    "Bireysel karbon ayak izi hesaplayıcıları farkındalık yaratmakta etkisizdir ve kaynak israfıdır.",
    "İklim değişikliği mültecilerine iltica hakkı tanınmalı ve uluslararası hukuk buna uygun güncellenmelidir.",
    "Deniz altı madenciliği kritik mineraller için zorunludur ve ekosistem hasarı kabul edilebilir düzeydedir.",
]

climate_pro = [
    "World Bank raporuna göre karbon vergisi uygulayan ülkelerde emisyonlar %15-25 oranında azalmıştır.",
    "IEA analizine göre nükleer enerji Avrupa'da elektrik üretiminin %25'ini karşılayarak fosil yakıt bağımlılığını azaltmaktadır.",
    "BloombergNEF verilerine göre güneş enerjisi maliyeti son 10 yılda %89 düşmüştür.",
    "Nature Sustainability çalışmasına göre kontrollü yanma orman yangın riskini %50 azaltmaktadır.",
    "IPCC raporuna göre 2100 yılına kadar deniz seviyesi 1 metreyi aşarak 800 milyon kişiyi etkileyecektir.",
    "FAO verilerine göre hayvancılık küresel sera gazı emisyonlarının %14,5'ini tek başına oluşturmaktadır.",
    "Science çalışmasına göre karbon yakalama tesisleri yılda milyarlarca ton CO2 depolayabilmektedir.",
    "UNHCR raporuna göre 2050 yılına kadar iklim değişikliği nedeniyle 216 milyon kişi göç etmek zorunda kalacaktır.",
    "Geological Survey analizine göre deniz altı nodülleri kıta tabanı madenciliğinden %70 daha az karbon salımı içermektedir.",
    "NASA uydu verilerine göre Amazon yağmur ormanları son 20 yılda atmosfere daha fazla karbon salmaya başlamıştır.",
]

climate_con = [
    "IMF çalışmasına göre karbon vergisi düşük gelirli hanelere gelirlerinin %5'ine varan ek maliyet getirmektedir.",
    "Nature Energy raporuna göre nükleer santral inşaat maliyeti yenilenebilir enerjiye göre 3 kat daha yüksektir.",
    "Forbes analizine göre güneş paneli üretiminde silikon rafinajı kömürden %30 daha fazla zehirli atık üretmektedir.",
    "Environmental Science dergisine göre kontrollü yanma hava kalitesini %20 bozmaktadır.",
    "Brookings Enstitüsü raporuna göre kıyı şehirlerinin taşınması trilyonlarca dolar maliyet getirmektedir.",
    "Science Advances araştırmasına göre bitkisel et alternatifleri su kullanımını hayvancılığa göre %50 artırmaktadır.",
    "The Guardian haberine göre karbon yakalama tesisleri enerji tüketimi nedeniyle %21 verimlilik kaybına uğramaktadır.",
    "Yale Program on Climate Change anketine göre bireysel karbon hesaplayıcıları kullanıcı davranışını sadece %3 değiştirmektedir.",
    "The Lancet çalışmasına göre iklim mülteci tanımı mevcut hukuki çerçevede uygulanabilir değildir.",
    "Deep Sea Conservation Coalition raporuna göre deniz altı madenciliği bilinmeyen binlerce türü yok etme riski taşımaktadır.",
]

# ---- SOSYAL MEDYA ----
social_claims = [
    "Sosyal medya platformları kullanıcı verilerini reklam için satma hakkına sahiptir.",
    "Devletler sosyal medyada yalan haber yayınını yasal olarak cezalandırabilmelidir.",
    "Sosyal medya bağımlılığı gençlerde depresyon ve kaygı bozukluklarını artırmaktadır.",
    "Platformlar algoritmalarının şeffaflığını sağlamalı ve kullanıcılara kontrol imkanı tanımalıdır.",
    "Anonim hesaplar sosyal medyada ifade özgürlüğünü korur ancak nefret söylemini de yayar.",
    "Influencer pazarlaması tüketicileri bilinçsizce harcamaya teşvik ederek finansal zarara yol açmaktadır.",
    "Sosyal medya platformları çocuk kullanıcıları için ayrı ve güvenli arayüzler sunmalıdır.",
    "Mikro blog formatı derinlemesine düşünce üretimini engellemektedir.",
    "Kullanıcılar dijital miraslarının ölümlerinden sonra silinmesini talep etme hakkına sahiptir.",
    "Sosyal medya toplumsal hareketleri organize etmede demokratik bir araçtır.",
]

social_pro = [
    "Pew Research Center anketine göre sosyal medya haber tüketiminin %68'ini oluşturmaktadır.",
    "Journal of Adolescent Health çalışmasına göre günlük 3 saatten fazla sosyal medya kullanımı gençlerde depresyon riskini %26 artırmaktadır.",
    "MIT Media Lab araştırmasına göre şeffaf algoritmalar kullanıcı güvenini %40 artırmaktadır.",
    "Edelman Trust Barometer raporuna göre influencer pazarlaması genç tüketicilerde marka güvenini %35 yükseltmektedir.",
    "UNICEF analizine göre çocuk dostu platform arayüzleri çevrimiçi zorbalığı %30 azaltmaktadır.",
    "Oxford Internet Institute çalışmasına göre sosyal medya kampanyaları sivil katılımı %45 artırmaktadır.",
    "Stanford HAI raporuna göre kullanıcı kontrollü akış algoritmaları memnuniyeti %25 artırmaktadır.",
    "Nature Human Behaviour makalesine göre anonim platformlar muhalif görüşlerin %60 daha fazla paylaşılmasına olanak tanımaktadır.",
    "Pew Research verilerine göre sosyal medya hareketleri seçim katılımını %7 artırmıştır.",
    "Digital Marketing Institute raporuna göre mikro blog formatı bilgi erişimini %50 hızlandırmaktadır.",
]

social_con = [
    "The Guardian haberine göre Meta platformları kullanıcı verilerini yılda 114 milyar dolarlık reklam geliri için işlemektedir.",
    "Freedom House raporuna göre yalan haber yasaları %65 ülkede ifade özgürlüğünü kısıtlamaktadır.",
    "JAMA Psychiatry araştırmasına göre sosyal medya bağımlılığı tedavisinde uzun vadeli başarı oranı sadece %35'tir.",
    "ProPublica analizine göre algoritma şeffaflığı rekabet avantajı kaybına yol açtığı için şirketler direnmektedir.",
    "Southern Poverty Law Center raporuna göre anonim hesaplar nefret söyleminin %70'ini üretmektedir.",
    "Bankrate anketi sonuçlarına göre influencer etkisiyle alışveriş yapan gençlerin %42'si borçlanmaktadır.",
    "Common Sense Media araştırmasına göre çocuk dostu arayüzler yetişkin içerik filtreleme başarısını %20 düşürmektedir.",
    "MIT Technology Review makalesine göre mikro blog formatı yanlış anlaşılmaları %40 artırmaktadır.",
    "Reuters Institute raporuna göre dijital miras yasaları sadece %15 ülkede mevcuttur.",
    "Science Advances çalışmasına göre sosyal medya protestoları somut politik değişiklik üretme oranı sadece %12'dir.",
]

# ---- GIDA TEKNOLOJİSİ ----
food_claims = [
    "Laboratuvar üretimi et hayvancılığın yerini almalı ve küresel ölçekte teşvik edilmelidir.",
    "Genetiği değiştirilmiş organizmalar gıda güvenliği açısından risk taşıdığı için yasaklanmalıdır.",
    "Dikey tarım şehirlerde gıda güvenliğini artırarak tarım arazisi ihtiyacını azaltmaktadır.",
    "İnsek proteinleri geleneksel hayvansal proteinlere sürdürülebilir bir alternatif sunmaktadır.",
    "Yapay tatlandırıcılar şekerden daha zararlıdır ve gıda etiketlerinde uyarıcı ibare bulunmalıdır.",
    "Gıda israfını önlemek için süpermarketler yaklaşan son kullanma tarihli ürünleri ücretsiz dağıtmalıdır.",
    "Su ürünleri yetiştiriciliği doğal balık stoklarını kurtarmak için tek çözümdür.",
    "Hazır gıda endüstrisi obezite oranlarını artırdığı için ağır vergilendirilmelidir.",
    "Yerel ve mevsimsel gıda tüketimi ithal gıdalara göre karbon ayak izini %30 azaltmaktadır.",
    "Gıda ambalajındaki mikroplastikler insan sağlığına ciddi zarar vermektedir.",
]

food_pro = [
    "Nature Food dergisine göre laboratuvar eti su kullanımını hayvancılığa göre %96 azaltmaktadır.",
    "FAO raporuna göre dikey tarım sistemleri dönüm başına geleneksel tarıma göre 10 kat fazla ürün vermektedir.",
    "Journal of Cleaner Production çalışmasına göre insek yetiştiriciliği metan salımını sığır yetiştiriciliğine göre %99 azaltmaktadır.",
    "European Food Safety Authority analizine göre onaylı GMO ürünleri 25 yıldır sağlık riski oluşturmamaktadır.",
    "Harvard T.H. Chan School araştırmasına göra yapay tatlandırıcılar kan şekeri üzerinde doğal şekerden %30 daha az etki yapmaktadır.",
    "WRAP raporuna göre gıda israfı önleme uygulamaları yılda 1,9 milyon ton gıda kurtarmaktadır.",
    "Nature Communications çalışmasına göre su ürünleri yetiştiriciliği doğal balık popülasyonlarını %35 toparlamaktadır.",
    "Local Food Systems analizine göre yerel gıda ağları ekonomik dönüşümü %25 hızlandırmaktadır.",
    "Environmental Health Perspectives dergisine göre mevsimsel sebze tüketimi pestisit maruziyetini %60 azaltmaktadır.",
    "Science of the Total Environment çalışmasına göre biyobozunur ambalaj kullanımı mikroplastik kirliliğini %80 azaltmaktadır.",
]

food_con = [
    "Consumer Reports anketi sonuçlarına göre laboratuvar eti fiyatları geleneksel etten %300 daha pahalıdır.",
    "Greenpeace raporuna göre dikey tarım enerji tüketimi açık tarıma göre 5 kat daha yüksektir.",
    "Journal of Insects as Food and Feed çalışmasına göre Batı tüketicilerinin %70'i insek proteinini reddetmektedir.",
    "Union of Concerned Scientists analizine göre GMO tekelleri tohum çeşitliliğini %75 azaltmaktadır.",
    "PLOS Medicine araştırmasına göre yapay tatlandırıcılar bağırsak mikrobiyomunu %20 bozmaktadır.",
    "Financial Times haberine göre ücretsiz gıda dağıtımı perakende sektöründe fiyat dengesizliği yaratmaktadır.",
    "Marine Policy dergisine göre aşırı su ürünleri yetiştiriciliği kıyı ekosistemlerini %40 tahrip etmektedir.",
    "Obesity Reviews meta-analizine göre hazır gıda vergisi obezite oranlarını sadece %3 azaltmaktadır.",
    "Journal of Industrial Ecology çalışmasına göre sera yetiştiriciliği ithal sebzeden %200 daha fazla karbon salmaktadır.",
    "Environmental Science & Technology raporuna göre mikroplastik maruziyeti günlük 5 gramı aşabilmektedir.",
]

# Combine all domains
domains = [
    (ai_claims, ai_pro, ai_con, "yapay_zeka"),
    (work_claims, work_pro, work_con, "calisma_modelleri"),
    (ev_claims, ev_pro, ev_con, "elektrikli_araclar"),
    (health_claims, health_pro, health_con, "saglik_teknolojileri"),
    (edu_claims, edu_pro, edu_con, "egitim_sistemi"),
    (climate_claims, climate_pro, climate_con, "iklim_degisikligi"),
    (social_claims, social_pro, social_con, "sosyal_medya"),
    (food_claims, food_pro, food_con, "gida_teknolojisi"),
]

all_samples = []

for claims, pro_evs, con_evs, topic in domains:
    for i in range(min(len(claims), len(pro_evs), len(con_evs))):
        sample = make_sample(
            claims[i],
            [pro_evs[i]],
            [con_evs[i]],
            topic
        )
        all_samples.append(sample)

# Now add samples with 2 pro + 2 con for richer structure
for claims, pro_evs, con_evs, topic in domains:
    for i in range(min(5, len(claims), len(pro_evs) - 1, len(con_evs) - 1)):
        idx = (i + 5) % len(claims)
        idx2 = (i + 6) % len(pro_evs)
        sample = make_sample(
            claims[idx],
            [pro_evs[idx], pro_evs[idx2]],
            [con_evs[idx], con_evs[idx2]],
            topic
        )
        all_samples.append(sample)

print(f"Generated {len(all_samples)} new samples")

# Load existing cleaned data
with open("data/argument_dataset_cleaned.json") as f:
    existing = json.load(f)

# Combine
combined = existing + all_samples
random.shuffle(combined)

# Reassign IDs
for i, item in enumerate(combined):
    item["id"] = i + 1

with open("data/argument_dataset.json", "w", encoding="utf-8") as f:
    json.dump(combined, f, ensure_ascii=False, indent=2)

print(f"Total combined dataset: {len(combined)} samples")

# Verify all spans have exact offsets
bad = 0
for item in combined:
    d = item["data"]
    text = d["text"]
    for c in d.get("claim", []):
        if text[c["start"]:c["end"]] != c["text"]:
            bad += 1
    for e in d.get("Evidence", []):
        if text[e["start"]:e["end"]] != e["text"]:
            bad += 1

print(f"Offset mismatches: {bad}")
