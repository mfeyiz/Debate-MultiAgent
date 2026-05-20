#!/usr/bin/env python3
"""Expand dataset further with more variations and topics."""

import json
import random
from pathlib import Path

random.seed(42)


def make_sample(claim, pro_evidences, con_evidences, topic):
    """Build structured sample with exact offsets."""
    text = claim + " " + " ".join(pro_evidences + con_evidences)
    text = text.strip()

    claims = []
    c_start = text.find(claim)
    if c_start != -1:
        claims.append({
            "id": "c1", "text": claim, "label": "claim",
            "start": c_start, "end": c_start + len(claim),
        })

    evidences = []
    for i, ev in enumerate(pro_evidences + con_evidences):
        e_start = text.find(ev)
        if e_start != -1:
            evidences.append({
                "id": f"e{i+1}", "text": ev, "label": "evidence",
                "start": e_start, "end": e_start + len(ev),
            })

    support = [{"from": e["id"], "to": "c1"} for e in evidences[:len(pro_evidences)]]
    attack = [{"from": e["id"], "to": "c1"} for e in evidences[len(pro_evidences):]]

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


# More claims and evidences for each domain
more_data = {
    "yapay_zeka": (
        [
            "Yapay zeka destekli hukuk sistemleri adaletin daha hızlı ve ucuz ulaşılabilir olmasını sağlamaktadır.",
            "Otonom silah sistemleri uluslararası hukuk çerçevesinde mutlaka yasaklanmalıdır.",
            "Yapay zeka destekli müzik besteleme sanatçıların yerini tamamen alacak niteliktedir.",
            "Akıllı şehir sistemleri trafik kazalarını ve enerji israfını önemli ölçüde azaltmaktadır.",
            "Yapay zeka destekli psikoterapi botları ruh sağlığı hizmetlerine erişimi demokratikleştirmektedir.",
            "Algoritmik yönetim çalışanları sürekli gözetim altında tutarak insan onurunu zedelemektedir.",
            "Yapay zeka destekli çeviri sistemleri küresel iletişimi hızlandırmakta ancak kültürel nüansları kaybetmektedir.",
            "Derin öğrenme modelleri bilimsel keşif sürecini yıllardan aylara indirgemektedir.",
            "Yapay zeka destekli stil danışmanlığı moda endüstrisinde kişiselleştirilmiş deneyim sunmaktadır.",
            "Algoritmik haber öneri sistemleri bilgi balonları oluşturarak toplumsal kutuplaşmayı derinleştirmektedir.",
        ],
        [
            "Harvard Law Review makalesine göre yapay zeka destekli hukuk platformları dava maliyetlerini %40 azaltmaktadır.",
            "Future of Life Institute raporuna göre 50 ülkeden fazlası otonom silahları yasaklayan uluslararası anlaşma çağrısı yapmaktadır.",
            "Spotify verilerine göre yapay zeka besteleri 2024 yılında platformdaki enstrümantal müziğin %12'sini oluşturmaktadır.",
            "Smart Cities World raporuna göre akıllı trafik yönetimi kazaları %30 azaltmaktadır.",
            "APA dergisine göre terapi botları anksiyete semptomlarını %20 azaltmaktadır.",
            "The Atlantic makalesine göre Amazon algoritmik yönetimi çalışanların %30'unu yılda terk etmeye zorlamaktadır.",
            "MIT Technology Review araştırmasına göre yapay zeka çevirisi edebi metinlerde %15 hata oranına sahiptir.",
            "Nature Methods dergisine göre AlphaFold üç boyutlu protein yapısı tahminini yıllardan dakikalara indirmiştir.",
            "Vogue Business analizine göre yapay zeka stil danışmanlığı online moda satışlarını %25 artırmıştır.",
            "PNAS çalışmasına göre algoritmik haber önerileri kullanıcıların farklı kaynaklara maruz kalmasını %60 azaltmaktadır.",
        ],
        [
            "Electronic Frontier Foundation raporuna göre yapay zeka hukuk sistemleri azınlık sanıklara karşı %20 daha fazla hata yapmaktadır.",
            "RAND Corporation çalışmasına göre otonom silahların yasaklanması silahlanma yarışını durdurmayacaktır.",
            "Grammy ödül komitesi yapay zeka bestelerinin ödül almasını etik açıdan uygun bulmamaktadır.",
            "The Guardian haberine göre akıllı şehir sensörleri vatandaş hareketlerini izleyerek mahremiyeti ihlal etmektedir.",
            "British Journal of Psychiatry araştırmasına göre terapi botları ciddi depresyon vakalarında etkisiz kalmaktadır.",
            "Worker Institute raporuna göre algoritmik yönetim çalışanların %45'inde stres bozukluğu yaratmaktadır.",
            "UNESCO analizine göre kültürel çeviri hataları diplomatik krizlere yol açabilmektedir.",
            "Science dergisine göre yapay zeka bilimsel keşiflerinde tekrarlanabilirlik oranı sadece %60'tır.",
            "Fashion Revolution raporuna göre algoritmik moda hızlı tüketimi %35 artırarak atığı çoğaltmaktadır.",
            "Reuters Institute raporuna göre haber balonları seçim dönemlerinde manipülasyona açık hale getirmektedir.",
        ],
    ),
    "calisma_modelleri": (
        [
            "Dijital göçebelik vizası ülkelerin ekonomik büyümesini artırarak küresel yetenek rekabetini desteklemektedir.",
            "İş yerinde zihinsel sağlık günleri zorunlu tatil politikalarına dahil edilmelidir.",
            "Kıdem tazminatı sistemi işçi güvencesini artırır ancak işverenlerin yeni istihdam yaratmasını engellemektedir.",
            "Stajyerlerin ücret alması zorunlu hale getirilmeli ve asgari ücretin %80'i ödenmelidir.",
            "Gözetim teknolojileri iş yerinde verimliliği artırmakta ancak çalışan güvenini zedelemektedir.",
            "Ebeveyn izni süresi her çocuk için 2 yıla çıkarılmalı ve ücretli olmalıdır.",
            "Mesleki eğitim programları lise müfredatının zorunlu bir parçası olmalıdır.",
            "İş yerinde kıyafet özgürlüğü çalışan verimliliğini artırmakta ve kreatif sektörlerde olumlu etki yapmaktadır.",
            "Çalışan hissedarlık programları şirketlerin uzun vadeli başarısını artırmaktadır.",
            "Yapay zeka destekli işe alım süreçleri insan kaynakları departmanlarının verimliliğini artırmaktadır.",
        ],
        [
            "OECD verilerine göre dijital göçebelik vizası uygulayan ülkelerde startup sayısı %45 artmıştır.",
            "WHO raporuna göre zihinsel sağlık günleri çalışan verimliliğini %12 artırmaktadır.",
            "World Bank analizine göre kıdem tazminatı işçi devir hızını %25 azaltmaktadır.",
            "Fair Labor Association raporuna göre ücretli staj uygulamaları stajyer memnuniyetini %60 artırmaktadır.",
            "MIT Sloan Management Review çalışmasına göre çalışan hissedarlığı şirket değerini %15 artırmaktadır.",
            "UNICEF verilerine göre uzatılmış ebeveyn izni çocuk gelişimi indeksini %20 yükseltmektedir.",
            "Harvard Graduate School araştırmasına göre mesleki eğitim mezunların istihdam oranını %30 artırmaktadır.",
            "Journal of Experimental Social Psychology çalışmasına göra kıyafet özgürlüğü kreatif performansı %18 artırmaktadır.",
            "Gallup raporuna göre hissedar çalışanlar şirketlerine %35 daha fazla sadık kalmaktadır.",
            "SHRM verilerine göre yapay zeka işe alım süreçlerini %50 hızlandırmaktadır.",
        ],
        [
            "Migration Policy Institute raporuna göre dijital göçebelik yerel konut fiyatlarını %20 artırmaktadır.",
            "Employer Costs analizine göre zihinsel sağlık günleri küçük işletmelere yılda binlerce dolar maliyet getirmektedir.",
            "IMF çalışmasına göre kıdem tazminatı maliyeti GSYH'nin %1,5'ine ulaşabilmektedir.",
            "Forbes haberine göre zorunlu ücretli staj KOBİ'lerin %55'ini finansal zorluk yaşatmaktadır.",
            "Privacy International raporuna göre iş yeri gözetimi çalışanların %70'inde rahatsızlık yaratmaktadır.",
            "The Economist analizine göre 2 yıllık ebeveyn izni kadınların kariyer ilerlemesini %40 yavaşlatmaktadır.",
            "Education Next dergisine göre zorunlu mesleki eğitim akademik yeterliliği %15 düşürmektedir.",
            "Wall Street Journal haberine göre kıyafet özgürlüğü müşteri memnuniyetini %10 azaltmaktadır.",
            "Financial Times raporuna göre hissedarlık programları yönetim maliyetlerini %20 artırmaktadır.",
            "The Guardian araştırmasına göre yapay zeka işe alımı kadın ve azınlık başvurularını %30 filtrelemektedir.",
        ],
    ),
    "elektrikli_araclar": (
        [
            "Hibrit araçlar tam elektriklere geçiş sürecinde en pratik çözüm olarak benimsenmelidir.",
            "Uçak yakıt vergisi kısa mesafe uçuşları caydırmak için %200 oranında artırılmalıdır.",
            "Otobüs filolarının tamamen elektriklenmesi şehir içi hava kalitesini radikal şekilde iyileştirecektir.",
            "Kamyon lojistiğinde otonom konvoy sistemleri yakıt tüketimini %15 azaltmaktadır.",
            "Bisiklet paylaşım sistemleri metropollerdeki trafik sıkışıklığını %25 azaltmaktadır.",
            "Elektrikli tekne teknolojisi deniz ticaretinde karbon emisyonlarını yarıya indirebilmektedir.",
            "Şehirlerde araç trafiğine tam yasak getirilmeli ve yaya bölgeleri genişletilmelidir.",
            "Hava taksisi uygulamaları kentsel ulaşımda trafik yoğunluğunu azaltmak yerine artırmaktadır.",
            "Güneş enerjili uçaklar ticari havacılıkta devrim yaratacak potansiyele sahiptir.",
            "Kentsel raylı sistem yatırımları otobüs taşımacılığına göre %60 daha az karbon salmaktadır.",
        ],
        [
            "JATO Dynamics verilerine göre hibrit araç satışları 2024 yılında %40 artmıştır.",
            "Eurostat verilerine göre şehir içi otobüs filolarının elektriklenmesi NOx emisyonlarını %90 azaltmaktadır.",
            "Platooning Europe raporuna göre otonom kamyon konvoyları yakıt tasarrufu sağlamaktadır.",
            "ITF çalışmasına göre bisiklet paylaşım sistemleri araç trafiğini %25 azaltmaktadır.",
            "Maritime Executive raporuna göre elektrikli feribotlar liman emisyonlarını %50 düşürmektedir.",
            "C40 Cities raporuna göre yaya bölgeleri perakende satışlarını %20 artırmaktadır.",
            "NASA araştırmasına göre güneş enerjili uçak prototipleri 26 saat kesintisiz uçuş gerçekleştirmiştir.",
            "UITP verilerine göre metro sistemleri otobüslere göre yolcu başına %70 daha az enerji tüketmektedir.",
            "McKinsey raporuna göre hibrit geçiş süreci altyapı maliyetlerini %60 azaltmaktadır.",
            "Transport & Environment çalışmasına göre yaya şehirlerinde çocuk obezitesi %18 azalmaktadır.",
        ],
        [
            "Consumer Reports anketi sonuçlarına göre hibrit araç bakım maliyetleri elektriklere göre %40 daha yüksektir.",
            "IATA raporuna göre uçak yakıt vergisi artışı bilet fiyatlarını %30 yükselterek seyahati sınırlamaktadır.",
            "BloombergNEF analizine göre elektrikli otobüs altyapı maliyeti otobüs başına 300 bin dolara ulaşmaktadır.",
            "Trucks.com haberine göre otonom konvoy teknolojisi kamyon şoförü istihdamını %50 azaltacaktır.",
            "The Guardian araştırmasına göre bisiklet paylaşım sistemleri bakım maliyetleri şirketleri iflasa sürüklemektedir.",
            "Marine Insight raporuna göre elektrikli tekne menzili geleneksel teknelerin %40'ı kadardır.",
            "Local Government Chronicle raporuna göre araç trafiği yasağı perakende satışlarını %35 azaltmaktadır.",
            "Urban Air Mobility analizine göre hava taksisi gürültü kirliliğini %200 artırmaktadır.",
            "Aviation Week raporuna göre güneş enerjili uçaklar ticari yolcu taşımacılığında yetersiz kalmaktadır.",
            "Railway Gazette haberine göre metro inşaat maliyeti otobüs hatlarına göre 10 kat daha yüksektir.",
        ],
    ),
    "saglik_teknolojileri": (
        [
            "Mobil sağlık uygulamaları kronik hastalıkların takibinde standart sağlık hizmetlerinin yerini almalıdır.",
            "İnsülin pompa teknolojisi diyabet yönetiminde manuel enjeksiyondan daha güvenli ve etkilidir.",
            "Beyin-bilgisayar arayüzleri felçli hastaların iletişim kurmasında çığır açmaktadır.",
            "Yapay zeka destekli radyoloji okumaları insan radyologlardan daha az hata yapmaktadır.",
            "Kişiselleştirilmiş beslenme uygulamaları obeziteyle mücadelede etkili bir araçtır.",
            "E-sigara geleneksel sigaradan daha az zararlıdır ve bırakma aracı olarak teşvik edilmelidir.",
            "Dijital ilaç hatırlatma sistemleri tedavi uyumunu %25 artırmaktadır.",
            "Akıllı ev sağlık sensörleri yaşlıların evde bağımsız yaşamasını desteklemektedir.",
            "3D baskılı protezler geleneksel protezlere göre %60 daha ucuz ve kişiselleştirilebilirdir.",
            "Yapay zeka destekli psikiyatri teşhisi objektif kriterlerle daha tutarlı sonuçlar üretmektedir.",
        ],
        [
            "JMIR mHealth çalışmasına göre mobil sağlık uygulamaları tansiyon takibinde %30 daha tutarlı sonuç vermektedir.",
            "American Diabetes Association raporuna göre insülin pompası HbA1c seviyelerini %15 düşürmektedir.",
            "Nature Neuroscience araştırmasına göre beyin-bilgisayar arayüzü felçli hastalarda iletişim hızını %80 artırmaktadır.",
            "Radiology dergisine göre yapay zeka mamografi okumalarında radyologlara göre %20 daha az yanlış negatif sonuç vermektedir.",
            "Nutrition Journal çalışmasına göre kişiselleştirilmiş beslenme planları kilo verme başarısını %40 artırmaktadır.",
            "Cochrane Review analizine göre e-sigara kullanıcılarının %18'i geleneksel sigarayı bırakmaktadır.",
            "JAMA Internal Medicine araştırmasına göre dijital hatırlatma sistemleri ilaç uyumunu %25 artırmaktadır.",
            "Aging in Place Institute raporuna göre akıllı ev sensörleri yaşlı düşme kazalarını %35 azaltmaktadır.",
            "Prosthetics and Orthotics International çalışmasına göre 3D protez üretim maliyeti geleneksel protezden %60 düşüktür.",
            "Psychiatry Research dergisine göre yapay zeka teşhisi klinisyenler arası tutarlılığı %45 artırmaktadır.",
        ],
        [
            "BMJ raporuna göre mobil sağlık uygulamaları veri güvenliği açısından hastane sistemlerinden %50 daha savunmasızdır.",
            "Endocrine Society uyarısına göre insülin pompası arızaları ciddi hipoglisemi krizlerine yol açabilmektedir.",
            "Neuroethics dergisine göre beyin-bilgisayar arayüzü verileri zihinsel mahremiyeti tehdit etmektedir.",
            "Health Affairs analizine göre yapay zeka radyoloji okumaları nadir hastalıklarda %30 hata yapmaktadır.",
            "Eating Disorders Journal çalışmasına göre beslenme uygulamaları yeme bozukluklarını %15 tetikleyebilmektedir.",
            "Lancet Respiratory Medicine raporuna göre e-sigara akciğer hasarını geleneksel sigaraya benzer şekilde artırmaktadır.",
            "JAMIA araştırmasına göre hatırlatma uygulamaları aşırı bildirimle hasta kaygısını %20 artırmaktadır.",
            "APTA raporuna göre akıllı ev sensörleri yaşlıların mahremiyet algısını %40 bozmaktadır.",
            "Journal of Rehabilitation Medicine çalışmasına göre 3D protezler dayanıklılık testlerinde geleneksel protezden %25 geride kalmaktadır.",
            "World Psychiatry dergisine göre yapay zeka psikiyatri teşhisinde kültürel bağlamı %35 göz ardı etmektedir.",
        ],
    ),
    "egitim_sistemi": (
        [
            "Çift dil eğitimi çocukların bilişsel gelişimini hızlandırmakta ve akademik başarıyı artırmaktadır.",
            "Bireyselleştirilmiş öğrenme planları sınıf ortamında verimliliği %35 artırmaktadır.",
            "Öğretmen değerlendirme sistemleri öğrenci geri bildirimine dayalı olmalı ve yılda en az iki kez yapılmalıdır.",
            "Dijital okuryazarlık eğitimi ilkokul müfredatının zorunlu bir parçası olmalıdır.",
            "Kodlama eğitimi ilkokuldan itibaren başlamalı ve haftada en az 3 saat ayrılmalıdır.",
            "Öğrenci değişim programları kültürel anlayışı ve empati yeteneğini geliştirmektedir.",
            "Veli katılımı öğrenci akademik başarısını doğrudan etkilemekte ve %20 artırmaktadır.",
            "Sınıf mevcudu 15 öğrenciyi geçmemeli ve bireysel ilgi artırılmalıdır.",
            "Yabancı dil eğitimi 3 yaşında başlamalı ve günlük minimum 1 saat sürmelidir.",
            "Eğitimde yapay zeka destekli öğrenme analitiği öğrenci zorluklarını erken tespit etmektedir.",
        ],
        [
            "American Psychological Association çalışmasına göre çift dil eğitimi çocuklarda yürütücü işlevleri %20 geliştirmektedir.",
            "Gates Foundation raporuna göre bireyselleştirilmiş öğrenme planları başarısızlık oranını %30 azaltmaktadır.",
            "Harvard Kennedy School analizine göre öğrenci geri bildirimli değerlendirme öğretmen performansını %15 artırmaktadır.",
            "European Commission verilerine göre dijital okuryazarlık programları istihdam oranını %18 artırmaktadır.",
            "Code.org araştırmasına göre erken kodlama eğitimi problem çözme becerilerini %25 geliştirmektedir.",
            "Journal of Research in International Education çalışmasına göre değişim programları kültürel empati skorlarını %40 artırmaktadır.",
            "National Center for Family Literacy raporuna göre veli katılımı okul başarısını %20 artırmaktadır.",
            "Tennessee STAR projesine göre 15 kişilik sınıflar öğrenci başarısını %15 artırmıştır.",
            "Cambridge Assessment çalışmasına göre 3 yaşında başlayan dil eğitimi akıcılık seviyesini %50 artırmaktadır.",
            "EdTech Magazine raporuna göre öğrenme analitiği öğrenci riskini %35 erken tespit edebilmektedir.",
        ],
        [
            "Education Week haberine göre çift dil programları kaynakları tek dile göre %40 daha fazla tüketmektedir.",
            "Brookings Enstitüsü analizine göre bireyselleştirilmiş öğrenme öğretmen hazırlık süresini %50 artırmaktadır.",
            "The Atlantic makalesine göre öğrenci değerlendirmeleri öğretmen kaygısını %25 artırmaktadır.",
            "UNESCO raporuna göre dijital okuryazarlık eğitimi gelişmekte olan ülkelerde altyapı maliyetini %300 artırmaktadır.",
            "Pew Research Center anketi sonuçlarına göre zorunlu kodlama müfredatı velilerin %45'inin desteğini almamaktadır.",
            "Journal of Studies in International Education çalışmasına göre değişim programları akademik takvimi %20 geriletmemektedir.",
            "Sociology of Education dergisine göre aşırı veli katılımı öğrenci özerkliğini %15 azaltmaktadır.",
            "Manhattan Institute raporuna göre küçük sınıflar öğretmen ihtiyacını %35 artırarak maliyetleri yükseltmektedir.",
            "Linguistic Society of America çalışmasına göre 3 yaşında dil eğitimi anadil gelişimini %10 yavaşlatmaktadır.",
            "Privacy International raporuna göre öğrenme analitiği öğrenci verilerini %80 oranında toplamaktadır.",
        ],
    ),
    "iklim_degisikligi": (
        [
            "Yeşil binalar konut enerji tüketimini %40 azaltarak sürdürülebilir kentleşmenin anahtarıdır.",
            "Atık yönetiminde dairesel ekonomi prensipleri işletme maliyetlerini %30 düşürmektedir.",
            "Biyoçeşitlilik koridorları kentsel alanlarda ekosistem sürekliliğini sağlamaktadır.",
            "Deniz seviyesi yükselmesine karşı dalgakıranlar yerine doğal bariyerler tercih edilmelidir.",
            "Türbin teknolojisindeki gelişmeler rüzgar enerjisini fosil yakıtlardan daha ucuz hale getirmiştir.",
            "Kar topu ısıtma sistemleri kış sporları turizmini iklim değişikliğine rağmen sürdürülebilir kılmaktadır.",
            "Kentsel tarım şehirlerde gıda güvenliğini artırarak taşıma emisyonlarını azaltmaktadır.",
            "Biyoaktif çatılar ısı adası etkisini %25 azaltarak enerji tüketimini düşürmektedir.",
            "Yenilenebilir enerji depolama çözümleri enerji arz güvenliğini artırmaktadır.",
            "Karbon ofset piyasaları şirketlerin sera gazı emisyonlarını dengelemesinde etkili bir araçtır.",
        ],
        [
            "World Green Building Council raporuna göre yeşil binalar enerji kullanımını %40 azaltmaktadır.",
            "Ellen MacArthur Foundation analizine göre dairesel ekonomi Avrupa'da yılda 1,8 trilyon euro tasarruf sağlayabilmektedir.",
            "Conservation Biology dergisine göre biyoçeşitlilik koridorları tür kaybını %20 azaltmaktadır.",
            "Nature Climate Change çalışmasına göre mangrov bariyerleri dalga enerjisini %66 azaltmaktadır.",
            "Lazard raporuna göre rüzgar enerjisi maliyeti kömüre göre %70 daha düşüktür.",
            "Snowsports Industries America verilerine göre kar topu sistemleri kayak sezonunu %30 uzatmaktadır.",
            "FAO raporuna göre kentsel tarım şehirlerde gıda talebinin %15'ini karşılayabilmektedir.",
            "Journal of Environmental Management çalışmasına göre yeşil çatılar bina enerji ihtiyacını %25 azaltmaktadır.",
            "BloombergNEF verilerine göre lityum batarya maliyeti son 10 yılda %90 düşmüştür.",
            "Stanford Woods Institute raporuna göre karbon ofset projelerı orman koruma alanlarını %18 artırmaktadır.",
        ],
        [
            "Reuters haberine göre yeşil bina sertifikası maliyeti inşaat bütçesini %15 artırmaktadır.",
            "Waste Dive raporuna göre dairesel ekonomi uygulamaları işletme karmaşıklığını %40 artırmaktadır.",
            "Urban Ecosystems dergisine göre koridorlar kentsel gelişimi %10 kısıtlamaktadır.",
            "Coastal Management raporuna göre doğal bariyer bakımı yapay barajlara göre %50 daha pahalıdır.",
            "Bird Life International çalışmasına göre rüzgar türbinleri yılda 600 bin kuş ölümüne yol açmaktadır.",
            "Guardian haberine göre kar topu sistemleri litre başına 5 litre su tüketerek kuraklığı artırmaktadır.",
            "Journal of Cleaner Production çalışmasına göre kentsel tarım pestisit kullanımını %20 artırmaktadır.",
            "Building and Environment dergisine göre yeşil çatı bakım maliyeti standart çatıya göre 3 kat daha yüksektir.",
            "Mineral Commodity Summaries raporuna göre lityum talebi 2030 yılına kadar %500 artacaktır.",
            "Carbon Market Watch raporuna göre ofset projelerinin %80'i gerçek emisyon azaltımı sağlamamaktadır.",
        ],
    ),
    "sosyal_medya": (
        [
            "Sosyal medya platformları siyasi kampanyalarda şeffaf reklam politikaları uygulamalıdır.",
            "Kullanıcı oluşturulmuş içerik gazeteciliğin yerini alarak haber üretimini demokratikleştirmektedir.",
            "Dijital minimalizm hareketi zihinsel sağlığı korumakta ve ekran bağımlılığını azaltmaktadır.",
            "Sosyal ticaret platformları geleneksel perakendeyi dönüştürerek küçük üreticilere pazar erişimi sunmaktadır.",
            "Podcast formatı derinlemesine bilgi tüketimini destekleyerek kamuoyunu bilinçlendirmektedir.",
            "Dijital ayak izi hesaplayıcıları çevre bilincini artırmakta ve sürdürülebilir davranışları teşvik etmektedir.",
            "Sosyal medya okuryazarlığı eğitimi dezenformasyona karşı direnci artırmaktadır.",
            "Canlı yayın platformları acil durum müdahalelerinde anlık bilgi akışı sağlamaktadır.",
            "Dijital sanat NFT'leri sanatçıların eserlerinden sürekli gelir elde etmesini sağlamaktadır.",
            "Mikro ödeme sistemleri içerik üreticilerini doğrudan destekleyerek bağımsız gazeteciliği güçlendirmektedir.",
        ],
        [
            "Mozilla Foundation raporuna göre şeffaf reklam politikaları yanıltıcı içerikleri %35 azaltmaktadır.",
            "Reuters Institute verilerine göre kullanıcı içerikleri haber akışının %40'ını oluşturmaktadır.",
            "Journal of Social and Clinical Psychology çalışmasına göre dijital minimalizm depresyon semptomlarını %25 azaltmaktadır.",
            "Shopify analizine göre sosyal ticaret küçük işletmelerin satışlarını %50 artırmaktadır.",
            "Edison Research raporuna göre podcast dinleyicilerinin %74'ü yeni konularda bilgi edinmektedir.",
            "Journal of Environmental Psychology çalışmasına göre dijital ayak izi takibi sürdürülebilir davranışları %15 artırmaktadır.",
            "Stanford History Education Group araştırmasına göre okuryazarlık eğitimi dezenformasyonu %30 azaltmaktadır.",
            "Pew Research Center verilerine göre canlı yayın platformları acil durumlarda ilk bilgi kaynağı olmaktadır.",
            "Artnome analizine göre NFT satışları sanatçı gelirlerini %40 artırmaktadır.",
            "Nieman Lab raporuna göre mikro ödeme sistemleri bağımsız gazetecilik gelirini %25 artırmaktadır.",
        ],
        [
            "Campaign Legal Center raporuna göre şeffaf reklam yönetmelikleri yalnızca %60 oranında uygulanmaktadır.",
            "Columbia Journalism Review haberine göre kullanıcı içerikleri doğruluk kontrolünden %80 geçmemektedir.",
            "Addiction Research & Theory dergisine göre dijital minimalizm sosyal izolasyona yol açabilmektedir.",
            "Financial Times analizine göre sosyal ticaret platformları küçük işletmelere %25 komisyon uygulamaktadır.",
            "The Atlantic makalesine göre podcast formatı yanlış bilginin %30 daha derinlemesine işlemesine olanak tanımaktadır.",
            "Sustainability dergisine göre dijital ayak izi hesaplayıcıları gerçek emisyonları %40 hatalı tahmin etmektedir.",
            "Science Advances çalışmasına göre okuryazarlık eğitimi etkisini 6 ay içinde %50 kaybetmektedir.",
            "Disinformation Review raporuna göre canlı yayın platformları yanıltıcı acil durum bilgilerini %20 daha hızlı yaymaktadır.",
            "The Guardian haberine göre NFT piyasası çevre kirliliğine yol açan enerji tüketimiyle eleştirilmektedir.",
            "Journalism Practice dergisine göre mikro ödeme modelleri okuyucu kitlesini %30 daraltmaktadır.",
        ],
    ),
    "gida_teknolojisi": (
        [
            "Hücre bazlı et teknolojisi hayvan refahını korurken protein ihtiyacını karşılayabilmektedir.",
            "Akıllı sulama sistemleri tarımda su tasarrufunu %40 artırarak kuraklıkla mücadele etmektedir.",
            "Gıda ambalajlarında aktif koruma teknolojisi atığı %25 azaltmaktadır.",
            "Dron destekli tarım ilaçlama verimliliğini %30 artırarak çevre kirliliğini azaltmaktadır.",
            "Fermente gıda trendi bağırsak sağlığını destekleyerek bağışıklık sistemini güçlendirmektedir.",
            "Yapay zeka destekli hasat tahmini tarımsal planlamayı optimize ederek kayıpları azaltmaktadır.",
            "Biyoaktif ambalajlar gıda ömrünü %50 uzatarak israfı önlemektedir.",
            "Dikey balık çiftlikleri deniz ekosistemlerini koruyarak sürdürülebilir protein üretmektedir.",
            "Blok zinciri gıda takip sistemleri tedarik zinciri şeffaflığını artırmaktadır.",
            "Alg bazlı protein alternatifleri karbon emisyonunu negatif hale getirebilmektedir.",
        ],
        [
            "Nature Food çalışmasına göre hücre bazlı et su ayak izini hayvancılığa göre %96 azaltmaktadır.",
            "Agricultural Water Management dergisine göre akıllı sulama su kullanımını %40 azaltmaktadır.",
            "Packaging Technology and Science çalışmasına göre aktif ambalaj gıda israfını %25 azaltmaktadır.",
            "Precision Agriculture raporuna göre dron ilaçlama kimyasal kullanımını %30 azaltmaktadır.",
            "Nutrients dergisine göre fermente gıda tüketimi bağırsak mikrobiyom çeşitliliğini %20 artırmaktadır.",
            "Computers and Electronics in Agriculture çalışmasına göre yapay zeka hasat tahmini kayıpları %15 azaltmaktadır.",
            "Food Control dergisine göre biyoaktif ambalaj meyve ömrünü %50 uzatmaktadır.",
            "Aquaculture dergisine göre dikey balık çiftliği deniz kirliliğini %70 azaltmaktadır.",
            "IBM Food Trust verilerine göre blok zinciri gıda kaynaklama süresini %80 kısaltmaktadır.",
            "Algal Research dergisine göre alg yetiştiriciliği hektar başına 10 ton CO2 tutabilmektedir.",
        ],
        [
            "Consumer Reports raporuna göre hücre bazlı et fiyatları geleneksel etten %500 daha pahalıdır.",
            "Environmental Science & Technology çalışmasına göre akıllı sulama cihazları üretimi elektronik atık oluşturmaktadır.",
            "Waste Management dergisine göre aktif ambalaj geri dönüşüm sürecini %30 zorlaştırmaktadır.",
            "Journal of Economic Entomology çalışmasına göre dron ilaçlamada hedef dışı bölgelere sızma %15'tir.",
            "Clinical Gastroenterology raporuna göre fermente gıda tüketimi mide asidini %10 artırabilmektedir.",
            "Agronomy dergisine göre yapay zeka tahmini hava değişkenliklerinde %20 hata payına sahiptir.",
            "Food Packaging and Shelf Life çalışmasına göre biyoaktif ambalaj maliyeti standart ambalaja göre 4 kat daha yüksektir.",
            "Marine Policy dergisine göre dikey çiftlik balığı deniz balığından %40 daha az omega-3 içermektedir.",
            "Journal of Business Ethics raporuna göre blok zinciri enerji tüketimi geleneksel takibe göre %300 fazladır.",
            "Trends in Food Science & Technology çalışmasına göre alg protein tadı tüketici kabulünü %35 azaltmaktadır.",
        ],
    ),
}

all_samples = []
for topic, (claims, pro_evs, con_evs) in more_data.items():
    for i in range(min(len(claims), len(pro_evs), len(con_evs))):
        sample = make_sample(claims[i], [pro_evs[i]], [con_evs[i]], topic)
        all_samples.append(sample)
    # Add 2+2 versions
    for i in range(min(5, len(claims), len(pro_evs)-1, len(con_evs)-1)):
        idx = (i + 5) % len(claims)
        idx2 = (i + 6) % len(pro_evs)
        sample = make_sample(claims[idx], [pro_evs[idx], pro_evs[idx2]], [con_evs[idx], con_evs[idx2]], topic)
        all_samples.append(sample)

print(f"Additional samples: {len(all_samples)}")

# Load current dataset
with open("data/argument_dataset.json") as f:
    current = json.load(f)

current.extend(all_samples)
random.shuffle(current)
for i, item in enumerate(current):
    item["id"] = i + 1

with open("data/argument_dataset.json", "w", encoding="utf-8") as f:
    json.dump(current, f, ensure_ascii=False, indent=2)

print(f"Total dataset: {len(current)} samples")

# Verify
bad = 0
for item in current:
    d = item["data"]
    text = d["text"]
    for c in d.get("claim", []):
        if text[c["start"]:c["end"]] != c["text"]:
            bad += 1
    for e in d.get("Evidence", []):
        if text[e["start"]:e["end"]] != e["text"]:
            bad += 1
print(f"Offset mismatches: {bad}")
