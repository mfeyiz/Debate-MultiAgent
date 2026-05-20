#!/usr/bin/env python3
"""Third expansion: more topics and variations."""
import json
import random
random.seed(42)

def make_sample(claim, pro, con, topic):
    text = claim + " " + " ".join(pro + con)
    text = text.strip()
    claims = [{"id": "c1", "text": claim, "label": "claim",
              "start": text.find(claim), "end": text.find(claim) + len(claim)}]
    evidences = []
    for i, ev in enumerate(pro + con):
        s = text.find(ev)
        evidences.append({"id": f"e{i+1}", "text": ev, "label": "evidence",
                          "start": s, "end": s + len(ev)})
    support = [{"from": e["id"], "to": "c1"} for e in evidences[:len(pro)]]
    attack = [{"from": e["id"], "to": "c1"} for e in evidences[len(pro):]]
    return {"data": {"text": text, "claim": claims, "Evidence": evidences,
                   "support": support, "attack": attack, "topic": topic, "type": "opinion_forum"}}

# More diverse topics
extra = {
    "nukleer_enerji": (
        [
            "Küçük modüler reaktörler güvenli ve ekonomik nükleer enerji çözümü sunmaktadır.",
            "Nükleer atık depolama teknolojileri uzun vadede çevre güvenliği sağlamaktadır.",
            "Nükleer enerji fosil yakıtlara bağımlılığı azaltarak enerji bağımsızlığını artırmaktadır.",
            "Füzyon enerjisi araştırmaları temiz enerji geleceğinin anahtarıdır.",
            "Nükleer enerji santralleri rüzgar ve güneşten daha güvenilir baz yük enerjisi sağlamaktadır.",
            "Radyasyon tedavisi kanser vakalarında kemoterapiden daha etkilidir.",
            "Nükleer denizaltılar uzun süreli sualtı operasyonlarında stratejik üstünlük sağlamaktadır.",
            "Radyoizotoplar tıbbi görüntülemede teşhis doğruluğunu %40 artırmaktadır.",
            "Nükleer enerji istihdamı bölgesel ekonomilere istikrarlı gelir kaynağı oluşturmaktadır.",
            "Üçüncü nesil reaktörler soğutma sistemi arızalarında otomatik kapanma özelliğine sahiptir.",
        ],
        [
            "International Atomic Energy Agency raporuna göre küçük modüler reaktörler inşaat maliyetini %50 azaltmaktadır.",
            "Science çalışmasına göre derin jeolojik depolama atık güvenliğini 100 bin yıl garanti edebilmektedir.",
            "World Nuclear Association verilerine göre nükleer enerji ülkelerin enerji ithalatını %30 azaltmaktadır.",
            "MIT Plasma Science raporuna göre füzyon reaktörleri sera gazı salımı yapmadan enerji üretebilmektedir.",
            "Nuclear Energy Institute verilerine göre nükleer santraller kapasite faktöründe %93 verimlilik sağlamaktadır.",
            "Lancet Oncology çalışmasına göre radyoterapi lokal kanser kontrolünü %80 sağlayabilmektedir.",
            "Naval Technology raporuna göre nükleer denizaltılar 20 yıl boyunca yakıt ikmaline gerek duymamaktadır.",
            "Journal of Nuclear Medicine araştırmasına göre PET taramaları teşhis doğruluğunu %40 artırmaktadır.",
            "Department of Energy raporuna göre nükleer santraller binlerce yüksek ücretli istihdam yaratmaktadır.",
            "Nuclear Regulatory Commission belgelerine göre üçüncü nesil reaktörler pasif güvenlik sistemleriyle donatılmıştır.",
        ],
        [
            "Bulletin of the Atomic Scientists raporuna göre küçük modüler reaktörler henüz ticari olarak kanıtlanmamıştır.",
            "Greenpeace analizine göre nükleer atık depolama alanları sızıntı riski taşımaktadır.",
            "Renewable Energy Policy Network raporuna göre nükleer enerji inşaat maliyeti yenilenebilire göre 3 kat daha yüksektir.",
            "Nature Physics çalışmasına göre füzyon enerjisi ticari uygulamaya geçişi en az 30 yıl uzaktadır.",
            "Union of Concerned Scientists raporuna göre nükleer santraller sıcak su salımıyla deniz ekosistemlerini %20 bozmaktadır.",
            "Cancer Research UK verilerine göre radyoterapi yan etkileri hastaların %60'ında görülmektedir.",
            "The Guardian haberine göre nükleer denizaltı kazaları okyanuslara radyoaktif madde salmaktadır.",
            "Health Physics dergisine göre radyoizotop üretimi nükleer yayılma riskini artırmaktadır.",
            "Economic Policy Institute raporuna göre nükleer santral kapanışı bölgesel istihdamı %40 azaltmaktadır.",
            "Nuclear Monitor çalışmasına göre üçüncü nesil reaktörlerin maliyeti bütçe tahminlerinin %200 üzerindedir.",
        ],
    ),
    "uzay_kesfi": (
        [
            "Özel uzay şirketleri uzay turizmini ekonomik olarak sürdürülebilir hale getirmektedir.",
            "Mars kolonizasyonu insan türünün uzun vadeli hayatta kalması için zorunludur.",
            "Uydu internet projeleri kırsal alanlarda dijital eşitsizliği ortadan kaldırmaktadır.",
            "Asteroid madenciliği Dünya'nın metal kaynaklarını tüketme baskısını azaltmaktadır.",
            "Uluslararası Uzay İstasyonu bilimsel iş birliğinin en önemli örneğidir.",
            "Ay tabanı istasyonu derin uzay görevleri için stratejik sıçrama tahtasıdır.",
            "Uzay çöplüğü aktif temizlik operasyonları gerektiren acil bir tehdittir.",
            "Roket yeniden kullanılabilirliği uzay erişim maliyetini %70 azaltmaktadır.",
            "Uzay tabanlı güneş enerjisi Dünya'nın enerji ihtiyacını karşılayabilmektedir.",
            "Kütle çekim dalga dedektörleri evrenin doğasını anlamada çığır açmaktadır.",
        ],
        [
            "Virgin Galactic raporuna göre uzay turizmi pazarı 2030 yılına kadar 8 milyar dolara ulaşacaktır.",
            "SpaceX verilerine göre Starship sistemleri Mars yolculuğu maliyetini kişi başı 100 bin dolara indirebilecektir.",
            "Starlink verilerine göre uydu internet 40 ülkede 2 milyondan fazla aboneye hizmet vermektedir.",
            "Asteroid Mining Corporation analizine göre tek bir platinyum asteroidi 50 trilyon dolar değere sahiptir.",
            "NASA raporuna göre ISS üzerinde 3 binden fazla bilimsel deney gerçekleştirilmiştir.",
            "Artemis Program raporuna göre Ay tabanı 2028 yılına kadar sürekli insan varlığı hedeflemektedir.",
            "ESA Space Debris Office verilerine göre yörüngede 36 binden fazla parça uzay çöpü izlenmektedir.",
            "SpaceX verilerine göre Falcon 9 roketleri 200'den fazla kez yeniden kullanılmıştır.",
            "Science Advances çalışmasına göre uzay güneş enerjisi potansiyeli Dünya'nın mevcut tüketiminin 100 katıdır.",
            "LIGO Scientific Collaboration raporuna göre kütle çekim dalga gözlemleri evreni yeni bir pencereden izlemektedir.",
        ],
        [
            "Gizmodo haberine göre uzay turizmi roket yakıtı emisyonlarını uçuş başına 100 ton artırmaktadır.",
            "Nature Astronomy çalışmasına göre Mars radyasyonu kolonistleri kanser riskiyle karşı karşıya bırakmaktadır.",
            "Astronomy Magazine raporuna göre uydu kümesi astronomi gözlemlerini %40 bozmaktadır.",
            "Science Policy dergisine göre asteroid madenciliği hukuki çerçevesi uluslararası anlaşmazlık riski taşımaktadır.",
            "The Guardian analizine göre ISS bakım maliyeti yılda 3 milyar dolara ulaşmaktadır.",
            "Planetary Society raporuna göre Ay kolonizasyonu maliyeti en az 100 milyar dolar gerektirmektedir.",
            "Space.com haberine göre uzay çöpü temizliği maliyeti parça başına 100 milyon doları aşabilmektedir.",
            "Journal of Spacecraft and Rockets çalışmasına göre roket yeniden kullanımı bakım maliyetini %40 artırmaktadır.",
            "Energy Policy dergisine göre uzay güneş enerjisi iletim maliyeti Dünya'dan %500 daha yüksektir.",
            "Physical Review Letters çalışmasına göre kütle çekim dalga dedektörleri yılda sadece birkaç olay kaydetmektedir.",
        ],
    ),
    "genetik_muhendisligi": (
        [
            "CRISPR teknolojisi kalıtsal hastalıkları tedavi etmede devrim yaratmaktadır.",
            "Genetik olarak modifiye edilmiş mikroorganizmalar çevre temizliğinde etkili araçtır.",
            "Fitaları genetik olarak optimize etmek gıda güvenliğini ve verimliliğini artırmaktadır.",
            "Genetik veri tabanları kişiselleştirilmiş tıbbın gelişimine olanak tanımaktadır.",
            "Sentetik biyoloji yeni biyolojik fonksiyonlar tasarlayarak endüstriyel üretimi dönüştürmektedir.",
            "Genetik tarama testleri gebelik öncesi sağlık planlamasında kritik bir rol oynamaktadır.",
            "Biyobrick standardı sentetik biyolojiyi modüler ve öngörülebilir hale getirmektedir.",
            "Genetik mühendisliği laboratuvar hayvanlarının kullanımını azaltmaktadır.",
            "Epigenetik araştırmalar çevresel faktörlerin gen ifadesini nasıl etkilediğini ortaya koymaktadır.",
            "Genetik kütüphaneler ilaç geliştirme sürecini yıllardan aylara indirmektedir.",
        ],
        [
            "New England Journal of Medicine çalışmasına göre CRISPR orak hücre anemisinde %93 tedavi başarısı sağlamıştır.",
            "Nature Biotechnology raporuna göre genetik mikroorganizmalar petrol kirliliğini %80 azaltabilmektedir.",
            "Nature Plants dergisine göre genetik optimize bitkiler verimini %30 artırarak su kullanımını %25 azaltmaktadır.",
            "Nature Genetics çalışmasına göre genetik veri tabanları hastalık riski tahminini %45 iyileştirmektedir.",
            "Science çalışmasına göre sentetik biyoloji biyoyakıt üretimini %50 artırmaktadır.",
            "JAMA raporuna göre genetik tarama testleri gebelik komplikasyonlarını %35 önceden tespit edebilmektedir.",
            "ACS Synthetic Biology dergisine göre biyobrick standardı biyolojik devre tasarımını %40 hızlandırmaktadır.",
            "Alternatives to Laboratory Animals çalışmasına göre genetik modellemeler hayvan deneylerini %60 azaltmaktadır.",
            "Frontiers in Genetics araştırmasına göre epigenetik veriler çevre maruziyetini %70 doğrulukla tahmin etmektedir.",
            "Drug Discovery Today raporuna göre genetik kütüphaneler ilaç keşif süresini %40 kısaltmaktadır.",
        ],
        [
            "Nature Medicine çalışmasına göre CRISPR düzenlemelerinde istenmeyen mutasyon riski %10'dur.",
            "Environmental Science & Technology raporuna göre genetik mikroorganizmalar ekosistem dengesini bozabilmektedir.",
            "Food and Water Watch analizine göre genetik bitki tohumları çiftçilerin %75'ini bağımlı hale getirmektedir.",
            "American Journal of Human Genetics çalışmasına göre genetik veri ihlalleri 5 milyondan fazla kişiyi etkilemiştir.",
            "Ethics in Biology dergisine göre sentetik organizmalar biyo-güvenlik riski taşımaktadır.",
            "The Hastings Center raporuna göre genetik tarama ayrımcı uygulamalara zemin hazırlayabilmektedir.",
            "PLOS Biology çalışmasına göre biyobrick devreleri öngörülemeyen davranışlar sergileyebilmektedir.",
            "Alternatives to Laboratory Animals çalışmasına göre genetik modeller hayvan refahını %100 ortadan kaldırmamaktadır.",
            "Epigenetics & Chromatin dergisine göre epigenetik değişikliklerin kalıcılığı %30 belirsizdir.",
            "Nature Reviews Drug Discovery raporuna göre genetik kütüphane araştırmaları etik sınırları zorlamaktadır.",
        ],
    ),
    "kripto_para": (
        [
            "Merkeziyetsiz finans sistemleri geleneksel bankacılığın dışlanmış kesimlerine erişim sağlamaktadır.",
            "Blockchain teknolojisi tedarik zinciri şeffaflığını artırarak sahteciliği azaltmaktadır.",
            "Stablecoin'ler uluslararası para transferlerini hızlandırmakta ve maliyetleri düşürmektedir.",
            "Akıllı sözleşmeler noter ve aracı kurum maliyetlerini ortadan kaldırmaktadır.",
            "Kripto para madenciliği yenilenebilir enerji kullanımını teşvik etmektedir.",
            "Tokenizasyon gayrimenkul gibi geleneksel varlıklara parçalı sahiplik olanağı tanımaktadır.",
            "Kripto cüzdanları bireylerin finansal özerkliğini artırmaktadır.",
            "DAO'lar merkeziyetsiz organizasyon yönetiminde yeni bir model sunmaktadır.",
            "NFT'ler dijital sanatın mülkiyet ve özgünlük sorununu çözmektedir.",
            "Kripto para bağış sistemleri insani yardım kuruluşlarına şeffaf fon akışı sağlamaktadır.",
        ],
        [
            "World Bank raporuna göre DeFi platformları 2 milyardan fazla bankasız kişiye potansiyel erişim sunmaktadır.",
            "IBM Blockchain çalışmasına göre tedarik zinciri blockchain uygulamaları sahteciliği %70 azaltmaktadır.",
            "Chainalysis raporuna göre stablecoin transferleri geleneksel havalelere göre %90 daha hızlı gerçekleşmektedir.",
            "Harvard Business Review analizine göre akıllı sözleşmeler aracı kurum maliyetlerini %60 azaltmaktadır.",
            "Cambridge Bitcoin Electricity Consumption Index verilerine göre madenciliğin %40'ı yenilenebilir enerjiden sağlanmaktadır.",
            "Deloitte raporuna göre tokenizasyon pazarı 2030 yılına kadar 16 trilyon dolara ulaşacaktır.",
            "CoinMetrics verilerine göre kripto cüzdan sayısı 200 milyonu aşarak finansal erişimi artırmaktadır.",
            "MIT Sloan Management Review çalışmasına göre DAO'lar organizasyonel şeffaflığı %45 artırmaktadır.",
            "NonFungible.com raporuna göre NFT pazarı 2024 yılında 25 milyar dolar hacme ulaşmıştır.",
            "UNICEF raporuna göre kripto bağış sistemleri yardım fonlarının %95'inin hedefe ulaşmasını sağlamaktadır.",
        ],
        [
            "Chainalysis raporuna göre kripto para dolandırıcılığı 2024 yılında 5 milyar doları aşmıştır.",
            "Financial Times analizine göre blockchain enerji tüketimi Yunanistan'ın toplam tüketimine eşittir.",
            "Bloomberg haberine göre stablecoin'lerin %30'u yeterli rezervle desteklenmemektedir.",
            "Reuters araştırmasına göre akıllı sözleşme açıkları 3 milyar dolarlık kayba yol açmıştır.",
            "Nature Sustainability çalışmasına göre Bitcoin madenciliği yılda 65 megaton CO2 salmaktadır.",
            "SEC raporuna göre tokenizasyon düzenlemeleri yatırımcı korumasını %40 zayıflatmaktadır.",
            "Journal of Financial Crime dergisine göre kripto cüzdanları kara para aklama için kullanılabilmektedir.",
            "The Guardian haberine göre DAO'lar yasal statüsü belirsizliği nedeniyle çıkmaza girebilmektedir.",
            "Artforum analizine göre NFT pazarı spekülasyona dayalı olarak %80 değer kaybetmiştir.",
            "Oxfam raporuna göre kripto bağışları kuruluşların operasyonel kapasitesini %25 zorlaştırmaktadır.",
        ],
    ),
    "kentsel_donusum": (
        [
            "Tarihi binaların adaptif yeniden kullanımı kentsel kimliği korurken modern ihtiyaçları karşılamaktadır.",
            "Kentsel yeşil alanlar şehirlerde sıcaklık adası etkisini önemli ölçüde azaltmaktadır.",
            "Yüksek yoğunluklu konut projeleri toplu taşıma kullanımını artırarak trafik yoğunluğunu azaltmaktadır.",
            "Gecekondu alanlarının dönüşümü güvenli ve sağlıklı konut stokunu artırmaktadır.",
            "Kentsel dönüşüm projeleri mülk sahiplerinin rızasıyla yürütülmeli ve zorla tahliye uygulanmamalıdır.",
            "Akıllı sokak aydınlatması enerji tüketimini azaltırken kamu güvenliğini artırmaktadır.",
            "Yaya öncelikli kent planlaması sosyal etkileşimi artırarak toplumsal bağları güçlendirmektedir.",
            "Kentsel tarım projeleri gıda güvenliğine katkı sağlamakta ve toplulukları birleştirmektedir.",
            "Eski endüstriyel alanların yeniden işlevlendirilmesi kentsel ekonomiye canlılık kazandırmaktadır.",
            "Kamusal sanat projeleri kent estetiğini zenginleştirerek vatandaş memnuniyetini artırmaktadır.",
        ],
        [
            "UNESCO raporuna göre adaptif yeniden kullanım bina ömrünü %40 uzatmaktadır.",
            "Journal of Environmental Management çalışmasına göre kentsel yeşil alanlar sıcaklığı %4 azaltmaktadır.",
            "Journal of Transport Geography araştırmasına göre yoğun konut toplu taşıma kullanımını %30 artırmaktadır.",
            "World Bank raporuna göre gecekondu dönüşümü 1 milyondan fazla kişiyi güvenli konuta kavuşturmuştur.",
            "Urban Studies dergisine göre gönüllü kentsel dönüşüm mülk değerini %25 artırmaktadır.",
            "International Journal of Sustainable Lighting çalışmasına göre akıllı aydınlatma enerji tüketimini %60 azaltmaktadır.",
            "Journal of Urban Design raporuna göre yaya öncelikli planlama sosyal etkileşimi %20 artırmaktadır.",
            "Landscape and Urban Planning dergisine göre kentsel tarım gıda güvenliğine %10 katkı sağlayabilmektedir.",
            "Economic Development Quarterly çalışmasına göre endüstriyel dönüşüm istihdamı %35 artırmaktadır.",
            "Cities dergisine göre kamusal sanat projeleri kent memnuniyetini %15 artırmaktadır.",
        ],
        [
            "Journal of Cultural Heritage çalışmasına göre adaptif dönüşüm orijinal yapı özelliklerinin %30'unu kaybettirmektedir.",
            "Habitat International raporuna göre yeşil alan dönüşümü konut stokunu %20 azaltmaktadır.",
            "Transportation Research dergisine göre yoğun konut trafik yoğunluğunu %15 artırabilmektedir.",
            "Forced Migration Review raporuna göre gecekondu dönüşümü binlerce kişiyi zorla yerinden etmektedir.",
            "International Journal of Urban and Regional Research çalışmasına göre zorla tahliye yoksul mahalleleri dağıtmaktadır.",
            "Lighting Research & Technology dergisine göre akıllı aydınlatma maliyeti geleneksel sistemden %200 yüksektir.",
            "Journal of Transport Geography çalışmasına göre yaya bölgeleri ticari geliri %10 azaltabilmektedir.",
            "Urban Forestry & Urban Greening dergisine göre kentsel tarım toprak kirliliği riskini artırmaktadır.",
            "Regional Studies çalışmasına göre endüstriyel dönüşüm eski sanayi istihdamını %50 azaltmaktadır.",
            "Public Art Review raporuna göre kamusal sanat projeleri bütçelerin %5'ini tüketmektedir.",
        ],
    ),
    "su_kaynaklari": (
        [
            "Deniz suyu arıtma teknolojileri kurak bölgelerde içme suyu güvenliğini sağlamaktadır.",
            "Yağmur suyu hasadı şehirlerde sel riskini azaltarak su kaynaklarını çeşitlendirmektedir.",
            "Atık su geri dönüşümü endüstriyel su ihtiyacının %50'sini karşılayabilmektedir.",
            "Su tasarruflu tarım teknolojileri kuraklıkla mücadelede kritik bir rol oynamaktadır.",
            "Buzul suyu kaynaklarının korunması uzun vadeli su güvenliği için zorunludur.",
            "Su fiyatlandırma reformu israfı önlemekte ve sürdürülebilir kullanımı teşvik etmektedir.",
            "Nehir restorasyon projeleri ekosistem hizmetlerini yeniden kazandırmaktadır.",
            "Kentsel permeabil yüzeyler yağmur suyunun toprağa sızmasına olanak tanımaktadır.",
            "Su hakkı anayasal güvence altına alınarak herkesin temiz suya erişimi garanti edilmelidir.",
            "Akıllı su sayaçları kaçakları erken tespit ederek kayıp kaçağı %30 azaltmaktadır.",
        ],
        [
            "International Desalination Association raporuna göre deniz suyu arıtma maliyeti son 10 yılda %50 düşmüştür.",
            "Journal of Hydrology çalışmasına göre yağmur hasadı şehir sel riskini %20 azaltmaktadır.",
            "Water Research dergisine göre atık su geri dönüşüm endüstriyel su ihtiyacının %50'sini karşılayabilmektedir.",
            "FAO raporuna göre damla sulama su kullanımını %40 azaltarak verimi koruyabilmektedir.",
            "Nature Climate Change çalışmasına göre buzulların erimesi 2 milyar kişinin su kaynağını tehdit etmektedir.",
            "Water Policy dergisine göre adil fiyatlandırma su israfını %25 azaltmaktadır.",
            "Ecological Engineering çalışmasına göre nehir restorasyonu biyoçeşitliliği %30 artırmaktadır.",
            "Journal of Sustainable Water çalışmasına göre permeabil yüzeyler yeraltı suyunu %15 beslemektedir.",
            "UNDP raporuna göre su hakkı güvencesi sağlık göstergelerini %20 iyileştirmektedir.",
            "Journal of Water Resources Planning çalışmasına göre akıllı sayaçlar kayıp kaçağı %30 azaltmaktadır.",
        ],
        [
            "Environmental Science & Technology çalışmasına göre arıtma tesisleri deniz ekosistemlerini tuzlu su salımıyla bozmaktadır.",
            "Urban Water Journal raporuna göre yağmur hasadı sistemi bakım maliyeti yılda binlerce dolardır.",
            "Water Environment Research dergisine göre atık su geri dönüşümü enerji tüketimini %70 artırmaktadır.",
            "Agricultural Systems çalışmasına göre damla sulama maliyeti geleneksel sulamaya göre 3 kat daha yüksektir.",
            "Science Advances raporuna göre buzul koruma maliyeti yılda 100 milyar dolara ulaşmaktadır.",
            "World Bank analizine göre su fiyatlandırması düşük gelirli haneleri %15 zorlamaktadır.",
            "River Research and Applications dergisine göre nehir restorasyonu tarım arazilerini %10 azaltmaktadır.",
            "Journal of Environmental Management çalışmasına göre permeabil yüzeyler kış bakım maliyetini %40 artırmaktadır.",
            "Global Environmental Politics raporuna göre su hakkı anayasallaşması yargı yükünü %50 artırmaktadır.",
            "Journal of Cleaner Production çalışmasına göre akıllı sayaçlar veri gizliliği riski taşımaktadır.",
        ],
    ),
    "bio cesitlilik": (
        [
            "Korunan alanlar biyoçeşitlilik kaybını durdurmakta ve türlerin yeniden üremesini desteklemektedir.",
            "Yerli tohum hareketi tarım biyoçeşitliliğini koruyarak gıda güvenliğini artırmaktadır.",
            "Vahşi yaşam koridorları hayvanların göç yollarını güvenli hale getirmektedir.",
            "Mercan restorasyonu deniz ekosistemlerini iklim değişikliğine karşı dirençli kılmaktadır.",
            "Sulak alan korunması kuş göç güzergahlarının sürekliliğini sağlamaktadır.",
            "Ağaçlandırma projeleri karbon tutma kapasitesini artırarak iklim hedeflerine katkı sağlamaktadır.",
            "Biyolojik mücadele tarımda kimyasal kullanımını azaltarak çevre dostu üretim yapmaktadır.",
            "Ekosistem bazlı afet yönetimi sel ve heyelan riskini doğal yollarla azaltmaktadır.",
            "Deniz koruma alanları balık stoklarını toparlayarak sürdürülebilir avcılığa olanak tanımaktadır.",
            "Endemik tür koruma programları bölgesel turizmi destekleyerek ekonomik kalkınmayı sağlamaktadır.",
        ],
        [
            "Science çalışmasına göre korunan alanlar tür kaybını %30 yavaşlatmaktadır.",
            "FAO raporuna göre yerli tohum kullanımı tarım genetik çeşitliliğini %25 artırmaktadır.",
            "Ecological Applications çalışmasına göre vahşi yaşam koridorları hayvan çarpışmalarını %50 azaltmaktadır.",
            "Nature Communications raporuna göre mercan restorasyonu balık popülasyonlarını %40 artırmaktadır.",
            "Wetlands International verilerine göre sulak alanlar kuş göçmenlerinin %80'ine ev sahipliği yapmaktadır.",
            "Nature Climate Change çalışmasına göre ağaçlandırma hektar başına 10 ton karbon tutabilmektedir.",
            "Journal of Applied Ecology araştırmasına göre biyolojik mücadele pestisit kullanımını %60 azaltmaktadır.",
            "Natural Hazards dergisine göre ekosistem bazlı yönetim sel zararlarını %35 azaltmaktadır.",
            "Marine Ecology Progress Series çalışmasına göre deniz koruma alanları balık biyokütlesini %200 artırmaktadır.",
            "Conservation Biology raporuna göre endemik tür turizmi yerel gelirleri %30 artırmaktadır.",
        ],
        [
            "Land Use Policy çalışmasına göre korunan alanlar tarım arazilerini %15 azaltmaktadır.",
            "Seed Science and Technology dergisine göre yerli tohum verimi hibrit tohumlara göre %20 düşüktür.",
            "Biological Conservation raporuna göre koridorlar hayvan-besin kaynakları çatışmasını %10 artırmaktadır.",
            "Coral Reefs dergisine göre mercan restorasyonu başarı oranı sadece %30'tur.",
            "Ambio çalışmasına göre sulak alan koruma tarım sulama suyunu %25 kısıtlamaktadır.",
            "Journal of Arid Environments raporuna göre yanlış ağaçlandırma biyoçeşitliliği %20 azaltmaktadır.",
            "Agricultural Economics dergisine göre biyolojik mücadele hasat kaybını %15 artırmaktadır.",
            "Environmental Management çalışmasına göre doğal afet yönetimi insan müdahalesini geciktirmektedir.",
            "Fish and Fisheries dergisine göre deniz koruma alanları avcı balıkçı gelirini %20 azaltmaktadır.",
            "Tourism Management çalışmasına göre endemik tür turizmi habitat bozulmasını %10 tetiklemektedir.",
        ],
    ),
}

all_samples = []
for topic, (claims, pro_evs, con_evs) in extra.items():
    for i in range(min(len(claims), len(pro_evs), len(con_evs))):
        all_samples.append(make_sample(claims[i], [pro_evs[i]], [con_evs[i]], topic))
    for i in range(min(5, len(claims), len(pro_evs)-1, len(con_evs)-1)):
        idx = (i + 5) % len(claims)
        idx2 = (i + 6) % len(pro_evs)
        all_samples.append(make_sample(claims[idx], [pro_evs[idx], pro_evs[idx2]], [con_evs[idx], con_evs[idx2]], topic))

print(f"Additional samples: {len(all_samples)}")

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
        if text[c["start"]:c["end"]] != c["text"]:
            bad += 1
    for e in d.get("Evidence", []):
        if text[e["start"]:e["end"]] != e["text"]:
            bad += 1
print(f"Offset mismatches: {bad}")
