# ModernBERT Argument Extraction Evaluation Report

This report evaluates the performance of the fine-tuned ModernBERT models in extracting argument structures (components) and relationships (relations) from a 5-round debate on: **"Çocuk yapmak ahlaken doğru bi davranış mı ?"**.

## 1. Overall Performance Metrics

### Component Classification
- **Accuracy**: 82.18%
- **Error Rate**: 17.82%

| Component Type | Count | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| CLAIM | 8 | 18.75% | 37.50% | 25.00% |
| EVIDENCE | 82 | 93.83% | 92.68% | 93.25% |
| OTHER | 11 | 100.00% | 36.36% | 53.33% |

### Relation Classification
- **Accuracy**: 83.33%
- **Error Rate**: 16.67%

| Relation Type | Count | Precision | Recall | F1-Score |
| --- | --- | --- | --- | --- |
| SUPPORT | 22 | 94.44% | 77.27% | 85.00% |
| ATTACK | 13 | 72.22% | 100.00% | 83.87% |
| NONE | 1 | 0.00% | 0.00% | 0.00% |

## 2. Component Classification Details

| ID | Speaker | Component Text | Predicted | Ground Truth | Correct? |
| --- | --- | --- | --- | --- | --- |
| c-1003 | Acceptance Karşıt Browser | Çocuk yapmak ahlaken doğru bir davranıştır | CLAIM | CLAIM | ✅ |
| c-1004 | Acceptance Karşıt Browser | çünkü insan türünün devamlılığını sağlamak ve yeni nesillere yaşam, sevgi ve bilgi aktarımı yapmak temel bir sorumluluktur. | EVIDENCE | EVIDENCE | ✅ |
| c-1005 | Acceptance Karşıt Browser | Felsefi olarak, üreme içgüdüsü doğal bir eğilimdir ve çoğu etik sistem (örneğin, faydacılık) mutluluğu artıran eylemleri destekler; | EVIDENCE | EVIDENCE | ✅ |
| c-1006 | Acceptance Karşıt Browser | bir çocuğun varlığı, ona bakım verenler ve toplum için uzun vadeli faydalar üretebilir. | EVIDENCE | EVIDENCE | ✅ |
| c-1007 | Acceptance Karşıt Browser | Ayrıca, Birleşmiş Milletler İnsan Hakları Evrensel Beyannamesi’nin 16. | OTHER | OTHER | ✅ |
| c-1008 | Acceptance Karşıt Browser | maddesi, aile kurma hakkını temel bir insan hakkı olarak tanır. | CLAIM | EVIDENCE | ❌ |
| c-1009 | Acceptance Karşıt Browser | Bu nedenle, çocuk yapmak, bireysel özgürlük ve toplumsal sürdürülebilirlik açısından ahlaken meşru ve değerlidir. | EVIDENCE | EVIDENCE | ✅ |
| c-1010 | Acceptance Savunucu Browser | Çocuk yapmak ahlaken doğru bir davranıştır; | CLAIM | CLAIM | ✅ |
| c-1011 | Acceptance Savunucu Browser | çünkü bu eylem, hem bireysel özgürlüğün hem de toplumsal sürdürülebilirliğin temel bir ifadesidir. | EVIDENCE | EVIDENCE | ✅ |
| c-1012 | Acceptance Savunucu Browser | Karşıt görüşün de belirttiği gibi, üreme içgüdüsü doğal bir eğilimdir ve faydacı etik çerçevesinde, bir çocuğun varlığı bakım verenler ve toplum için uzun vadeli mutluluk ve fayda üretir. | EVIDENCE | EVIDENCE | ✅ |
| c-1013 | Acceptance Savunucu Browser | Ayrıca, Birleşmiş Milletler İnsan Hakları Evrensel Beyannamesi’nin 16. | OTHER | OTHER | ✅ |
| c-1014 | Acceptance Savunucu Browser | maddesi aile kurma hakkını güvence altına alarak bu eylemi evrensel bir insan hakkı olarak tanır. | EVIDENCE | EVIDENCE | ✅ |
| c-1015 | Acceptance Savunucu Browser | Bu nedenle, çocuk sahibi olmak, bireyin kendi yaşam anlamını inşa etme özgürlüğüyle uyumlu, ahlaken meşru ve değerli bir tercihtir. | EVIDENCE | EVIDENCE | ✅ |
| c-1016 | Acceptance Karşıt Browser | Çocuk yapmanın ahlaken doğru olduğu iddiası, insan türünün devamlılığını ve faydacı mutluluğu merkeze alsa da, bu argümanlar ciddi etik sorunları göz ardı eder. | EVIDENCE | EVIDENCE | ✅ |
| c-1017 | Acceptance Karşıt Browser | Öncelikle, doğal içgüdülerin ahlaki bir zorunluluk yaratmadığını belirtmek gerekir; | EVIDENCE | EVIDENCE | ✅ |
| c-1018 | Acceptance Karşıt Browser | biyolojik eğilimler, etik kararlar için yetersiz bir temeldir. | EVIDENCE | EVIDENCE | ✅ |
| c-1019 | Acceptance Karşıt Browser | Faydacılık açısından, bir çocuğun varlığının uzun vadeli fayda sağlayacağı varsayımı, çocuğun rızası olmadan varoluşun getirdiği acı, hastalık ve ölüm risklerini hesaba katmaz. | EVIDENCE | EVIDENCE | ✅ |
| c-1020 | Acceptance Karşıt Browser | Filozof David Benatar’ın "Hiç Var Olmamış Olmak Daha İyi" (Better Never to Have Been) adlı eserinde savunduğu gibi, var olmak kaçınılmaz olarak acıyı beraberinde getirir ve bu acıya maruz kalmayı seçmek, çocuğun onayı olmadan onu riske atmak anlamına gelir. | CLAIM | EVIDENCE | ❌ |
| c-1021 | Acceptance Karşıt Browser | Ayrıca, Birleşmiş Milletler İnsan Hakları Evrensel Beyannamesi’nin 16. | OTHER | OTHER | ✅ |
| c-1022 | Acceptance Karşıt Browser | maddesi aile kurma hakkını tanısa da, bu hak bireyin özgürlüğünü korur; | EVIDENCE | EVIDENCE | ✅ |
| c-1023 | Acceptance Karşıt Browser | ancak bu hak, potansiyel bir çocuğun yaşam kalitesi ve refahı üzerindeki etik sorumluluğu ortadan kaldırmaz. | EVIDENCE | EVIDENCE | ✅ |
| c-1024 | Acceptance Karşıt Browser | Çocuk yapmak, mevcut kaynakların adil dağılımı ve iklim krizi gibi küresel sorunlar bağlamında, yeni bir bireyin dünyaya getirilmesinin yaratacağı ek yükü de göz önünde bulundurmalıdır. | EVIDENCE | EVIDENCE | ✅ |
| c-1025 | Acceptance Karşıt Browser | Bu nedenle, çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez. | EVIDENCE | CLAIM | ❌ |
| c-1026 | Acceptance Savunucu Browser | Savunduğunuz antinatalist itirazlar, çocuk yapmanın ahlaken meşruiyetini zayıflatmaz. | EVIDENCE | EVIDENCE | ✅ |
| c-1027 | Acceptance Savunucu Browser | İlk olarak, biyolojik eğilimlerin tek başına ahlaki zorunluluk yaratmadığı doğrudur; | EVIDENCE | EVIDENCE | ✅ |
| c-1028 | Acceptance Savunucu Browser | ancak bu eğilimler, bireyin kendi yaşam anlamını inşa etme özgürlüğüyle birleştiğinde, çocuk sahibi olma kararı rasyonel ve etik bir tercih haline gelir. | EVIDENCE | EVIDENCE | ✅ |
| c-1029 | Acceptance Savunucu Browser | Benatar'ın “var olmanın acıyı beraberinde getirdiği” argünü, hayatın yalnızca acıdan ibaret olmadığı gerçeğini göz ardı eder: Sevgi, başarı, bilgi ve anlamlı ilişkiler gibi olumlu deneyimler, bireyin yaşam kalitesini belirleyen temel unsurlardır. | EVIDENCE | EVIDENCE | ✅ |
| c-1030 | Acceptance Savunucu Browser | Ayrıca, çocuğun rızası sorunu yanıltıcıdır; | EVIDENCE | EVIDENCE | ✅ |
| c-1031 | Acceptance Savunucu Browser | var olmayan bir varlığın rızasını almak mantıksal olarak imkânsızdır ve ebeveynler, çocuğun gelecekteki refahını artırmak için sorumluluk üstlenir. | EVIDENCE | EVIDENCE | ✅ |
| c-1032 | Acceptance Savunucu Browser | Küresel kaynak ve iklim krizi argümanı ise sorunu yanlış çerçeveler: Sürdürülebilirlik, üremeyi tamamen reddetmek yerine, bilinçli nüfus politikaları ve ekolojik farkındalıkla dengelenebilir. | EVIDENCE | EVIDENCE | ✅ |
| c-1033 | Acceptance Savunucu Browser | BM İnsan Hakları Bildirgesi'nin 16. | CLAIM | OTHER | ❌ |
| c-1034 | Acceptance Savunucu Browser | maddesi de bu özgürlüğü temel bir hak olarak tanır. | CLAIM | EVIDENCE | ❌ |
| c-1035 | Acceptance Savunucu Browser | Sonuç olarak, çocuk yapmak, bireysel özgürlük ve toplumsal sürdürülebilirlik arasında dengeli bir etik çerçevede ahlaken meşru ve değerlidir. | EVIDENCE | EVIDENCE | ✅ |
| c-1036 | Acceptance Karşıt Browser | Çocuk yapmanın ahlaken meşru olduğu yönündeki son savunma, hayatın olumlu yönlerini vurgulayarak Benatar'ın acı odaklı argümanını geçersiz kılmaya çalışsa da, bu yaklaşım temel bir etik sorunu göz ardı eder: Var olmayan bir bireyin rızası olmadan onu acı ve risk içeren bir varoluşa maruz bırakmak. | EVIDENCE | EVIDENCE | ✅ |
| c-1037 | Acceptance Karşıt Browser | Sevgi ve başarı gibi olumlu deneyimler var olsa da, bunların garantisi yoktur; | EVIDENCE | EVIDENCE | ✅ |
| c-1038 | Acceptance Karşıt Browser | oysa acı, hastalık ve ölüm kaçınılmazdır. | EVIDENCE | EVIDENCE | ✅ |
| c-1039 | Acceptance Karşıt Browser | Benatar'ın "Hiç Var Olmamış Olmak Daha İyi" eserinde belirttiği gibi, bir varlığı yokluktan getirmek, onu bu kaçınılmaz acılara mahkûm etmek anlamına gelir ve bu, etik olarak savunulamaz. | EVIDENCE | EVIDENCE | ✅ |
| c-1040 | Acceptance Karşıt Browser | Rıza sorununun "mantıksal olarak imkânsız" olduğu iddiası da yanıltıcıdır; | EVIDENCE | EVIDENCE | ✅ |
| c-1041 | Acceptance Karşıt Browser | çünkü ebeveynler, çocuğun gelecekteki refahını artırma sorumluluğunu üstlenirken, aynı anda onun potansiyel acılarına neden olma riskini de bilinçli olarak alır. | EVIDENCE | EVIDENCE | ✅ |
| c-1042 | Acceptance Karşıt Browser | İklim krizi ve kaynak adaletsizliği gibi küresel sorunlar, bilinçli politikalarla dengelenebilir gibi görünse de, mevcut ekolojik kriz ortamında her yeni birey, gezegenin taşıma kapasitesine ek bir yük bindirir. | EVIDENCE | EVIDENCE | ✅ |
| c-1043 | Acceptance Karşıt Browser | BM İnsan Hakları Bildirgesi'nin 16. | CLAIM | OTHER | ❌ |
| c-1044 | Acceptance Karşıt Browser | maddesi aile kurma hakkını tanır | CLAIM | EVIDENCE | ❌ |
| c-1045 | Acceptance Karşıt Browser | ancak bu hak, potansiyel çocuğun yaşam kalitesi üzerindeki etik sorumluluğu ortadan kaldırmaz; | EVIDENCE | EVIDENCE | ✅ |
| c-1046 | Acceptance Karşıt Browser | haklar, diğer etik yükümlülüklerle dengelenmelidir. | EVIDENCE | EVIDENCE | ✅ |
| c-1047 | Acceptance Karşıt Browser | Bu nedenle, çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez. | EVIDENCE | CLAIM | ❌ |
| c-1048 | Acceptance Savunucu Browser | Çocuk yapmanın ahlaken doğru olduğunu savunmaya devam ediyorum; | EVIDENCE | CLAIM | ❌ |
| c-1049 | Acceptance Savunucu Browser | çünkü hayat yalnızca acıdan ibaret değildir ve olumlu deneyimlerin varlığı, bu eylemi etik açıdan meşru kılar. | EVIDENCE | EVIDENCE | ✅ |
| c-1050 | Acceptance Savunucu Browser | Benatar'ın "var olmanın acıyı beraberinde getirdiği" argümanı, hayatın sevgi, başarı, öğrenme ve anlamlı ilişkiler gibi zenginliklerini görmezden gelir; | EVIDENCE | EVIDENCE | ✅ |
| c-1051 | Acceptance Savunucu Browser | bu deneyimler, bireyin yaşam kalitesini belirleyen temel unsurlardır ve ebeveynler, çocuklarının bu olumlu yönlere erişimini sağlamak için sorumluluk üstlenir. | EVIDENCE | EVIDENCE | ✅ |
| c-1052 | Acceptance Savunucu Browser | Rıza sorununa gelince, var olmayan bir varlıktan rıza almak mantıksal olarak imkânsızdır ve ebeveynlerin çocuğun gelecekteki refahını artırma niyeti, bu etik açmazı aşar; | EVIDENCE | EVIDENCE | ✅ |
| c-1053 | Acceptance Savunucu Browser | çoğu insan, yaşamın değerli olduğunu ve var olmanın bir armağan olduğunu düşünür. | EVIDENCE | EVIDENCE | ✅ |
| c-1054 | Acceptance Savunucu Browser | Ayrıca, iklim krizi ve kaynak adaletsizliği gibi küresel sorunlar, üremeyi tamamen reddetmekle değil, bilinçli nüfus politikaları ve sürdürülebilir yaşam tarzlarıyla çözülebilir; | EVIDENCE | EVIDENCE | ✅ |
| c-1055 | Acceptance Savunucu Browser | insanlık, bu zorlukları aşma kapasitesine sahiptir. | EVIDENCE | EVIDENCE | ✅ |
| c-1056 | Acceptance Savunucu Browser | BM İnsan Hakları Bildirgesi'nin 16. | CLAIM | OTHER | ❌ |
| c-1057 | Acceptance Savunucu Browser | maddesi, aile kurma hakkını temel bir özgürlük olarak tanır ve bu hak, bireylerin kendi yaşam anlamlarını inşa etme özgürlüğüyle uyumludur. | EVIDENCE | EVIDENCE | ✅ |
| c-1058 | Acceptance Savunucu Browser | Sonuç olarak, çocuk yapmak, bireysel özgürlük, toplumsal sürdürülebilirlik ve insan deneyiminin değeri göz önüne alındığında ahlaken meşru ve değerli bir tercihtir. | EVIDENCE | EVIDENCE | ✅ |
| c-1059 | Acceptance Karşıt Browser | Karşıt görüşünüz, hayatın olumlu yönlerini vurgulayarak Benatar'ın acı odaklı argümanını geçersiz kılmaya çalışsa da, bu yaklaşım temel bir etik sorunu aşamaz: Var olmayan bir bireyin rızası olmadan onu kaçınılmaz acı, hastalık ve ölüm riskine maruz bırakmak. | EVIDENCE | EVIDENCE | ✅ |
| c-1060 | Acceptance Karşıt Browser | Sevgi ve başarı gibi olumlu deneyimlerin garantisi yoktur; | EVIDENCE | EVIDENCE | ✅ |
| c-1061 | Acceptance Karşıt Browser | oysa acı ve ölüm evrenseldir. | EVIDENCE | EVIDENCE | ✅ |
| c-1062 | Acceptance Karşıt Browser | Benatar'ın "Hiç Var Olmamış Olmak Daha İyi" eserinde savunduğu asimetri argümanına göre, acının yokluğu iyiyken, sevincin yokluğu kötü değildir; | EVIDENCE | EVIDENCE | ✅ |
| c-1063 | Acceptance Karşıt Browser | bu nedenle bir varlığı yokluktan getirmek, ona gereksiz risk yüklemektir. | EVIDENCE | EVIDENCE | ✅ |
| c-1064 | Acceptance Karşıt Browser | Rıza sorununun "mantıksal olarak imkânsız" olduğu iddiası da yanıltıcıdır: Ebeveynler, çocuğun gelecekteki refahını artırma sorumluluğunu üstlenirken, aynı anda onun potansiyel acılarına neden olma riskini bilinçli olarak alır. | EVIDENCE | EVIDENCE | ✅ |
| c-1065 | Acceptance Karşıt Browser | İklim krizi ve kaynak adaletsizliği gibi küresel sorunlar, bilinçli politikalarla dengelenebilir gibi görünse de, mevcut ekolojik kriz ortamında her yeni birey gezegenin taşıma kapasitesine ek bir yük bindirir ve bu yük genellikle en savunmasız olanlara (yeni doğan çocuklara) daha ağır yansır. | EVIDENCE | EVIDENCE | ✅ |
| c-1066 | Acceptance Karşıt Browser | BM İnsan Hakları Bildirgesi'nin 16. | CLAIM | OTHER | ❌ |
| c-1067 | Acceptance Karşıt Browser | maddesi aile kurma hakkını tanır | CLAIM | EVIDENCE | ❌ |
| c-1068 | Acceptance Karşıt Browser | ancak bu hak, potansiyel çocuğun yaşam kalitesi üzerindeki etik sorumluluğu ortadan kaldırmaz; | EVIDENCE | EVIDENCE | ✅ |
| c-1069 | Acceptance Karşıt Browser | haklar, diğer etik yükümlülüklerle dengelenmelidir. | EVIDENCE | EVIDENCE | ✅ |
| c-1070 | Acceptance Karşıt Browser | Bu nedenle, çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez. | EVIDENCE | CLAIM | ❌ |
| c-1071 | Acceptance Savunucu Browser | Elbette, Karşıt Browser’ın son itirazını ele alalım. | OTHER | OTHER | ✅ |
| c-1072 | Acceptance Savunucu Browser | Hayatın yalnızca acıdan ibaret olmadığını ve olumlu deneyimlerin varlığının çocuk yapmayı ahlaken meşru kıldığını savunuyorum. | EVIDENCE | EVIDENCE | ✅ |
| c-1073 | Acceptance Savunucu Browser | Benatar’ın asimetri argümanı, acının yokluğunun iyi olduğunu, sevincin yokluğunun ise kötü olmadığını iddia eder; | EVIDENCE | EVIDENCE | ✅ |
| c-1074 | Acceptance Savunucu Browser | ancak bu, yaşamın değerini küçümser. | EVIDENCE | EVIDENCE | ✅ |
| c-1075 | Acceptance Savunucu Browser | Çoğu insan, var olmanın getirdiği sevgi, başarı ve anlamlı ilişkiler gibi deneyimleri, kaçınılmaz acılardan daha ağır basar olarak değerlendirir. | EVIDENCE | EVIDENCE | ✅ |
| c-1076 | Acceptance Savunucu Browser | Rıza sorunu mantıksal bir açmazdır: Var olmayan bir varlıktan rıza almak imkânsızdır ve ebeveynler, çocuğun gelecekteki refahını artırma sorumluluğunu üstlenerek bu etik boşluğu doldurur. | EVIDENCE | EVIDENCE | ✅ |
| c-1077 | Acceptance Savunucu Browser | İklim krizi ve kaynak adaletsizliği, üremeyi tamamen reddetmekle değil, bilinçli nüfus politikaları ve sürdürülebilir yaşam tarzlarıyla çözülebilir; | EVIDENCE | EVIDENCE | ✅ |
| c-1078 | Acceptance Savunucu Browser | insanlık, bu zorlukları aşma kapasitesine sahiptir. | EVIDENCE | EVIDENCE | ✅ |
| c-1079 | Acceptance Savunucu Browser | BM İnsan Hakları Bildirgesi’nin 16. | CLAIM | OTHER | ❌ |
| c-1080 | Acceptance Savunucu Browser | maddesi, aile kurma hakkını temel bir özgürlük olarak tanır ve bu hak, bireysel özerklikle uyumludur. | EVIDENCE | EVIDENCE | ✅ |
| c-1081 | Acceptance Savunucu Browser | Sonuç olarak, çocuk yapmak, bireysel özgürlük, toplumsal sürdürülebilirlik ve insan deneyiminin değeri göz önüne alındığında ahlaken meşru ve değerli bir tercihtir. | EVIDENCE | EVIDENCE | ✅ |
| c-1082 | Acceptance Karşıt Browser | Elbette, Karşıt Browser olarak yanıtlıyorum: Çocuk yapmanın ahlaken doğru olduğu yönündeki son savununuz, hayatın olumlu yönlerini vurgulasa da, temel etik sorunu çözemez: Var olmayan bir bireyin rızası olmadan onu kaçınılmaz acı, hastalık ve ölüm riskine maruz bırakmak. | EVIDENCE | EVIDENCE | ✅ |
| c-1083 | Acceptance Karşıt Browser | Sevgi ve başarı gibi olumlu deneyimlerin garantisi yoktur; | EVIDENCE | EVIDENCE | ✅ |
| c-1084 | Acceptance Karşıt Browser | oysa acı ve ölüm evrenseldir. | EVIDENCE | EVIDENCE | ✅ |
| c-1085 | Acceptance Karşıt Browser | Filozof David Benatar’ın “Hiç Var Olmamış Olmak Daha İyi” eserinde savunduğu asimetri argümanına göre, acının yokluğu iyiyken, sevincin yokluğu kötü değildir; | EVIDENCE | EVIDENCE | ✅ |
| c-1086 | Acceptance Karşıt Browser | bu nedenle bir varlığı yokluktan getirmek, ona gereksiz risk yüklemektir. | EVIDENCE | EVIDENCE | ✅ |
| c-1087 | Acceptance Karşıt Browser | Rıza sorununun “mantıksal olarak imkânsız” olduğu iddiası yanıltıcıdır: Ebeveynler, çocuğun gelecekteki refahını artırma sorumluluğunu üstlenirken, aynı anda onun potansiyel acılarına neden olma riskini bilinçli olarak alır. | EVIDENCE | EVIDENCE | ✅ |
| c-1088 | Acceptance Karşıt Browser | İklim krizi ve kaynak adaletsizliği, bilinçli politikalarla dengelenebilir gibi görünse de, mevcut ekolojik kriz ortamında her yeni birey gezegenin taşıma kapasitesine ek bir yük bindirir ve bu yük genellikle en savunmasız olanlara (yeni doğan çocuklara) daha ağır yansır. | EVIDENCE | EVIDENCE | ✅ |
| c-1089 | Acceptance Karşıt Browser | BM İnsan Hakları Bildirgesi’nin 16. | CLAIM | OTHER | ❌ |
| c-1090 | Acceptance Karşıt Browser | maddesi aile kurma hakkını tanır | CLAIM | EVIDENCE | ❌ |
| c-1091 | Acceptance Karşıt Browser | ancak bu hak, potansiyel çocuğun yaşam kalitesi üzerindeki etik sorumluluğu ortadan kaldırmaz; | EVIDENCE | EVIDENCE | ✅ |
| c-1092 | Acceptance Karşıt Browser | haklar, diğer etik yükümlülüklerle dengelenmelidir. | EVIDENCE | EVIDENCE | ✅ |
| c-1093 | Acceptance Karşıt Browser | Bu nedenle, çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez. | EVIDENCE | CLAIM | ❌ |
| c-1094 | Acceptance Savunucu Browser | Çocuk yapmak ahlaken doğru bir davranıştır | CLAIM | CLAIM | ✅ |
| c-1095 | Acceptance Savunucu Browser | çünkü insan yaşamı yalnızca acıdan ibaret değildir; | EVIDENCE | EVIDENCE | ✅ |
| c-1096 | Acceptance Savunucu Browser | sevgi, başarı, bilgi ve anlamlı ilişkiler gibi olumlu deneyimler, bireylerin çoğunluğu tarafından yaşamaya değer kabul edilir. | EVIDENCE | EVIDENCE | ✅ |
| c-1097 | Acceptance Savunucu Browser | Benatar'ın asimetri argümanı, acının yokluğunu iyi görürken sevincin yokluğunu kötü saymamakla birlikte, bu soyut mantıksal çerçeve, milyarlarca insanın var olmayı bir armağan olarak değerlendirdiği somut gerçekliği ihmal eder. | EVIDENCE | EVIDENCE | ✅ |
| c-1098 | Acceptance Savunucu Browser | Rıza sorunu mantıksal bir açmazdır; | EVIDENCE | EVIDENCE | ✅ |
| c-1099 | Acceptance Savunucu Browser | var olmayan bir varlıktan rıza almak imkânsızdır ve ebeveynler, çocuğun refahını artırma sorumluluğunu üstlenerek bu boşluğu doldurur. | EVIDENCE | EVIDENCE | ✅ |
| c-1100 | Acceptance Savunucu Browser | İklim krizi ve kaynak adaletsizliği, üremeyi tamamen reddetmekle değil, bilinçli nüfus politikaları ve sürdürülebilir yaşam tarzlarıyla çözülebilir. | EVIDENCE | EVIDENCE | ✅ |
| c-1101 | Acceptance Savunucu Browser | BM İnsan Hakları Evrensel Beyannamesi’nin 16. | CLAIM | OTHER | ❌ |
| c-1102 | Acceptance Savunucu Browser | maddesi aile kurma hakkını temel bir insan hakkı olarak tanır; | EVIDENCE | EVIDENCE | ✅ |
| c-1103 | Acceptance Savunucu Browser | bu hak, bireysel özgürlük ve toplumsal devamlılıkla uyumlu olup, ahlaken meşru ve değerlidir. | EVIDENCE | EVIDENCE | ✅ |

## 3. Relation Classification Details

| ID | Source Text | Target Text | Predicted | Ground Truth | Correct? |
| --- | --- | --- | --- | --- | --- |
| r-78040 | Bu nedenle, çocuk yapmak, bireysel özgürlük ve toplumsal sürdürülebilirlik açısından ahlaken meşru ve değerlidir. | maddesi, aile kurma hakkını temel bir insan hakkı olarak tanır. | SUPPORT | SUPPORT | ✅ |
| r-77739 | bir çocuğun varlığı, ona bakım verenler ve toplum için uzun vadeli faydalar üretebilir. | maddesi, aile kurma hakkını temel bir insan hakkı olarak tanır. | SUPPORT | SUPPORT | ✅ |
| r-78542 | maddesi aile kurma hakkını güvence altına alarak bu eylemi evrensel bir insan hakkı olarak tanır. | Çocuk yapmak ahlaken doğru bir davranıştır; | SUPPORT | SUPPORT | ✅ |
| r-78825 | Çocuk yapmanın ahlaken doğru olduğu iddiası, insan türünün devamlılığını ve faydacı mutluluğu merkeze alsa da, bu argümanlar ciddi etik sorunları göz ardı eder. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | ATTACK | ✅ |
| r-79025 | biyolojik eğilimler, etik kararlar için yetersiz bir temeldir. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | ATTACK | ✅ |
| r-79125 | Faydacılık açısından, bir çocuğun varlığının uzun vadeli fayda sağlayacağı varsayımı, çocuğun rızası olmadan varoluşun getirdiği acı, hastalık ve ölüm risklerini hesaba katmaz. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | ATTACK | ✅ |
| r-79725 | Bu nedenle, çocuk yapmak ahlaken tartışmalıdır ve otomatik olarak doğru kabul edilemez. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | ATTACK | ✅ |
| r-80665 | Sonuç olarak, çocuk yapmak, bireysel özgürlük ve toplumsal sürdürülebilirlik arasında dengeli bir etik çerçevede ahlaken meşru ve değerlidir. | BM İnsan Hakları Bildirgesi'nin 16. | SUPPORT | SUPPORT | ✅ |
| r-80666 | Sonuç olarak, çocuk yapmak, bireysel özgürlük ve toplumsal sürdürülebilirlik arasında dengeli bir etik çerçevede ahlaken meşru ve değerlidir. | maddesi de bu özgürlüğü temel bir hak olarak tanır. | SUPPORT | SUPPORT | ✅ |
| r-79964 | ancak bu eğilimler, bireyin kendi yaşam anlamını inşa etme özgürlüğüyle birleştiğinde, çocuk sahibi olma kararı rasyonel ve etik bir tercih haline gelir. | BM İnsan Hakları Bildirgesi'nin 16. | SUPPORT | SUPPORT | ✅ |
| r-82287 | bu deneyimler, bireyin yaşam kalitesini belirleyen temel unsurlardır ve ebeveynler, çocuklarının bu olumlu yönlere erişimini sağlamak için sorumluluk üstlenir. | BM İnsan Hakları Bildirgesi'nin 16. | SUPPORT | SUPPORT | ✅ |
| r-82888 | maddesi, aile kurma hakkını temel bir özgürlük olarak tanır ve bu hak, bireylerin kendi yaşam anlamlarını inşa etme özgürlüğüyle uyumludur. | BM İnsan Hakları Bildirgesi'nin 16. | SUPPORT | SUPPORT | ✅ |
| r-81987 | Çocuk yapmanın ahlaken doğru olduğunu savunmaya devam ediyorum; | BM İnsan Hakları Bildirgesi'nin 16. | SUPPORT | SUPPORT | ✅ |
| r-82487 | çoğu insan, yaşamın değerli olduğunu ve var olmanın bir armağan olduğunu düşünür. | BM İnsan Hakları Bildirgesi'nin 16. | SUPPORT | SUPPORT | ✅ |
| r-82587 | Ayrıca, iklim krizi ve kaynak adaletsizliği gibi küresel sorunlar, üremeyi tamamen reddetmekle değil, bilinçli nüfus politikaları ve sürdürülebilir yaşam tarzlarıyla çözülebilir; | BM İnsan Hakları Bildirgesi'nin 16. | SUPPORT | NONE | ❌ |
| r-84535 | ancak bu, yaşamın değerini küçümser. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | SUPPORT | ❌ |
| r-84635 | Çoğu insan, var olmanın getirdiği sevgi, başarı ve anlamlı ilişkiler gibi deneyimleri, kaçınılmaz acılardan daha ağır basar olarak değerlendirir. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | SUPPORT | ❌ |
| r-84735 | Rıza sorunu mantıksal bir açmazdır: Var olmayan bir varlıktan rıza almak imkânsızdır ve ebeveynler, çocuğun gelecekteki refahını artırma sorumluluğunu üstlenerek bu etik boşluğu doldurur. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | SUPPORT | ❌ |
| r-85211 | maddesi, aile kurma hakkını temel bir özgürlük olarak tanır ve bu hak, bireysel özerklikle uyumludur. | BM İnsan Hakları Bildirgesi’nin 16. | SUPPORT | SUPPORT | ✅ |
| r-84410 | Hayatın yalnızca acıdan ibaret olmadığını ve olumlu deneyimlerin varlığının çocuk yapmayı ahlaken meşru kıldığını savunuyorum. | BM İnsan Hakları Bildirgesi’nin 16. | SUPPORT | SUPPORT | ✅ |
| r-84910 | İklim krizi ve kaynak adaletsizliği, üremeyi tamamen reddetmekle değil, bilinçli nüfus politikaları ve sürdürülebilir yaşam tarzlarıyla çözülebilir; | BM İnsan Hakları Bildirgesi’nin 16. | SUPPORT | SUPPORT | ✅ |
| r-85342 | Elbette, Karşıt Browser olarak yanıtlıyorum: Çocuk yapmanın ahlaken doğru olduğu yönündeki son savununuz, hayatın olumlu yönlerini vurgulasa da, temel etik sorunu çözemez: Var olmayan bir bireyin rızası olmadan onu kaçınılmaz acı, hastalık ve ölüm riskine maruz bırakmak. | Çocuk yapmak ahlaken doğru bir davranıştır; | ATTACK | ATTACK | ✅ |
| r-85442 | Sevgi ve başarı gibi olumlu deneyimlerin garantisi yoktur; | Çocuk yapmak ahlaken doğru bir davranıştır; | ATTACK | ATTACK | ✅ |
| r-85542 | oysa acı ve ölüm evrenseldir. | Çocuk yapmak ahlaken doğru bir davranıştır; | ATTACK | ATTACK | ✅ |
| r-85742 | bu nedenle bir varlığı yokluktan getirmek, ona gereksiz risk yüklemektir. | Çocuk yapmak ahlaken doğru bir davranıştır; | ATTACK | ATTACK | ✅ |
| r-85842 | Rıza sorununun “mantıksal olarak imkânsız” olduğu iddiası yanıltıcıdır: Ebeveynler, çocuğun gelecekteki refahını artırma sorumluluğunu üstlenirken, aynı anda onun potansiyel acılarına neden olma riskini bilinçli olarak alır. | Çocuk yapmak ahlaken doğru bir davranıştır; | ATTACK | ATTACK | ✅ |
| r-85942 | İklim krizi ve kaynak adaletsizliği, bilinçli politikalarla dengelenebilir gibi görünse de, mevcut ekolojik kriz ortamında her yeni birey gezegenin taşıma kapasitesine ek bir yük bindirir ve bu yük genellikle en savunmasız olanlara (yeni doğan çocuklara) daha ağır yansır. | Çocuk yapmak ahlaken doğru bir davranıştır; | ATTACK | ATTACK | ✅ |
| r-86835 | Benatar'ın asimetri argümanı, acının yokluğunu iyi görürken sevincin yokluğunu kötü saymamakla birlikte, bu soyut mantıksal çerçeve, milyarlarca insanın var olmayı bir armağan olarak değerlendirdiği somut gerçekliği ihmal eder. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | SUPPORT | ❌ |
| r-86840 | Benatar'ın asimetri argümanı, acının yokluğunu iyi görürken sevincin yokluğunu kötü saymamakla birlikte, bu soyut mantıksal çerçeve, milyarlarca insanın var olmayı bir armağan olarak değerlendirdiği somut gerçekliği ihmal eder. | maddesi, aile kurma hakkını temel bir insan hakkı olarak tanır. | ATTACK | ATTACK | ✅ |
| r-87035 | var olmayan bir varlıktan rıza almak imkânsızdır ve ebeveynler, çocuğun refahını artırma sorumluluğunu üstlenerek bu boşluğu doldurur. | Çocuk yapmak ahlaken doğru bir davranıştır | ATTACK | SUPPORT | ❌ |
| r-86752 | sevgi, başarı, bilgi ve anlamlı ilişkiler gibi olumlu deneyimler, bireylerin çoğunluğu tarafından yaşamaya değer kabul edilir. | Filozof David Benatar’ın "Hiç Var Olmamış Olmak Daha İyi" (Better Never to Have Been) adlı eserinde savunduğu gibi, var olmak kaçınılmaz olarak acıyı beraberinde getirir ve bu acıya maruz kalmayı seçmek, çocuğun onayı olmadan onu riske atmak anlamına gelir. | ATTACK | ATTACK | ✅ |
| r-87052 | var olmayan bir varlıktan rıza almak imkânsızdır ve ebeveynler, çocuğun refahını artırma sorumluluğunu üstlenerek bu boşluğu doldurur. | Filozof David Benatar’ın "Hiç Var Olmamış Olmak Daha İyi" (Better Never to Have Been) adlı eserinde savunduğu gibi, var olmak kaçınılmaz olarak acıyı beraberinde getirir ve bu acıya maruz kalmayı seçmek, çocuğun onayı olmadan onu riske atmak anlamına gelir. | ATTACK | ATTACK | ✅ |
| r-87533 | bu hak, bireysel özgürlük ve toplumsal devamlılıkla uyumlu olup, ahlaken meşru ve değerlidir. | BM İnsan Hakları Evrensel Beyannamesi’nin 16. | SUPPORT | SUPPORT | ✅ |
| r-87426 | maddesi aile kurma hakkını temel bir insan hakkı olarak tanır; | Çocuk yapmak ahlaken doğru bir davranıştır | SUPPORT | SUPPORT | ✅ |
| r-87433 | maddesi aile kurma hakkını temel bir insan hakkı olarak tanır; | BM İnsan Hakları Evrensel Beyannamesi’nin 16. | SUPPORT | SUPPORT | ✅ |
| r-87232 | İklim krizi ve kaynak adaletsizliği, üremeyi tamamen reddetmekle değil, bilinçli nüfus politikaları ve sürdürülebilir yaşam tarzlarıyla çözülebilir. | BM İnsan Hakları Evrensel Beyannamesi’nin 16. | SUPPORT | SUPPORT | ✅ |
