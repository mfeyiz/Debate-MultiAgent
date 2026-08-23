# Canlı Tartışma Performans Yol Haritası

Demoda "gerçekten iyi çalışan" bir canlı tartışma deneyimi için, gözlemlenen
sorunlardan yola çıkan önceliklendirilmiş plan. Temel bulgu: **bileşen çıkarımı
(claim/evidence/other) iyi**, asıl zayıflık **konuya göre stance/polarity
(support/attack/none)** sınıflandırmasında.

## Kök neden analizi (gözlemlenen)

3 turluk örnek koşuda (`run_demo_debate.py`, konu: "Sosyal medya demokrasiye zarar
veriyor"):

1. **Polarity körlüğü — en kritik.** Konu-ilişkisi sınıflandırıcısı bir cümlenin
   *kendi içinde* olumlu/iddialı mı yoksa karşıt/olumsuzlayıcı mı olduğunu yakalıyor;
   *spesifik tez cümlesini* destekleyip desteklemediğini değil. Olumsuz çerçeveli bir
   konuda (`X zarar veriyor`), karşıt tarafın "X faydalıdır" diyen olumlu cümleleri
   sıklıkla `support` etiketleniyor — oysa konuya göre `attack` olmalı.
   - Örn. #1154 "vatandaş denetimi güçlendirir" → `support` (yanlış, `attack` olmalı)
   - Örn. #1188 "okuryazarlık dezenformasyonu %30 azaltır" → `support` (yanlış)
2. **Yön (direction) tutarsızlığı.** `analyze_debate` ilişkiyi `(konu → metin)`
   yönünde çağırıyor (`app/services/debate_service.py`). Eğitim verisi farklı yönde
   (`metin → hedef`) ise model kafa karışıklığı yaşar.
3. **Bileşen tipi sağlam.** Kaynak içeren cümleler (*Science 2018*, *PNAS*, *AB
   Komisyonu*) güvenilir şekilde `evidence`. Küçük kayıplar var (#1190 evidence →
   claim) ama demo için yeterli.

## P0 — Demo için hızlı kazanımlar (1–2 gün)

- [x] **Olumlu çerçeveli konular kullan.** Polarity zaafını lehimize çevirir:
      olumlu tez + olumlu destekleyici kanıt → doğru `support`. Dashboard'a demo
      konu çipleri eklendi; lab'a `Örnek 4` (uzaktan çalışma → support, doğrulandı
      strength≈0.98) eklendi.
- [ ] **İlişki yönünü tek tipe sabitle.** `analyze_debate` içindeki `(konu, metin)`
      çağrısını eğitimdeki yönle hizala; her iki yönü ölçüp tutarlı olanı seç ve
      kod yorumunda belgele.
- [ ] **Stance guard (heuristik kalkan).** Konu polaritesini (olumlu/olumsuz tez)
      tespit edip düşük güvenli (<0.6) topic-relation tahminlerinde işareti
      düzelt/maskele. Demoda yanlış kırmızı/yeşil vurguları engeller.

## P1 — Veri ve model (1–2 hafta)

- [ ] **Polarity-dengeli eğitim verisi.** Relation sınıflandırıcısına *yüzey
      duygusu ≠ stance* olan hard-negative örnekler ekle: olumsuz çerçeveli teze
      yöneltilmiş olumlu-tonlu çürütme cümleleri `attack` olarak etiketli olsun.
      Hem olumlu hem olumsuz çerçeveli konuları dengele.
- [ ] **Konu-göreceli eğitim formatı.** İlişki örneklerini açıkça
      `(tez_cümlesi, argüman)` çiftleri olarak ve her iki tez polaritesinde üret;
      `scripts/generate_dataset_v5.py` + `consensus_label.py` akışını bu eksen için
      genişlet.
- [ ] **Kalibrasyon yenile.** `models/relation_classifier/final/calibration.json`
      ve `RELATION_GRAPH_THRESHOLD`/`relation_threshold` (şu an 0.55) polarity-dengeli
      holdout üzerinde yeniden fit edilsin.

## P2 — Pipeline ve UX (orta vade)

- [ ] **İki aşamalı doğrulama.** Çelişkili/düşük güvenli topic-relation
      tahminlerini LLM-as-judge ile doğrula (ensemble). Sadece sınır vakalarda
      çağrılır, maliyet düşük kalır.
- [ ] **Rol bilgisini girdiye kat.** İlişki girdisine konuşmacının
      proponent/opponent rolünü işaret olarak ekle; model stance yönünü öğrenmede
      ek sinyal alır.
- [ ] **Regeneration döngüsü ayarı.** Canlı arenada zayıf skorlu mesaj yeniden
      üretim eşiğini ve feedback metnini demoda akıcı kalacak şekilde ayarla.

## P3 — Değerlendirme ve izleme

- [ ] **Polarity-dengeli holdout.** Hem olumlu hem olumsuz çerçeveli konulardan
      oluşan ayrı bir gold-holdout; topic-relation macro-F1'i konu polaritesine
      göre ayrı raporla. (`scripts/build_gold_holdout.py` genişlet.)
- [ ] **Üretim izleme.** Canlı tartışmalarda sınıf bazlı confusion ve düşük güven
      oranını logla; demo öncesi regresyon kapısı olarak kullan.

## Başarı ölçütü

- Olumsuz çerçeveli konularda karşıt-taraf cümlelerinin `support` yanlış-pozitif
  oranı **< %10**.
- Polarity-dengeli holdout'ta topic-relation macro-F1 **≥ 0.80** (her iki polarite
  için ayrı ayrı).
- Canlı arenada bir tartışma boyunca gözle bakıldığında stance vurguları
  (yeşil=support / kırmızı=attack) tutarlı görünüyor.
