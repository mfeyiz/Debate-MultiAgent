# Logos Teyit Chrome Extension

Bu klasör unpacked Chrome extension olarak kullanılabilir.

1. Backend'i çalıştırın:
   ```bash
   uv run python main.py
   ```
2. Chrome'da `chrome://extensions` sayfasını açın.
3. Developer mode'u açın.
4. `Load unpacked` ile bu `extension/` klasörünü seçin.
5. Bir haber sayfasında eklentiye basıp `Bu sayfayı analiz et` veya `Seçili metni analiz et` düğmesini kullanın.

Varsayılan API adresi `http://127.0.0.1:5000`. Farklı port kullanırsanız popup içindeki API adresini değiştirin.
