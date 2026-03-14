# Subarr - AI-Powered Subtitle Translation Service

## Konsept
Genel amaçlı altyazı çeviri backend servisi. Ses analizi yaparak karakter bazlı, duygu-farkındalıklı altyazı çevirisi.
Media server entegrasyonları (Jellyfin, Plex vb.) ayrı servisler olarak yazılacak.

## Mimari

### Subarr Core (Bu proje)
Bağımsız bir altyazı çeviri API'si. Girdi olarak ses dosyası + altyazı dosyası alır, çevrilmiş altyazı döner.

**Pipeline:**
1. Ses dosyasından speaker diarization (pyannote) - kim ne zaman konuşuyor
2. Emotion detection - duygusal ton çıkarma
3. Mevcut altyazı satırlarıyla eşleştirme (crowd/arka plan filtresi)
4. TMDB cast bilgileriyle konuşmacı → karakter eşleştirme (opsiyonel, metadata verilirse)
5. Gemini 2.5 Flash ile context-aware çeviri (karakter kişiliği, duygu, dizi akışı)
6. ASS/SSA formatında çıktı (karakter renkli stil)

### Entegrasyon Servisleri (Ayrı projeler)
- **subarr-jellyfin**: Jellyfin'den audio/subtitle çeker, Subarr API'ye gönderir, sonucu Jellyfin'e upload eder
- Gelecekte: subarr-plex, subarr-emby vb.

## Kararlar
- **API + Orchestration**: Rust (Actix-web) - API servisi, job yönetimi, subtitle parsing/formatting
- **ML Worker**: Python - pyannote, whisper, emotion detection, Gemini API çevirisi
- **Haberleşme**: Redis (Rust ↔ Python arası job queue)
- **Dashboard**: Next.js (Subarr Core dashboard) - kullanıcı ayarları dashboard üzerinden yönetilir
- **Config**: Kritik config (API key'ler, Redis URL, DB) env'den, geri kalan ayarlar dashboard üzerinden
- **ML**: pyannote, whisper, emotion detection
- **Çeviri**: Gemini 2.5 Flash (Google, ücretsiz tier)
- **Format**: ASS/SSA
- **Hedef dil**: Türkçe (ama configurable)
- **İsim**: Subarr (*arr ekosistemi uyumlu)

- **Deploy**: Kubernetes (k3s, phirios namespace), GitHub Container Registry
- **Repo**: github.com/Phirios/subarr (public)

## Yapılacaklar
- Dashboard (Next.js)
- Entegrasyon servisi ayrı repo olarak planlanacak
