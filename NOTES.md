# Subarr - AI-Powered Subtitle Translation Service

## Konsept
Genel amaçlı altyazı çeviri backend servisi. Ses analizi yaparak karakter bazlı, duygu-farkındalıklı altyazı çevirisi.
Media server entegrasyonları (Jellyfin, Plex vb.) ayrı servisler olarak yazılacak.

## Mimari

### Subarr Core (Bu proje)
Bağımsız bir altyazı çeviri API'si. Girdi olarak ses dosyası + altyazı dosyası alır, çevrilmiş altyazı döner.

**Pipeline:**
1. **Diarization** (pyannote) - ses dosyasından speaker segment'leri + speaker embedding'leri çıkar
   - **TMDB Fetch** (paralel) - metadata verilmişse bölüm cast'i (dizi) veya genel cast (film) çek
2. **Subtitle Mapping** - altyazı zaman aralıklarına denk gelmeyen segment'leri at (arka plan/crowd/müzik filtresi) + kalan segment'leri altyazı satırlarına ata (tek geçiş, zaman overlap hesabı)
3. **Overlap Detection** - aynı anda konuşan bölgeleri tespit et ve işaretle
4. **Emotion Detection** - mapped segment'ler üzerinde duygusal ton çıkar (sadece diyalog sesleri)
5. **Character ID** (Gemini LLM) - TMDB cast listesi + altyazı içeriği + overlap flag'leri ile speaker → karakter eşleştirmesi. Metadata/TMDB yoksa atlanır, SPEAKER_XX label'ları kalır
6. **Post-ID Merge** - LLM'in eşleştiremediği speaker'ları embedding cosine similarity ile tanınan karakterlere merge et + mapping label'larını güncelle. Hiçbirine benzemeyen speaker'lar ekstra/arka plan olarak bırakılır
7. **Translation** (Gemini LLM) - karakter ismi + duygu + context ile batch'li çeviri (50 altyazı/batch, 15s rate limit arası)
8. **ASS/SSA Output** - karakter bazlı renkli stiller, Rust API tarafında JSON → ASS dönüşümü

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
