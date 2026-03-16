# Subarr - AI-Powered Subtitle Translation Service

## Konsept
Genel amaçlı altyazı çeviri backend servisi. Ses analizi yaparak karakter bazlı, duygu-farkındalıklı altyazı çevirisi.
Media server entegrasyonları (Jellyfin, Plex vb.) ayrı servisler olarak yazılacak.

## Mimari

### Subarr Core (Bu proje)
Bağımsız bir altyazı çeviri API'si. Girdi olarak ses dosyası + altyazı dosyası alır, çevrilmiş altyazı döner.

**Pipeline:**
1. **Diarization** (pyannote) - ses dosyasından speaker segment'leri + speaker embedding'leri çıkar
   - **TMDB Fetch** (paralel) - cast + dizi/bölüm context'i (synopsis, genres) çek
2. **Subtitle Mapping** - segment'leri altyazı satırlarına ata (overlap hesabı + temporal proximity tiebreaker)
3. **Overlap Detection** - aynı anda konuşan bölgeleri tespit et ve işaretle
4. **Emotion Detection** (wav2vec2) - mapped segment'ler üzerinde duygusal ton çıkar
5. **Character ID** (Gemini LLM) - TMDB cast + dizi/bölüm context + overlap flag'leri ile speaker → karakter eşleştirmesi. LLM aynı zamanda merge kararı da verir. Metadata/TMDB yoksa atlanır
6. **Translation** (Gemini LLM) - karakter ismi + duygu + dizi/bölüm context ile batch'li çeviri (100/batch, 15s rate limit arası)
7. **ASS/SSA Output** - karakter bazlı renkli stiller, HTML→ASS tag dönüşümü, Rust API tarafında JSON → ASS

### Entegrasyon Servisleri (Ayrı projeler)
- **subarr-jellyfin**: Jellyfin'den audio/subtitle çeker, Subarr API'ye gönderir, sonucu Jellyfin'e upload eder
- Gelecekte: subarr-plex, subarr-emby vb.

## Kararlar
- **API + Orchestration**: Rust (Actix-web) - API servisi, job yönetimi, subtitle parsing/formatting
- **ML Worker**: Python - pyannote, whisper, emotion detection, Gemini API çevirisi
- **Haberleşme**: Redis (Rust ↔ Python arası job queue)
- **Dashboard**: Next.js (Subarr Core dashboard) - kullanıcı ayarları dashboard üzerinden yönetilir
- **Storage**: MinIO (S3 uyumlu) - ses/altyazı dosyaları ve sonuçlar
- **Config**: Env'den (API key'ler, Redis URL, S3)
- **ML**: pyannote (diarization), wav2vec2 (emotion detection)
- **Çeviri**: Gemini 2.5 Flash (Google, ücretsiz tier)
- **Format**: ASS/SSA
- **Hedef dil**: Türkçe (ama configurable)
- **İsim**: Subarr (*arr ekosistemi uyumlu)

- **Deploy**: Kubernetes (k3s, phirios namespace), GitHub Container Registry
- **Repo**: github.com/Phirios/subarr (public)

## Yapılacaklar
- Dashboard (Next.js)
- Entegrasyon servisi ayrı repo olarak planlanacak
