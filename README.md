# Subarr

> AI-powered subtitle translation service — character-aware, emotion-aware, built for the \*arr ecosystem.

Subarr takes an audio file and a subtitle file, runs them through an ML pipeline (speaker diarization → emotion detection → character identification → LLM translation), and returns a styled **ASS/SSA** subtitle file with per-character colors.

Integration services (Jellyfin, Plex, etc.) are planned as separate repos (`subarr-jellyfin`, etc.) that call this API.

---

## Architecture

```
┌─────────────┐    multipart     ┌──────────────────┐
│   Client    │ ──────────────→  │  subarr-api      │
│ (Jellyfin   │                  │  (Rust/Actix-web) │
│  plugin etc)│ ←────────────── │                  │
└─────────────┘   .ass download  └────────┬─────────┘
                                          │ Redis queue
                                          ▼
                                 ┌──────────────────┐
                                 │  ml-worker       │
                                 │  (Python)        │
                                 │                  │
                                 │  pyannote        │
                                 │  wav2vec2        │
                                 │  Gemini 2.5 Flash│
                                 └────────┬─────────┘
                                          │
                                 ┌────────▼─────────┐
                                 │  MinIO (S3)      │
                                 │  audio / results │
                                 └──────────────────┘
```

| Component     | Tech                          | Role                                      |
|---------------|-------------------------------|-------------------------------------------|
| `subarr-api`  | Rust, Actix-web               | REST API, job management, ASS formatting  |
| `ml-worker`   | Python                        | ML pipeline, Gemini translation           |
| Queue         | Redis                         | Job queue between API and worker          |
| Storage       | MinIO (S3-compatible)         | Audio, subtitle, and result files         |
| `dashboard`   | Next.js *(planned)*           | User settings UI                          |

---

## ML Pipeline

1. **Diarization** — extract speaker segments and embeddings (pyannote)
   - **TMDB Fetch** *(parallel)* — fetch cast list, synopsis, genres for context
2. **Subtitle Mapping** — assign segments to subtitle lines (overlap + temporal proximity)
3. **Overlap Detection** — flag regions where multiple speakers talk simultaneously
4. **Emotion Detection** — infer emotional tone per segment (wav2vec2)
5. **Character ID** — match speaker embeddings to named characters using TMDB cast + Gemini LLM; skipped when no TMDB metadata is available
6. **Translation** — batch translation with character name + emotion + show context (Gemini 2.5 Flash, 100 lines/batch)
7. **ASS/SSA Output** — per-character colored styles, HTML→ASS tag conversion, assembled by the Rust API

---

## API

Base path: `/api/v1`

### `GET /health`
Returns service version and status.

### `POST /translate`
Submit a translation job. Accepts `multipart/form-data`:

| Field            | Type   | Required | Description                        |
|------------------|--------|----------|------------------------------------|
| `audio`          | file   | yes      | Audio file of the episode          |
| `subtitle`       | file   | yes      | SRT subtitle file                  |
| `target_language`| string | no       | Target language code (default: `tr`) |
| `media_name`     | string | no       | Show/movie title                   |
| `metadata`       | JSON   | no       | `{"tmdb_id": 1234, "season": 1, "episode": 3, "title": "..."}` |

Returns a job object with an `id` and `status`.

### `GET /jobs/{job_id}`
Poll job status. Statuses: `queued` → `processing` → `completed` / `failed`.

### `GET /jobs/{job_id}/download`
Download the translated `.ass` file once the job is `completed`.

---

## Configuration

Both services are configured via environment variables.

### subarr-api

| Variable           | Description                    | Default       |
|--------------------|--------------------------------|---------------|
| `HOST`             | Bind address                   | `0.0.0.0`     |
| `PORT`             | Bind port                      | `8080`        |
| `REDIS_URL`        | Redis connection URL           | —             |
| `S3_ENDPOINT`      | MinIO/S3 endpoint URL          | —             |
| `S3_ACCESS_KEY`    | S3 access key                  | —             |
| `S3_SECRET_KEY`    | S3 secret key                  | —             |
| `S3_BUCKET`        | S3 bucket name                 | —             |
| `S3_REGION`        | S3 region                      | `us-east-1`   |
| `SUBARR_LOG_LEVEL` | Log level                      | `info`        |

### ml-worker

| Variable              | Description                          |
|-----------------------|--------------------------------------|
| `REDIS_URL`           | Redis connection URL                 |
| `S3_ENDPOINT`         | MinIO/S3 endpoint URL                |
| `S3_ACCESS_KEY`       | S3 access key                        |
| `S3_SECRET_KEY`       | S3 secret key                        |
| `S3_BUCKET`           | S3 bucket name                       |
| `GEMINI_API_KEY`      | Google Gemini API key                |
| `TMDB_API_KEY`        | TMDB API key (optional)              |
| `HF_TOKEN`            | Hugging Face token (for pyannote)    |
| `TARGET_LANGUAGE`     | Default target language              |

---

## Running Locally

```bash
# Start dependencies
docker compose up -d   # or docker compose -f docker-compose.gpu.yml up -d for GPU

# Build and run the API
cd subarr-api
cargo run

# Run the ML worker
cd ml-worker
pip install -r requirements.txt
python worker.py
```

---

## Kubernetes Deployment

Manifests are in `k8s/`. The stack runs on k3s under the `phirios` namespace.

```bash
kubectl apply -f k8s/
```

Images are published to GitHub Container Registry (`ghcr.io/phirios/subarr-api`, `ghcr.io/phirios/subarr-ml-worker`).

---

## Roadmap

- [ ] Dashboard (Next.js) — user settings, job history
- [ ] `subarr-jellyfin` — Jellyfin integration plugin
- [ ] Plex / Emby integration services
- [ ] Whisper fallback for audio-only jobs (no external subtitle)
