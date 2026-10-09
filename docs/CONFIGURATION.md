# Configuration

DocAssist has three levels of settings. Use the lowest level that does the job.

| Level | Who changes it | Where | Takes effect |
| --- | --- | --- | --- |
| **Server** | Whoever deploys | Environment variables (`.env`, `.env.docker`) | On restart |
| **Platform** | Superadmins | Portal → Platform settings, and Portal → AI providers | Immediately |
| **Organization** | Each organization's admins | Settings page | Immediately, for that organization |

Platform settings start from the server values the first time, so a fresh install behaves the same either way; after that, the portal is where they're changed. New organizations start from the platform defaults.

## Server settings (environment variables)

Read from the environment, `django_app/.env`, then `rag_bot/.env`. In Docker, from `.env.docker`. Examples: `django_app/.env.example`, `django_app/.env.docker.example`.

### Django and security

| Variable | Default | Notes |
| --- | --- | --- |
| `DJANGO_SECRET_KEY` | dev key | **Required** when debug is off. Also encrypts AI provider keys; keep it stable. |
| `DJANGO_DEBUG` | `true` | Set `false` in production (Docker does) |
| `DJANGO_ALLOWED_HOSTS` | `localhost,127.0.0.1` | Comma-separated host names |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | – | Full origins, e.g. `https://docs.example.com` |
| `SESSION_COOKIE_SECURE`, `CSRF_COOKIE_SECURE` | `true` when debug is off | Set `false` only for plain-http local testing |
| `DJANGO_TIME_ZONE` | `UTC` (this install: `Africa/Nairobi`) | Used for dates on pages, charts and "busiest times". The Home greeting and clock use each viewer's own device. |
| `PLATFORM_ENVIRONMENT` | `Development` when debug is on, else `Production` | Shown in the Platform Console status bar, e.g. `Staging` |
| `LOG_LEVEL` | `INFO` | |
| `ALLOW_SIGNUP` | `true` | Starting value for the platform setting |

### Database and storage

| Variable | Default | Notes |
| --- | --- | --- |
| `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`, `POSTGRES_HOST`, `POSTGRES_PORT` | – | Set `POSTGRES_DB` to use PostgreSQL; without it, SQLite (`db.sqlite3`) |
| `RAG_VECTOR_BACKEND` | `chroma` (Docker: `pgvector`) | `pgvector` keeps passages in PostgreSQL; `chroma` in a folder. Switching needs `manage.py reindex`. |
| `RAG_VECTOR_DIR` | `django_app/vector_store` | Chroma's folder |
| `MEDIA_ROOT` | `django_app/media` (Docker: `/data/media`) | Uploaded files. Keep on persistent storage. |

### Search and answers

| Variable | Default | Notes |
| --- | --- | --- |
| `GOOGLE_API_KEY` | – | Used by a Gemini provider whose key field is blank |
| `RAG_DEFAULT_MODEL` | `gemini-flash-latest` | Starting value for the platform's default Gemini model |
| `RAG_EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Changing it needs a re-index (and pgvector expects 384 dimensions) |
| `RAG_EMBEDDING_BACKEND` | `sentence-transformers` | `fake` is for tests only |
| `RAG_LLM_BACKEND` | `gemini` | `echo` is for tests only |
| `RAG_CHUNK_SIZE`, `RAG_CHUNK_OVERLAP` | 1000, 200 characters | Changing them needs a re-index |
| `RAG_TOP_K`, `RAG_HISTORY_TURNS`, `RAG_GAP_SCORE` | 5, 3, 0.35 | Starting values for the platform defaults |
| `RAG_MAX_OUTPUT_TOKENS`, `RAG_TEMPERATURE` | 600, 0.2 | "Balanced" answer length and "precise" creativity; organizations adjust them in Settings |
| `RAG_ANSWER_CACHE_SECONDS` | 86400 | Starting value for answer reuse (platform setting, in hours) |
| `RAG_QUOTED_FALLBACK` | `true` | Starting value for "quote the documents when every provider fails" |
| `RAG_MAX_UPLOAD_MB` | 25 | Starting value for the largest upload |
| `RAG_INGEST_MODE`, `RAG_INGEST_WORKERS` | `thread`, 2 | `sync` indexes inside the request (tests) |
| `RAG_WARMUP` | `true` | Load the embedding model at start so the first question is fast |

### Docker only

| Variable | Notes |
| --- | --- |
| `DJANGO_SUPERUSER_USERNAME`, `DJANGO_SUPERUSER_PASSWORD`, `DJANGO_SUPERUSER_EMAIL` | Create a first superadmin on start if it doesn't exist |

## Platform settings (portal)

Set on **Platform → Platform settings**; see the [Superadmin guide](SUPERADMIN_GUIDE.md#platform-settings).

| Setting | Default | Range |
| --- | --- | --- |
| Allow self sign-up | from `ALLOW_SIGNUP` | on/off |
| Stay signed in for (hours) | 336 (14 days) | 1–2160 |
| Failed sign-ins before a pause | 5 | 3–20 |
| Pause length (minutes) | 15 | 1–1440 |
| Default Gemini model | from `RAG_DEFAULT_MODEL` | text |
| Monthly question limit (new organizations) | 0 = unlimited | ≥ 0 |
| Passages per answer / Conversation memory / Knowledge-gap threshold (new organizations) | 5 / 3 / 0.35 | 1–12 / 0–10 / 0.05–0.9 |
| Largest upload (MB) | from `RAG_MAX_UPLOAD_MB` | 1–500 |
| Quote the documents when every AI provider fails | on | on/off |
| Reuse answers to repeated questions (hours) | 24 | 0–720 (0 = off) |
| Announcement, style | blank, information | 300 characters |
| Maintenance mode, message | off | 300 characters |
| Require two-factor sign-in for the Platform Console | off | on/off (set in the Security centre) |

AI providers (which models write answers, their keys and order) are also platform-level: see [AI_PROVIDERS.md](AI_PROVIDERS.md).

## Organization settings

Set on each organization's **Settings** page; see the [Admin guide](ADMIN_GUIDE.md#settings).

| Setting | Default |
| --- | --- |
| Name, assistant name | – , "DocAssist" |
| Welcome message, suggested questions (up to 6), show sources | default text, automatic, on |
| Answer length / creativity / knowledge scope / reply language | balanced / precise / documents only / same as the question |
| Use external AI providers | on |
| Gemini model, custom instructions | platform default, none |
| Passages per answer, conversation memory, knowledge-gap threshold | the platform defaults when the organization was created |
| Monthly question limit, reuse answers | the platform default, on |
