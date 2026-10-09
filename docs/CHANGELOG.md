# Changelog

All notable changes to DocAssist, newest first. The project uses [semantic versioning](https://semver.org/): **MAJOR** for a new generation or breaking change, **MINOR** for new features, **PATCH** for fixes. The current version is in [`django_app/VERSION`](../django_app/VERSION); each release is a git tag `vX.Y.Z`. How to record a change and cut a release: [CONTRIBUTING.md](../CONTRIBUTING.md).

## [3.4.1] - 2026-10-07

### Added

- The repository is versioned: `django_app/VERSION` is the single source of the version (shown in the Platform Console header and the API health check), with git tags per release.
- `CONTRIBUTING.md`: commit conventions, versioning and the release checklist.
- The Home page shows today's date and time under the greeting.

### Changed

- The Docker app is published on **port 8001** (`http://localhost:8001`); port 8000 belongs to another local project.
- The app's time zone is **Africa/Nairobi** (`DJANGO_TIME_ZONE`), so dates, charts and "busiest times" use local time.
- The Platform Console sidebar uses the normal light/dark theme instead of a separate dark palette.
- `.gitignore` covers secrets (`.env*`), databases, uploaded documents, vector indexes and the legacy JSON stores.

### Fixed

- The Home greeting said "Good afternoon" in the evening: it used UTC. It now uses the viewer's own clock.

### Security

- Legacy data files (`users.json`, `api_keys.json`, `analytics.json`), uploaded documents and the Chroma index are no longer tracked by git.

## [3.4.0] - 2026-10-05

### Added

- **Platform Console** at `/platform/`, separate from organization workspaces, with a status bar (environment, maintenance, AI provider health) and its own Ctrl+K search across organizations, users and API keys.
- **Console roles:** owner (superadmin), support (view, support mode) and finance (usage and plans).
- **Operations dashboard:** questions, active organizations and people, failure and backup rates against the previous period; AI provider insights (answers, share, speed, failures, last error); per-organization usage; usage CSV export.
- **Plans** (Free, Standard, Enterprise and your own): monthly questions, document and member caps, cloud AI allowed. Limits are enforced on upload and when adding members.
- **Organization lifecycle:** suspend with a reason; scheduled deletion with a 30-day grace period (`purge_organizations` command); data export as a zip.
- **Support mode:** view an organization as an admin would, read-only, with a banner and audit entries.
- **Security centre:** everyone with console access and their two-factor status, failed sign-ins and pauses, every API key flagged when never used, unused or due for rotation.
- **Two-factor sign-in** (authenticator app, RFC 6238) with backup codes; can be required for the console; `reset_2fa` command.
- **Audit log:** filters, before/after view of changes, server paging and CSV export. Failed sign-ins are recorded.
- Destructive actions ask you to type the name to confirm.

## [3.3.0] - 2026-10-05

### Added

- **Docker deployment with pgvector.** `docker compose up -d --build` runs the app and PostgreSQL 16 with pgvector 0.8. The embedding model is built into the image. See [DEPLOYMENT.md](DEPLOYMENT.md).
- **pgvector vector backend** (`RAG_VECTOR_BACKEND=pgvector`): passages live in the `documents_passage` table with an HNSW cosine index.
- Commands `setup_pgvector` and `reindex`. `whitenoise` serves static files when debug mode is off.
- Documentation set: user, admin and superadmin guides, AI providers, deployment, configuration, decisions; updated diagrams.

### Fixed

- A fresh database could fail its first migration because a platform-settings lookup ran before the table existed.

## [3.2.0] - 2026-10-04

### Added

- **Platform settings** page: sign-up, session length, sign-in lockout, defaults for new organizations, upload size, quoting fallback, answer reuse, announcement, maintenance mode. Changes apply without a restart and are audit-logged.
- **Sign-in lockout** after repeated wrong passwords; **maintenance mode**.
- **AI providers** managed in the portal: Gemini, OpenAI, Claude, Groq, DeepSeek, OpenRouter, Ollama or any OpenAI-compatible API, with encrypted keys, tests and an order.
- **Answer fallback chain:** next provider on failure, then quoting the best-matching sentences from the documents.
- Organization setting **Use external AI providers**.
- Tables page through results 10 rows at a time with search, sorting and page size.

### Fixed

- Gemini answers were cut off after about 20 words because hidden "thinking" tokens counted against the answer limit.

## [3.1.0] - 2026-10-03

### Added

- Organization **Settings** page in tabs: chat experience, answer style, AI model, retrieval and quality, usage.
- **Pause and resume documents**; confirmation dialogs on document actions.
- **Home dashboard** redesign: trends, activity by outcome, answer quality, busiest times, most-used documents.

### Changed

- Moved from SQLite to **PostgreSQL 16** for the local install.

## [3.0.0] - 2026-10-02

### Added

- The multi-tenant **Django platform** (`django_app/`): organizations, roles, hashed API keys, chat memory, page citations, background indexing, analytics, REST API (`/api/v1/`).
- Superadmin portal with organizations, users, system health and audit log.
- Question categories and topics, knowledge gaps, feedback, saved answers, access requests; customisable menu, Ctrl+K palette, Developer page.
- Speed work: model warm-up, reuse of answers to repeated questions, streaming answers.

### Changed

- Imported the Streamlit app's users, API keys and documents into the organization "ICTA" (`manage.py import_legacy`).

## [2.0.0] - 2025-10-08

The Streamlit app (`shit.py`) with login, roles, uploads, analytics and API keys, and the public FastAPI service (`shit_api.py`). Legacy: see [STREAMLIT_APP.md](STREAMLIT_APP.md).

## [1.0.0] - 2025-09-16

The first version: a FastAPI backend (`api.py`) and a Streamlit front end (`main.py`) without login.
