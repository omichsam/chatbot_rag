# DocAssist — Document Q&A Assistant

DocAssist answers questions from an organization's own documents and shows the file and page each answer came from. It finds the relevant passages itself (with a local embedding model and vector search), then has an AI model reword them into a reply. If every AI provider fails, it quotes the best-matching sentences instead, so people always get a sourced answer.

The current platform is the multi-tenant Django app in [`django_app/`](django_app/): organizations, roles, a superadmin portal, any AI provider (Gemini, OpenAI, Claude, Groq, DeepSeek, OpenRouter, Ollama), PostgreSQL with pgvector, and Docker.

## Quick start

**Docker (recommended)**, PostgreSQL 16 + pgvector included:

```bash
cd django_app
cp .env.docker.example .env.docker      # set DJANGO_SECRET_KEY, POSTGRES_PASSWORD, GOOGLE_API_KEY
docker compose up -d --build            # → http://localhost:8001
docker compose exec web python manage.py createsuperuser
```

**Local development:**

```powershell
venv\Scripts\activate
cd django_app
python manage.py migrate
python manage.py runserver 127.0.0.1:8123   # → http://127.0.0.1:8123
```

## Documentation

Everything is in [`docs/`](docs/README.md), organized by reader:

- **Using it:** [User guide](docs/USER_GUIDE.md) · [Admin guide](docs/ADMIN_GUIDE.md) · [Superadmin guide](docs/SUPERADMIN_GUIDE.md) · [AI providers](docs/AI_PROVIDERS.md)
- **Running it:** [Deployment](docs/DEPLOYMENT.md) · [Configuration](docs/CONFIGURATION.md) · [Developer setup](docs/SETUP_AND_OPERATIONS.md)
- **Understanding it:** [Architecture](docs/ARCHITECTURE.md) · [Platform reference](docs/DJANGO_PLATFORM.md) · [Diagrams](docs/PROCESS_DIAGRAMS.md) · [Decisions](docs/DECISIONS.md) · [API](docs/API_REFERENCE.md)
- **Tracking it:** [Changelog](docs/CHANGELOG.md) · [Known issues and roadmap](docs/KNOWN_ISSUES.md) · [Contributing and releases](CONTRIBUTING.md)

**Version:** see [`django_app/VERSION`](django_app/VERSION) and the git tags (`vX.Y.Z`). Maintained by **omichsam**.

## What's in this folder

| Path | What it is |
| --- | --- |
| `django_app/` | **The platform.** Django project, Dockerfile, docker-compose.yml, tests (`python manage.py test`) |
| `docs/` | Documentation and diagrams |
| `venv/` | Python 3.10 virtual environment for local development |
| `.env` | `GOOGLE_API_KEY` for local runs. Secret; never share or commit it. |
| `shit.py`, `shit_api.py`, `static/` | *Legacy* v2: the original Streamlit app and its public API. See [docs/STREAMLIT_APP.md](docs/STREAMLIT_APP.md). Their Gemini model is retired, so chat returns errors. |
| `api.py`, `main.py` | *Legacy* v1: FastAPI backend and Streamlit UI, no login |
| `uploaded_files/`, `chroma_db/`, `users.json`, `api_keys.json`, `analytics.json` | *Legacy* data, already imported into the platform (organization "ICTA") |
| `requirements.txt`, `Dockerfile`, `docker-compose.yml` | *Legacy* dependency pins and Docker setup for the Streamlit/FastAPI apps; the platform uses the ones in `django_app/` |
| `INDEXXX.HTML`, `static/index copy.html` | Unrelated or unused leftovers |
