# Developer setup and operations

How to run DocAssist on your own machine for development, run the tests, and use the management commands. For production, use Docker: see [DEPLOYMENT.md](DEPLOYMENT.md). The old Streamlit app's setup is in [STREAMLIT_APP.md](STREAMLIT_APP.md#setup-and-operations-legacy).

## Prerequisites

- Python **3.10** (the project's `venv` is 3.10.11). Windows is the tested platform; Linux and macOS work with the usual path changes.
- Optional: PostgreSQL 16 (otherwise SQLite is used), Docker Desktop (for the production-like setup with pgvector).
- A Gemini API key for real answers. Without one, answers fall back to quoting the documents.

## Install

```powershell
cd C:\Users\sam\Desktop\rag_bot
python -m venv venv                      # skip if venv/ exists
venv\Scripts\activate
pip install -r django_app\requirements.txt
```

The first run downloads the embedding model (about 90 MB).

## Configure

Copy `django_app/.env.example` to `django_app/.env` and fill it in. The Gemini key can stay in `rag_bot/.env`, which is read too. Every variable is described in [CONFIGURATION.md](CONFIGURATION.md).

- **SQLite (simplest):** leave `POSTGRES_DB` empty.
- **PostgreSQL:** set `POSTGRES_DB`, `POSTGRES_USER`, `POSTGRES_PASSWORD`. This machine uses PostgreSQL 16 on port 5432 with a `docassist` database and user.
- **Vectors:** locally `RAG_VECTOR_BACKEND=chroma` (a `vector_store/` folder). pgvector needs PostgreSQL with the `vector` extension, which the Windows PostgreSQL install doesn't include; use Docker for it.

## Run

```powershell
cd django_app
..\venv\Scripts\python.exe manage.py migrate
..\venv\Scripts\python.exe manage.py createsuperuser        # first time
..\venv\Scripts\python.exe manage.py runserver 127.0.0.1:8123
```

Open <http://127.0.0.1:8123>. Port 8123 is used because 8000 belongs to another local project and the Docker app.

## Tests

```powershell
..\venv\Scripts\python.exe manage.py test --noinput
```

- 133 tests. Locally 131 run and the 2 pgvector tests skip (no extension); inside Docker all 133 run: `docker compose exec web python manage.py test`.
- They run offline: fake embeddings, an "echo" AI writer, an in-memory Chroma per test, and mocked HTTP for the AI provider clients.
- `--noinput` replaces a test database left over from an interrupted run.
- A test that changes platform settings should use `rag.testing.platform_settings(...)`, which restores them afterwards.

## Management commands

| Command | Does |
| --- | --- |
| `reindex [--organization ID] [--missing]` | Re-reads, re-embeds and re-stores documents' passages in the current vector backend. Use after switching backend or changing the embedding/chunk settings. |
| `setup_pgvector` | Creates the `vector` extension and the passage table if missing (Docker runs it on every start) |
| `import_legacy --org-name NAME [--source DIR] [--skip-documents]` | Imports the Streamlit app's users, API keys and documents into an organization |
| `purge_organizations` | Deletes organizations whose 30-day scheduled-deletion period has ended (run daily from cron in production) |
| `seed_test_users [--password P]` | Creates or resets one test account per role and a Test Organization (test environments only; see [SERVER_SETUP.md](SERVER_SETUP.md)) |
| `reset_2fa USERNAME` | Removes someone's authenticator app so they can sign in with a password and set it up again |
| `changepassword USERNAME`, `createsuperuser` | Django's standard account commands |

## Code checks

```powershell
..\venv\Scripts\python.exe -m pyflakes rag chat documents tenants portal api
..\venv\Scripts\python.exe manage.py makemigrations --check --dry-run
```

## Troubleshooting

| Symptom | Cause / fix |
| --- | --- |
| Answers say "Quoted directly from your documents" | No AI provider could answer (missing key, quota). Check **Platform → AI providers** and run Test. |
| "Using backup" in the sidebar | The first AI provider failed and a later one answered. See Home → Needs attention for the reason. |
| Very slow first question after a start | The embedding model is loading (about 20 s on this laptop); later questions are fast |
| A document shows *Failed: No text could be extracted* | A scanned PDF without a text layer. Run OCR on it first. |
| Test run asks "Type 'yes' to delete the test database" | A previous run was interrupted. Add `--noinput`. |
| `setup_pgvector` says the extension isn't available | The database has no pgvector. Use the Docker setup, or keep `RAG_VECTOR_BACKEND=chroma`. |
| Sign-in says "Too many failed sign-ins" | The lockout is working. Wait for the pause, or restart the dev server to clear it (counts live in memory). |
| `docker compose exec … /tmp/x` fails on Windows | Git Bash rewrote the path. Prefix with `MSYS_NO_PATHCONV=1`. |
