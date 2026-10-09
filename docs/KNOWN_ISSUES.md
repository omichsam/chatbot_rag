# Known issues and roadmap

Current limitations of the platform, what to do about them, and what to build next. Reviewed 2026-10-05. When you find or fix something, update this page and [CHANGELOG.md](CHANGELOG.md).

## Current limitations

| # | Limitation | Impact | Workaround / fix |
| --- | --- | --- | --- |
| L1 | **Caches live in each process's memory** (sign-in failure counts, platform settings, reused answers) | Fine with one container. With several workers or servers, lockouts aren't shared and settings take up to 30 s to reach every process | Add a shared cache (Redis, or Django's database cache) before scaling out |
| L2 | **Indexing runs in threads inside the web process** | A restart mid-upload leaves the document *Processing* | Click *Re-index*, or `manage.py reindex`. Long term: a job queue (django-q, Celery) |
| L3 | **Scanned PDFs can't be read** (no OCR) | They show *Failed: No text could be extracted* | Run OCR (e.g. `ocrmypdf`) before uploading; add OCR to indexing |
| L4 | **Only PDF, TXT and Markdown** | Word, Excel, PowerPoint and web pages must be converted first | Add loaders in `rag/loaders.py` |
| L5 | **No email** | No invitations or self-service password reset; admins and superadmins set passwords | Add `EMAIL_*` settings and the invite/reset flows |
| L6 | **AI providers move on immediately on failure** | A brief blip skips to the backup instead of retrying | Add one short retry for 429/5xx before moving on |
| L7 | **Gemini uses a deprecated SDK** (`google-generativeai`) | Works today; will stop receiving updates | Move `rag/llm.py`'s Gemini client to `google-genai` |
| L8 | **Two Gemini providers with different keys** could pick up each other's key under concurrent load (the SDK's key is global) | Only matters with two Gemini rows using different keys | Use one Gemini provider, or move to `google-genai` (per-client keys) |
| L9 | **Usage is counted in questions, not tokens or cost** | No per-provider cost reporting or billing | Record token usage per message |
| L10 | **pgvector search filters after the index scan** | With very many paused documents or organizations, a search can return fewer than 5 passages | Turn on pgvector 0.8 iterative scans (`SET hnsw.iterative_scan = relaxed_order`) when the data grows |
| L11 | **Local Windows PostgreSQL has no pgvector**, so local development uses Chroma | Both vector backends must keep working (tests cover both, the pgvector ones inside Docker) | Run development in Docker, or install pgvector locally |
| L12 | **My history isn't paged** | Very long histories make a long page | Page it like the other lists |
| L13 | **Not under version control and no CI** | No history of changes, no automatic test runs | `git init`, a `.gitignore` (secrets, `media/`, `vector_store/`), and a CI job running the tests in Docker |

## Security follow-ups

- **Rotate exposed credentials:** the `ICTA_CHAT` API key and the superadmin password were shown in chat during development. Revoke and replace the key (*API keys*), and change the password (`changepassword platform-admin`).
- **Local PostgreSQL accepts passwordless local logins** (`trust` in `pg_hba.conf`). Fine for a laptop; switch to `scram-sha-256` anywhere shared.
- **Production checklist:** HTTPS in front, `SESSION_COOKIE_SECURE=true`, `CSRF_COOKIE_SECURE=true`, a strong `DJANGO_SECRET_KEY` kept stable, the database port not published, backups scheduled. See [DEPLOYMENT.md](DEPLOYMENT.md#going-to-production).

## Roadmap

1. **Version control and CI** (L13): protects everything else.
2. **Security follow-ups** above.
3. **Shared cache** (L1) and a **job queue** (L2), before running more than one web process.
4. **OCR and more file types** (L3, L4).
5. **Email invitations and password reset** (L5).
6. **Answer quality:** a set of test questions with known answers, to measure changes; hybrid keyword + meaning search with re-ranking for names and codes.
7. **Usage and cost per organization and provider** (L9).
8. **Provider retry** (L6) and the `google-genai` move (L7, L8).
9. **Kiswahili interface.**

## Legacy (Streamlit and FastAPI apps)

Found in the original apps (`shit.py`, `shit_api.py`, `api.py`, `main.py`). None were fixed there; the Django platform resolves all of them (see [DJANGO_PLATFORM.md](DJANGO_PLATFORM.md#what-changed-compared-to-streamlit)). Kept for reference.

### Security

| # | Issue | Where | Recommended fix |
| --- | --- | --- | --- |
| S1 | Default admin **`admin` / `admin123`** is created automatically whenever `users.json` is empty | `shit.py` (end of file) | Read the initial admin password from `.env`, or force a password change on first login |
| S2 | Passwords hashed with **unsalted SHA-256** | `shit.py` `hash_password` | Use `bcrypt` / `argon2` / `passlib` |
| S3 | Anyone can self-register from the login page | `shit.py` register form | Admin-only registration, or an invite code |
| S4 | API keys stored in plain text in `api_keys.json` | `shit.py` | Store a hash of each key and show the key only once |
| S5 | `GET /api/generate-key` needs no auth | `shit_api.py` | Remove it or protect it with an admin key |
| S6 | CORS `allow_origins=["*"]` together with `allow_credentials=True` | both APIs | List the allowed origins explicitly |
| S7 | User and model text rendered with `unsafe_allow_html=True`, so HTML/script can be injected into the chat view | `main.py`, `shit.py` chat rendering | Render with `st.chat_message` / `st.markdown` without unsafe HTML, or escape the text |
| S8 | Error responses include raw exception text | `shit_api.py` handlers | Log the details server-side and return a generic message |
| S9 | v1 API has no auth at all, including upload and delete-all | `api.py` | Retire v1, or add auth |
| S10 | Secrets (`.env`, `users.json`, `api_keys.json`) sit in the project folder with no `.gitignore` | root | Add a `.gitignore` before putting the project under git |

### Functional bugs

| # | Issue | Where | Fix |
| --- | --- | --- | --- |
| B1 | **API keys created in the UI don't work with the API.** `shit_api.py` only checks the `API_KEYS` env var, and revoking in the UI has no effect | `shit_api.py` `get_api_key` | Validate against `api_keys.json` (check `is_active`), sharing the logic with `shit.py` |
| B2 | **Both Gemini models in use are retired.** `gemini-pro` (v1) and `gemini-2.0-flash` (v2) now return 404, so every chat answer is an error message (confirmed 2026-10-01) | `api.py`, `shit.py`, `shit_api.py` `ask_gemini` | Use `gemini-flash-latest` (an alias that follows the current model), or move the model name into `.env`. Already done in `django_app/`. |
| B3 | Deleting a file in v1 leaves its chunks in Chroma | `api.py` `delete_file` | Delete by metadata (`vectorstore.delete(where={"file_id": ...})`) |
| B4 | `documents_count` in health is always 0 or 1 | `shit_api.py` `get_documents_count` | Use `vectorstore._collection.count()` |
| B5 | `static/index.html` documents `GET /api/analytics`, but no such endpoint exists | `shit_api.py` | Implement it (admin-only) or remove it from the page |
| B6 | "Clear All Files" puts a confirmation checkbox inside `if st.button(...)`. Ticking the checkbox reruns the script, the button reads False again, and nothing is deleted | `main.py`, `shit.py` Files page | Show the checkbox first, then enable the button when it's ticked (or use a form) |
| B7 | Re-uploading a file with the same name adds duplicate chunks | `shit.py` upload | Delete existing chunks for that filename before adding new ones |
| B8 | `session_id` doesn't give conversation memory, so follow-up questions lose context | all chat paths | Keep a per-session history and include the last few turns in the prompt |
| B9 | "Has documents" check runs a real embedding search for `"test"` on every chat | all | Use a collection count |
| B10 | v1 and v2 use different filename schemes in the shared `uploaded_files/`, so listings are confusing | `api.py` vs `shit.py` | Retire v1 |
| B12 | API Keys page "Copy" button prints the full key on screen instead of copying it | `shit.py` API Keys page | Remove the button, or use `st.code(key)`, which has a built-in copy icon |
| B13 | Register and Create User accept an empty username or password | `shit.py` | Validate before calling `register_user` |
| B11 | `shit_api.py` opens the vector store once at startup and doesn't notice when `shit.py` wipes and recreates `chroma_db/` | `shit_api.py` | Re-open per request, or run the store in client/server mode |

### Maintenance / housekeeping

- **`requirements.txt` is out of date** and missing `pypdf`, `pandas` and `matplotlib`. Regenerate it with `pip freeze`. See [STREAMLIT_APP.md](STREAMLIT_APP.md#setup-and-operations-legacy).
- **`Dockerfile` is empty** (just a UTF-16 BOM). A template is in the setup doc.
- **Deprecated imports:** `langchain.text_splitter` → `langchain_text_splitters`; `langchain_community` `Chroma` / `HuggingFaceEmbeddings` → `langchain_chroma` / `langchain_huggingface`. The newer packages are already installed in `venv`.
- **`google-generativeai` is deprecated** in favour of the `google-genai` SDK.
- **File names:** consider renaming `shit.py` → `app.py` and `shit_api.py` → `public_api.py`, and deleting `INDEXXX.HTML` (unrelated project) and `static/index copy.html`.
- **Duplicate code:** `load_file`, `process_file`, `ask_gemini` and the prompt appear in three files. Move them into a shared `rag_core.py`.
- **`.vscode/settings.json`** uses the deprecated `python.pythonPath`. Use `python.defaultInterpreterPath`.
- `analytics.json` grows forever. Add rotation, or move it and the other JSON files to SQLite.
- There are no tests.
