# Streamlit App (`shit.py`)

> **Legacy.** This describes the original Streamlit app, which has been replaced by the Django platform (see [docs/README.md](README.md)). It's kept for reference; nothing here applies to the platform.

`shit.py` is the main, working application. It is a single-file Streamlit app that does everything on its own: login, document upload and indexing, chat, analytics, API-key management and user administration. It does **not** need `api.py` or `shit_api.py` to be running.

```powershell
venv\Scripts\activate
streamlit run shit.py        # → http://localhost:8501
```

Always run it from the project root. It reads and writes `.env`, `uploaded_files/`, `chroma_db/`, `users.json`, `api_keys.json` and `analytics.json` using relative paths.

---

## How the script runs

Streamlit re-executes the whole file from top to bottom on every click or input. `shit.py` is laid out in this order:

| Lines (approx.) | Section | What happens |
| --- | --- | --- |
| 1–47 | Imports & setup | Loads `.env`, creates `uploaded_files/` and `chroma_db/`, loads the embedding model, opens Chroma (cached with `@st.cache_resource`), configures Gemini |
| 49–119 | Page config & CSS | Title "Document Assistant", wide layout, custom CSS for chat bubbles, file cards and the API-key box |
| 121–265 | Auth, keys & analytics helpers | Read/write the JSON files; hash passwords; create/revoke keys; `track_query` |
| 267–283 | Session state | Default values (see below) |
| 285–438 | RAG helpers | `load_file`, `ask_gemini`, `process_file`, `get_files_list`, `delete_single_file`, `clear_all_files`, `chat_with_docs` |
| 440–499 | Login / register screen | Shown when not logged in; ends with `st.stop()` so nothing below runs |
| 501–554 | Sidebar | Welcome, role badge, navigation, document & Gemini status, Logout |
| 556–984 | Pages | One `if/elif` branch per page |
| 986–1034 | Footer messages | Welcome tips on an empty Chat page; a sidebar warning if `GOOGLE_API_KEY` is missing |
| 1036–1039 | Bootstrap | Creates `admin` / `admin123` if `users.json` is empty |

### Session state

| Key | Default | Meaning |
| --- | --- | --- |
| `authenticated` | `False` | Logged in or not |
| `username` | `None` | Current user |
| `show_login` / `show_register` | `True` / `False` | Which auth form is showing |
| `page` | `"chat"` | Current page |
| `messages` | `[]` | Chat history for this browser tab: `{role, content, sources}` |
| `session_id` | Unix timestamp | Shown in Settings. Not used by the model |
| `uploaded_files` | `{}` | Names already uploaded in this tab, so a Streamlit rerun doesn't upload the same file twice |

Session state lives per browser tab and in memory only. Refreshing the page logs the user out and clears the chat.

---

## Pages

Navigation is a sidebar radio. Regular users see **Chat, API Keys, Settings**. Admins also see **Files, Analytics, Admin**.

### 💬 Chat (everyone)
- **Admins** get an uploader for PDF and TXT files. The upload flow:
  1. The file is saved to `uploaded_files/<original name>`.
  2. `process_file` loads it (PyPDF, or TextLoader with fallback encodings) and splits it into 1000-character chunks with 200 characters of overlap.
  3. Each chunk is tagged with `{"filename": ...}` and added to Chroma.
  4. If processing fails, the saved file is deleted again.
- **Users** see a note that only admins can upload.
- Asking a question calls `chat_with_docs`, which runs these steps:
  1. Pulls the top 5 matching chunks from Chroma.
  2. Builds the prompt and sends it to Gemini `gemini-2.0-flash`.
  3. Shows the answer with a "Sources: …" caption.
  4. Records the question with `track_query` in `analytics.json`.

### 📁 Files (admin)
- Lists each document with its size and upload time.
- **Delete <file>** removes the file, wipes `chroma_db/` and re-indexes every remaining file. This is slow when there are many documents.
- **Clear All Files** is meant to delete everything after a confirmation checkbox, but the checkbox currently never takes effect (see Known issues).

### 📊 Analytics (admin)
Built from `analytics.json` with pandas and matplotlib:
- Metric tiles: Total Queries, Active Users, Most Active User, Unique Questions
- Top 10 most-asked questions (compared in lower case)
- Daily query line chart
- Per-user table: query count and last active time
- The last 10 queries, each expandable to show the question, the first 200 characters of the answer, and the sources

### 🔑 API Keys (everyone)
- **Generate API Key** creates `sk_<random>` in `api_keys.json` under the current user, then shows it once.
- The list shows each of your keys with its status. **Revoke** sets `is_active: false`.
- **Copy** doesn't copy anything. It prints the full key on the page.
- These keys are **not** accepted by `shit_api.py` unless they are also added to `API_KEYS` in `.env`.

### ⚙️ Settings (everyone)
- System info: document count, whether the vector DB exists, and whether the Gemini key is set
- Session ID, plus a **New Session** button that clears the chat
- Your username and role
- **Change Password**: checks the current password, then saves the new hash

### 👑 Admin (admin)
Three tabs:
- **User Management:** list all users. For anyone other than yourself you can **Toggle Admin** or **Delete User**. You can also **Create New User**, with an optional admin flag.
- **API Key Management:** every user's keys, with **Toggle Active** and **Delete** buttons.
- **System Stats:** total users, total and active API keys, number of uploaded documents, and total storage in MB.

---

## Everyday tasks

| Task | Where |
| --- | --- |
| First login | `admin` / `admin123` (only created when `users.json` is empty), then change the password in Settings |
| Add a user | Admin → User Management → Create New User, or have them self-register on the login page |
| Make someone an admin | Admin → User Management → Toggle Admin |
| Add a document | Chat page (as admin) → uploader |
| Replace a document | Files → Delete the old one, then upload the new one (re-uploading without deleting duplicates its chunks) |
| See who is asking what | Analytics |
| Reset a forgotten password | No UI for this. An admin can delete the user and recreate them, or edit `password_hash` in `users.json` (SHA-256 hex of the new password) |

---

## Known issues specific to the Streamlit app

Full list with fixes: [KNOWN_ISSUES.md](KNOWN_ISSUES.md).

- **Clear All Files never runs.** The confirmation checkbox sits inside `if st.button(...)`. Ticking it reruns the script, the button then reads False, and the delete code is skipped.
- **Default admin:** `admin` / `admin123` is recreated whenever `users.json` is empty.
- Passwords use unsalted SHA-256, and anyone can register from the login page.
- The Create User form (Admin page) and the registration form both accept an empty username or password.
- Chat messages and file names are rendered with `unsafe_allow_html=True`, so HTML typed into the chat is rendered as HTML.
- The "Copy" button prints the key on screen, even though the app says the key won't be shown again.
- There is no conversation memory. Each question is answered independently.
- Refreshing the browser logs you out, because login state is kept only in session state.
- Heavy work (upload indexing and delete-and-reindex) runs in the request and blocks the UI until it finishes.

## Architecture (legacy)

*Moved here from ARCHITECTURE.md on 2026-10-05.*

### High-level picture

```text
                ┌───────────────────────────── v2 ─────────────────────────────┐
                │                                                              │
 Admin / user ──► shit.py (Streamlit :8501)                                     │
                │   • login / register (users.json)                            │
                │   • upload + index docs ──► uploaded_files/  ──┐              │
                │   • chat                                       ▼              │
                │   • analytics (analytics.json)          chroma_db/ (Chroma)   │
                │   • API keys (api_keys.json)                   ▲              │
                │                                                │ read only    │
 Browser / ─────► shit_api.py (FastAPI :8006)  ─────────────────┘              │
 3rd-party app  │   • GET /            → static/index.html (web chat)          │
                │   • POST /api/chat   (Bearer key from env API_KEYS)          │
                └──────────────────────────────────────────────────────────────┘
                                     │
                                     ▼
                         Google Gemini (gemini-2.0-flash)

                ┌───────────────────────────── v1 ─────────────────────────────┐
 User ──────────► main.py (Streamlit :8501) ──HTTP──► api.py (FastAPI :8006)    │
                │                                     • upload / chat / files  │
                │                                     • Gemini "gemini-pro"    │
                └──────────────────────────────────────────────────────────────┘
```

`api.py` and `shit_api.py` both listen on port **8006**, so only one can run at a time.

### Components

#### Embeddings
All entry points use `HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")`. The model runs locally on CPU and produces 384-dimension vectors. It is downloaded from Hugging Face on first run (~90 MB) and cached under `~/.cache/huggingface`.

#### Vector store
`langchain_community.vectorstores.Chroma` persisted to `chroma_db/`, using the default collection. `chroma.sqlite3` holds documents and metadata. The UUID folder holds the HNSW index files.

Chunk metadata written by each app:

| Writer | Metadata |
| --- | --- |
| `shit.py` | `{"filename": "<original name>"}` |
| `api.py` | `{"filename": "<original name>", "file_id": "<uuid>_<name>"}` |

#### LLM
Both files use `google.generativeai`, configured from `GOOGLE_API_KEY`.

- `shit.py`, `shit_api.py` → `gemini-2.0-flash`
- `api.py` → `gemini-pro` (**retired by Google**; returns 404. See [KNOWN_ISSUES](KNOWN_ISSUES.md#legacy-streamlit-and-fastapi-apps).)

### The RAG pipeline

#### 1. Ingestion (upload)
1. File saved to `uploaded_files/`.
   - `shit.py` saves it under its original filename.
   - `api.py` saves it as `<uuid4>_<filename>`.
2. Loaded with `PyPDFLoader` (one Document per page) for `.pdf`. Otherwise `TextLoader` is used, trying the encodings `utf-8`, `latin-1`, `cp1252` and `iso-8859-1` in turn.
3. Split with `RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)` (characters, not tokens).
4. Metadata replaced with the filename (see table above).
5. `vectorstore.add_documents(chunks)`: embeds and persists.

#### 2. Retrieval + generation (chat)
1. **Empty check:** `similarity_search("test", k=1)`. If it returns nothing, the app replies "no documents yet".
2. `similarity_search(question, k=5)` returns the top 5 chunks by embedding similarity.
3. The chunks are joined with blank lines into `context`. `sources` is the set of distinct filenames among them.
4. A prompt template is filled with the context and question and sent to Gemini in a single call. It tells the model to answer from the context, to say so politely if the answer isn't there, and to be conversational.
5. The answer text and `sources` are returned.

There is **no conversation memory**. `session_id` is accepted and echoed back but never used to include earlier turns, so every question is answered on its own.

#### 3. Deletion
- **`shit.py`**: deletes the file, wipes `chroma_db/`, then **re-indexes every remaining file**. This is correct but slow for large collections.
- **`api.py`**: deletes the file and re-opens Chroma. **Its chunks stay in the index** (bug).
- **Clear all** (both): deletes all files and `rmtree`s `chroma_db/`.

### v2 application layer (`shit.py`)

A single-file Streamlit script. It reruns top to bottom on every interaction.

**Session state:** `authenticated`, `username`, `page`, `messages`, `show_login` / `show_register`.

**Pages** (sidebar radio):

| Page | Who | What |
| --- | --- | --- |
| 💬 Chat | all | Chat with docs. Admins also get an uploader here. |
| 📁 Files | admin | List, delete and clear documents |
| 📊 Analytics | admin | Query counts, per-user activity, popular questions, hourly/daily charts (pandas + matplotlib) |
| 🔑 API Keys | all | Create and revoke your own `sk_…` keys |
| ⚙️ Settings | all | Account info and API usage examples |
| 👑 Admin | admin | Manage users (toggle admin, delete, create) and all API keys |

**Auth:** passwords are stored as unsalted SHA-256 hex in `users.json` under `password_hash`. Login compares the hashes.

**Analytics:** every chat calls `track_query()`, which appends to `analytics.json`.

### Data file formats

**users.json**
```json
{ "<username>": { "password_hash": "<sha256 hex>", "is_admin": false, "created_at": "<iso>" } }
```

**api_keys.json**
```json
{ "sk_<token>": { "username": "sam", "name": "ICTA_CHAT", "created_at": "<iso>", "is_active": true } }
```

**analytics.json**
```json
{
  "queries": [ { "username", "question", "answer": "<first 200 chars>", "sources": [], "timestamp" } ],
  "users":   { "<username>": { "query_count": 0, "last_active": "<iso>" } },
  "popular_questions": { "<lower-cased question>": 3 }
}
```

> ⚠️ Keys created in the Streamlit **API Keys** page are stored in `api_keys.json`, but `shit_api.py` does **not** read that file. It only accepts keys listed in the `API_KEYS` environment variable. See [KNOWN_ISSUES](KNOWN_ISSUES.md#legacy-streamlit-and-fastapi-apps).

### Web chat page (`static/index.html`)

A self-contained HTML/CSS/JS page served by `shit_api.py` at `/`. It uses Font Awesome from cdnjs. `API_BASE = '/api'`.

- It asks for an API key, checks it with `GET /api/validate-key`, and saves it in `localStorage['apiKey']`.
- It sends questions to `POST /api/chat` with `Authorization: Bearer <key>` and keeps the returned `session_id`.
- It polls `GET /api/health` for status.
- Its built-in docs mention `GET /api/analytics`, which is **not implemented** in `shit_api.py`.

## Setup and operations (legacy)

*Moved here from SETUP_AND_OPERATIONS.md on 2026-10-05. Applies only to `shit.py`, `shit_api.py`, `api.py` and `main.py`.*

**Install:** `python -m venv venv`, then `pip install fastapi uvicorn streamlit langchain langchain-community langchain-text-splitters chromadb sentence-transformers google-generativeai python-dotenv aiofiles pydantic requests pypdf pandas matplotlib`. The root `requirements.txt` is stale (late-2023 pins; missing `pypdf`, `pandas`, `matplotlib`). The `chroma_db/` folder was written by chromadb 1.x, so the old 0.4.x pin can't read it.

**Configure:** `.env` in the project root with `GOOGLE_API_KEY` and, for `shit_api.py` only, `API_KEYS` (comma-separated bearer tokens; empty means no authentication). Start the apps from the project root, because paths (`uploaded_files/`, `chroma_db/`, `users.json`, `api_keys.json`, `analytics.json`, `static/`) are relative.

**Run (v2):** `streamlit run shit.py` (http://localhost:8501) and `python shit_api.py` (http://localhost:8006, Swagger at `/api/docs`). **Run (v1):** `python api.py` then `streamlit run main.py`. v1's `api.py` and v2's `shit_api.py` both use port 8006, so only one can run.

**Operating:**
- First start with an empty `users.json` creates **admin / admin123**; replace it straight away.
- Uploading indexes during the upload; uploading the same file name again adds duplicate chunks.
- *Files → Delete* re-indexes everything that remains; *Clear All* wipes documents and index.
- API keys created in the UI must also be added to `API_KEYS` in `.env` (and the API restarted) before the API accepts them; revoking in the UI doesn't block the API.
- Back up `uploaded_files/`, `chroma_db/`, the three JSON files and `.env` together. `chroma_db/` can be rebuilt by re-uploading.

**Troubleshooting:** `404 models/gemini-pro` → you're on v1, whose model is retired. "Google API key not set" → `.env` missing from the working directory. API `401` with a UI key → the key isn't in `API_KEYS`. "Cannot connect to API server" (main.py) → start `api.py` first. `Address already in use` on 8006 → the other API is running.
