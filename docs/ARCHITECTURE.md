# Architecture

DocAssist is one Django application with a retrieval-augmented generation (RAG) pipeline. It finds the passages in an organization's documents that answer a question, then has an AI model reword them into a reply with citations. Nothing is trained on the documents. (The old Streamlit architecture is in [STREAMLIT_APP.md](STREAMLIT_APP.md#architecture-legacy).)

## The big picture

```text
 Browser (members, admins)    External apps (API key)       Superadmins (portal)
            │                          │                            │
            ▼                          ▼                            ▼
 ┌─────────────────────────── Django web app (gunicorn) ───────────────────────────┐
 │  tenants     documents      chat           api            portal                │
 │  orgs,roles  upload,index   chat, history  REST /api/v1/  platform settings,    │
 │  settings    pause,reindex  analytics,     keys           AI providers, health, │
 │  menu        lockout        gaps, feedback                audit                 │
 │                                   │                                             │
 │            rag/: loaders → chunking → embeddings → vectorstore → pipeline       │
 └──────────────────┬───────────────────────┬──────────────────────┬───────────────┘
                    ▼                       ▼                      ▼
     PostgreSQL 16 + pgvector       uploaded files          AI providers (HTTPS)
     records + passage vectors      /data/media             Gemini, OpenAI, Claude, Groq,
     (or SQLite + Chroma locally)                           DeepSeek, OpenRouter, Ollama
```

Everything except the AI providers runs on your server. The embedding model runs locally on the CPU.

## Components

| Part | Code | Responsibility |
| --- | --- | --- |
| Tenants | `tenants/` | Organizations, memberships and roles, the current-organization middleware, organization Settings, the menu, access requests, sign-in with lockout |
| Documents | `documents/` | Uploads, duplicate detection (SHA-256), background indexing, pause/resume, the `Passage` model for pgvector, `reindex` and `setup_pgvector` commands |
| Chat | `chat/` | Chat sessions and messages, streaming, history, saved answers, feedback, knowledge gaps, question categories and topics, the Home dashboard, analytics, the Questions explorer |
| API | `api/` | Hashed API keys, the REST API, the Developer playground |
| Portal | `portal/` | Superadmin pages: organizations, users, platform settings, AI providers, system health, audit log, maintenance mode |
| RAG | `rag/` | Plain Python: text extraction, chunking, embeddings, vector store, prompt building, the AI provider clients, the answer chain, quoting, the answer cache, warm-up |
| Templates | `templates/` | Server-rendered pages, a hand-written CSS design system (light and dark), small vanilla JavaScript (streaming, Ctrl+K, dialogs, table paging) |

## How a document becomes searchable

1. **Upload:** type and size are checked; a SHA-256 checksum spots duplicates; the file is saved under `media/documents/org_<id>/`; a `Document` row is created as *Queued*.
2. **Background indexing** (a thread pool, 2 workers): `pypdf` reads text page by page (TXT/MD are decoded with fallbacks) → the text is split into passages of ~1,000 characters overlapping by 200, each remembering its page → `all-MiniLM-L6-v2` turns each passage into 384 numbers → the passages are stored in the vector store.
3. The status becomes *Ready* (or *Failed* with the reason).

## How a question is answered

1. **Who and where:** the session (web) or API key (API) decides the organization. The monthly limit is checked.
2. **Context:** the last few question-and-answer turns are loaded (organization setting), and the question is labelled with a type and topic.
3. **Reuse:** a first question identical to a recent one, with unchanged documents and settings, is answered from the cache.
4. **Retrieve:** the question (plus the previous one, for follow-ups) is embedded the same way, and the closest passages (5 by default) are found in this organization's passages only, skipping paused documents.
5. **Prompt:** the organization's instructions and answer rules (scope, length, language, citations), the numbered passages with file and page, the conversation, and the question.
6. **Write:** the enabled AI providers are tried in order ([AI_PROVIDERS.md](AI_PROVIDERS.md)); if all fail, the best-matching sentences are quoted from the passages instead.
7. **Respond and record:** the web app streams the answer word by word; the API returns it whole. The question, answer, sources, timing, match score and which provider answered are stored.

Low match scores mark the answer as a **knowledge gap** for admins.

## Storage

| What | Where |
| --- | --- |
| Organizations, users, documents list, chats, ratings, keys, settings, AI providers, audit log | PostgreSQL 16 (SQLite also works for development) |
| Passage text and vectors | `RAG_VECTOR_BACKEND=pgvector`: the `documents_passage` table with an HNSW cosine index. `chroma`: a Chroma folder with one collection per organization |
| Uploaded files | `MEDIA_ROOT` (`/data/media` in Docker); never served over HTTP |
| Cached answers, platform settings, sign-in failure counts | Django's in-process memory cache |
| The embedding model | Hugging Face cache (built into the Docker image) |

## Isolation and security

- **Tenant isolation:** every query filters by the current organization; object lookups use the organization too, so another tenant's IDs return 404. Vector searches are filtered by organization (pgvector) or use the organization's own collection (Chroma).
- **Accounts:** Django sessions and PBKDF2 passwords; sign-in pauses after repeated failures; roles enforced by decorators; superadmin-only portal.
- **Secrets:** API keys stored as SHA-256 hashes; AI provider keys encrypted (Fernet, from `DJANGO_SECRET_KEY`); secrets live in `.env` files.
- **Web:** CSRF protection, template escaping, and answers rendered as Markdown through DOMPurify.
- **Data leaving the server:** only the prompt for one question (instructions, a few passages, recent turns, the question) goes to a cloud AI provider, and none does when an organization turns external AI off.

## Performance

| Technique | Effect |
| --- | --- |
| Model warm-up at start | The first question doesn't wait ~20 s for the embedding model |
| Background indexing | Uploads return immediately |
| Answer reuse (24 h by default) | Repeated questions are instant and use no AI quota |
| Streaming | Text appears within a second or two |
| One gunicorn worker, 8 threads | The embedding model is loaded once per container |

## Where to look in the code

| Topic | File |
| --- | --- |
| Answer chain, prompt | `rag/pipeline.py` |
| Provider clients | `rag/llm.py` |
| Quoted answers | `rag/extractive.py` |
| Vector backends | `rag/vectorstore.py` |
| Indexing | `documents/services.py` |
| Chat services (ask, stream, cache, fallback recording) | `chat/services.py` |
| Platform settings, AI providers | `portal/models.py`, `portal/ai.py`, `portal/forms.py` |
| Settings by environment | `config/settings.py` |
