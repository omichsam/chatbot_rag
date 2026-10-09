# DocAssist: Processes and Diagrams

How a document becomes searchable knowledge, and how a question becomes a cited answer. The diagrams are written in [Mermaid](https://mermaid.js.org/). They render on GitHub and in editors with Mermaid support, and every one is also exported as an image in [`docs/diagrams/`](diagrams/).

*Updated 2026-10-05 for AI providers with fallback, pgvector and Docker.* After editing a diagram, regenerate the SVGs: the render script loads Mermaid in headless Chrome and writes one `diagrams/NN-name.svg` per block, in order.

**Contents**

1. [Use case diagram](#1-use-case-diagram)
2. [Component diagram](#2-component-diagram)
3. [Deployment diagram](#3-deployment-diagram)
4. [Activity diagram: uploading and indexing a document](#4-activity-diagram-uploading-and-indexing-a-document)
5. [State machine: document lifecycle](#5-state-machine-document-lifecycle)
6. [Sequence diagram: upload and indexing](#6-sequence-diagram-upload-and-indexing)
7. [Activity diagram: answering a question](#7-activity-diagram-answering-a-question)
8. [Sequence diagram: a question in the web app (streamed)](#8-sequence-diagram-a-question-in-the-web-app-streamed)
9. [Sequence diagram: a question through the API](#9-sequence-diagram-a-question-through-the-api)
10. [Class diagram: data model](#10-class-diagram-data-model)
11. [Data flow: what stays on the server and what leaves](#11-data-flow-what-stays-on-the-server-and-what-leaves)
12. [Technology summary](#12-technology-summary)

---

## 1. Use case diagram

Who interacts with the system, and what each actor can do.

```mermaid
flowchart LR
    Member(["Actor: Member"])
    Admin(["Actor: Owner / Admin"])
    Super(["Actor: Superadmin"])
    App(["Actor: External app<br/>(API key)"])

    subgraph DocAssist["DocAssist"]
        UC1(["Ask a question"])
        UC2(["View answer sources"])
        UC3(["Rate / save an answer"])
        UC4(["Browse documents"])
        UC5(["Upload / pause / re-index / delete documents"])
        UC6(["Review analytics, feedback and knowledge gaps"])
        UC7(["Manage members and API keys"])
        UC8(["Request admin access"])
        UC9(["Manage organizations, users,<br/>platform settings and AI providers"])
        UC10(["System health, maintenance mode, audit log"])
    end

    Member --- UC1 & UC2 & UC3 & UC4 & UC8
    Admin --- UC1 & UC5 & UC6 & UC7
    Super --- UC9 & UC10
    App --- UC1
```

- **Members** ask questions, see the sources behind each answer, rate or save answers, and can request admin access.
- **Owners / admins** do everything members do, and also maintain the knowledge base, review usage and manage people and keys for their organization.
- **Superadmins** operate the platform across all organizations: platform settings, AI providers and their order, maintenance mode, health and the audit log.
- **External apps** ask questions through the REST API, each authenticated by an organization's API key.

---

## 2. Component diagram

The main parts of the system and how they depend on each other.

```mermaid
flowchart TB
    subgraph Clients
        Browser["Web browser<br/>(HTML, CSS, JavaScript)"]
        Client["External app<br/>(any HTTP client)"]
    end

    subgraph Django["Django web application"]
        MW["Tenant middleware<br/>resolves user → organization → role"]
        Web["Web views<br/>chat, documents, insights, settings"]
        API["REST API<br/>/api/v1/…"]
        Portal["Platform portal<br/>superadmin"]
        Ingest["Ingestion service<br/>background workers"]
        Chat["Chat service<br/>history, cache, quota"]
        Cat["Categorizer<br/>type and topic"]
    end

    subgraph RAG["RAG core (rag/)"]
        Loader["Loader<br/>extract text per page"]
        Splitter["Chunker<br/>overlapping passages"]
        Embed["Embedder<br/>text → vector"]
        VS["Vector store adapter<br/>pgvector or Chroma"]
        Pipe["Pipeline<br/>retrieve → prompt → answer chain"]
        LLM["Provider clients<br/>Gemini SDK, OpenAI-compatible, Claude"]
        Quote["Quoter<br/>best sentences, last resort"]
    end

    DB[("PostgreSQL<br/>records + pgvector passages")]
    Vec[("Chroma folder<br/>local development only")]
    Files[("File storage")]
    Providers{{"AI providers, in order<br/>Gemini, OpenAI, Claude, Groq,<br/>DeepSeek, OpenRouter, Ollama"}}

    Browser --> MW --> Web
    Client --> API
    Web --> Chat
    API --> Chat
    Web --> Ingest
    Portal --> DB
    Ingest --> Loader --> Splitter --> Embed --> VS
    Chat --> Cat --> Embed
    Chat --> Pipe --> VS
    Pipe --> LLM --> Providers
    Pipe --> Quote --> Embed
    VS --> DB
    VS -.-> Vec
    Ingest --> Files
    Web & API & Chat & Ingest --> DB
```

| Component | Responsibility |
| --- | --- |
| Tenant middleware | Identifies the user, picks their current organization and role for every request |
| Web views / REST API | Accept requests, enforce permissions, return pages or JSON |
| Ingestion service | Turns an uploaded file into stored, searchable passages, in the background |
| Chat service | Applies quotas, gathers conversation history, uses the answer cache, stores questions and answers |
| Categorizer | Labels each question with a type and the closest topic |
| RAG core | Loader, chunker, embedder, vector store (pgvector or Chroma), pipeline, provider clients and quoter: the retrieval-augmented generation logic, independent of the web layer |
| Answer chain | Tries the enabled AI providers in order; if all fail, quotes the best-matching sentences |
| Platform portal | Cross-organization administration for superadmins: organizations, users, platform settings, AI providers, health, audit, maintenance |

---

## 3. Deployment diagram

What runs where, in the Docker setup.

```mermaid
flowchart LR
    subgraph UserDevice["User device"]
        B["Browser"]
    end
    subgraph ThirdParty["Third-party system"]
        X["App / website backend"]
    end

    subgraph Host["Docker host"]
        direction TB
        subgraph Web["Container: web"]
            WSGI["Django app<br/>gunicorn: 1 worker, 8 threads"]
            Workers["Indexing thread pool"]
            Model["Embedding model in memory<br/>(built into the image)"]
            WSGI --- Workers
            WSGI --- Model
        end
        subgraph Db["Container: db"]
            PG[("PostgreSQL 16 + pgvector<br/>records and passage vectors")]
        end
        VolA[("Volume appdata<br/>/data/media")]
        VolP[("Volume pgdata")]
        Workers --- VolA
        PG --- VolP
    end

    AI{{"AI providers (cloud)<br/>Gemini, OpenAI, Claude, …"}}
    Ollama["Optional container: Ollama<br/>local AI model"]

    B -- "HTTPS (via a reverse proxy)" --> WSGI
    X -- "HTTPS + API key" --> WSGI
    WSGI -- "SQL" --> PG
    WSGI -- "HTTPS: one question's prompt" --> AI
    WSGI -. "HTTP, stays on the host" .-> Ollama
```

- The **web** container holds the embedding model and runs migrations and `setup_pgvector` on start; a warm-up loads the model so the first question doesn't wait.
- The **db** container stores everything, including the passage vectors with an HNSW index, so one database backup covers it all.
- The only calls leaving the host go to cloud AI providers, and none do for organizations that switch external AI off.
- Local development can run without Docker, on SQLite or PostgreSQL with Chroma for vectors.

---

## 4. Activity diagram: uploading and indexing a document

```mermaid
flowchart TD
    Start(["Admin selects one or more files"]) --> Perm{"User is owner<br/>or admin?"}
    Perm -- No --> Deny(["403: admins only"])
    Perm -- Yes --> Valid{"Allowed type<br/>and size?"}
    Valid -- No --> Reject(["Reject file with message"])
    Valid -- Yes --> Hash["Compute SHA-256 fingerprint"]
    Hash --> Dup{"Same fingerprint already<br/>in this organization?"}
    Dup -- Yes --> Skip(["Skip as duplicate"])
    Dup -- No --> Save["Save file to private storage<br/>Create document record: Queued"]
    Save --> Queue["Hand document to background worker<br/>Respond to the browser immediately"]

    Queue --> Proc["Mark Processing"]
    Proc --> Extract["Extract text page by page<br/>PDF parser / text decoding"]
    Extract --> HasText{"Any text<br/>extracted?"}
    HasText -- No --> Fail["Mark Failed<br/>record reason"]
    HasText -- Yes --> Split["Split each page into overlapping chunks<br/>each chunk keeps its page number"]
    Split --> Clear["Remove any earlier vectors for this document"]
    Clear --> Emb["Embed every chunk<br/>text → fixed-length vector"]
    Emb --> Store["Store vectors + text + metadata<br/>tagged with the organization"]
    Store --> Still{"Document still exists?<br/>not deleted meanwhile"}
    Still -- No --> Clean["Remove the vectors just added"]
    Still -- Yes --> Ready(["Mark Ready<br/>record pages and chunk count"])
    Fail --> End(["Done"])
    Ready --> End
    Clean --> End
```

**What happens at each step**

| Step | What is done |
| --- | --- |
| Permission check | Only owners and admins of the current organization may upload |
| Validation | File type (PDF, TXT, Markdown) and maximum size are checked before anything is stored |
| Fingerprint | A SHA-256 hash of the content identifies duplicates, even when a file is renamed |
| Save | The original file goes to private storage under the organization's folder. A database record is created with status *Queued*. |
| Hand-off | Indexing runs in the background, so the upload request returns at once |
| Extract | Text is read page by page; text files are decoded with encoding fallbacks |
| Split | Text is cut into overlapping passages of a configured size; overlap keeps sentences that cross a boundary intact |
| Embed | A local model converts each passage into a vector that represents its meaning |
| Store | Vectors, passage text and metadata (organization, document, file name, page) are written to the vector store: the pgvector passage table, or the organization's Chroma collection |
| Finish | The record becomes *Ready* (or *Failed* with the reason). A document deleted during indexing leaves nothing behind. |

---

## 5. State machine: document lifecycle

```mermaid
stateDiagram-v2
    [*] --> Queued: upload accepted
    Queued --> Processing: worker picks it up
    Processing --> Ready: text extracted, chunks embedded and stored
    Processing --> Failed: no text / unreadable file / error
    Ready --> Queued: admin chooses Re-index
    Failed --> Queued: admin chooses Re-index
    Ready --> Paused: admin pauses
    Paused --> Ready: admin resumes (no re-index)
    Paused --> [*]: deleted
    Ready --> [*]: deleted (file + vectors removed)
    Failed --> [*]: deleted
    Queued --> [*]: deleted
```

Only *Ready* documents that aren't paused are searched when answering questions. Pausing keeps the file and its passages, so resuming is instant. Deleting a document removes its file and only its own vectors; other documents are untouched.

---

## 6. Sequence diagram: upload and indexing

```mermaid
sequenceDiagram
    autonumber
    actor Admin
    participant UI as Web page
    participant View as Upload view
    participant DB as Database
    participant FS as File storage
    participant W as Background worker
    participant L as Loader + Chunker
    participant E as Embedder
    participant V as Vector store

    Admin->>UI: choose files, Upload
    UI->>View: POST files
    View->>View: check role, type, size, fingerprint
    View->>FS: save original file
    View->>DB: create document (Queued)
    View-->>UI: redirect + "Uploaded, indexing in background"
    View->>W: enqueue document id (after commit)
    W->>DB: status = Processing
    W->>FS: read file
    W->>L: extract pages, split into chunks
    L-->>W: chunks with page numbers
    W->>V: delete old vectors for document
    W->>E: embed chunk texts
    E-->>W: vectors
    W->>V: add vectors + text + metadata
    W->>DB: status = Ready, pages, chunk count
    UI->>View: page auto-refreshes while indexing
    View-->>UI: document shown as Ready
```

---

## 7. Activity diagram: answering a question

```mermaid
flowchart TD
    Q(["Question arrives<br/>web chat or API"]) --> Auth{"Authenticated?<br/>session or API key"}
    Auth -- No --> E401(["401 / sign-in page"])
    Auth -- Yes --> Org["Resolve organization"]
    Org --> Quota{"Monthly limit<br/>reached?"}
    Quota -- Yes --> E429(["429: limit reached"])
    Quota -- No --> Hist["Load recent conversation turns"]
    Hist --> SaveQ["Store the question<br/>categorize: type + topic"]
    SaveQ --> First{"First question in the<br/>conversation?"}
    First -- Yes --> Cache{"Same question answered<br/>before, same documents?"}
    Cache -- Yes --> Reuse["Reuse cached answer and sources"]
    First -- No --> Search
    Cache -- No --> Search["Embed the question<br/>(with the previous question for follow-ups)"]
    Search --> Retrieve["Find the closest passages<br/>from this organization only<br/>(paused documents skipped)"]
    Retrieve --> Any{"Any passages?"}
    Any -- No --> NoDocs["Answer: no documents cover this"]
    Any -- Yes --> Prompt["Build prompt: instructions and answer rules<br/>+ numbered passages + conversation + question"]
    Prompt --> Gen["Next enabled AI provider<br/>(external ones only if allowed)"]
    Gen --> OK{"Provider<br/>answered?"}
    OK -- "No, more providers left" --> Gen
    OK -- "No, none left" --> QuoteOn{"Quoting<br/>enabled?"}
    QuoteOn -- Yes --> Quote["Quote the best-matching sentences<br/>with file and page"]
    QuoteOn -- No --> Err["Polite failure message<br/>record error detail"]
    OK -- Yes --> Put["Cache answer if first question<br/>and no provider failed"]
    Quote --> Store
    Reuse --> Store
    NoDocs --> Store
    Put --> Store
    Err --> Store["Store answer: text, sources (file + page),<br/>time taken, best match score"]
    Store --> Out(["Return answer + sources<br/>feeds analytics, gaps, feedback"])
```

**What happens at each step**

| Step | What is done |
| --- | --- |
| Authenticate | Web: session login. API: bearer key, hashed and looked up, revoked keys rejected. |
| Organization | Web: the user's current organization. API: the key's organization. |
| Quota | Optional monthly question limit per organization |
| History | The last few question/answer pairs are loaded so follow-up questions make sense |
| Categorize | The question gets a **type** (greeting, about the assistant, follow-up, document question) and the nearest admin-defined **topic** using the local model |
| Cache | A first question identical to an earlier one, with unchanged documents and settings, is answered instantly |
| Retrieve | The question is embedded with the same model used at upload; the vector store returns the most similar passages from this organization only |
| Prompt | Instructions, numbered passages with file and page, recent conversation, and the question |
| Generate | The enabled AI providers are tried in order until one answers; in the web app the answer streams word by word. If all fail, the best-matching sentences are quoted instead. Which provider answered, and why earlier ones failed, is recorded. |
| Store | The answer, its sources, the time taken and the best match score are saved. These drive Home, Analytics, Questions, Feedback and Knowledge gaps. |

---

## 8. Sequence diagram: a question in the web app (streamed)

```mermaid
sequenceDiagram
    autonumber
    actor User
    participant Page as Chat page (JS)
    participant View as Stream view
    participant Svc as Chat service
    participant Cat as Categorizer
    participant E as Embedder
    participant V as Vector store
    participant G as AI providers (in order)
    participant DB as Database

    User->>Page: type question, Enter
    Page->>View: POST question (+ session id)
    View->>Svc: ask_stream(session, question)
    Svc->>DB: check quota, load history, save question
    Svc->>Cat: type + topic
    Cat->>E: embed question (local)
    Svc->>Svc: cache lookup (first question only)
    Svc->>E: embed search text
    Svc->>V: top passages (this organization)
    V-->>Svc: passages + file + page + score
    Svc->>G: prompt to provider 1 (instructions + passages + history + question)
    alt provider fails before writing
        Svc->>G: same prompt to the next provider
    else every provider fails
        Svc->>E: pick the best-matching sentences (local)
    end
    loop while the answer is written
        G-->>Svc: text piece
        Svc-->>View: piece
        View-->>Page: {"type":"delta"} line
        Page->>Page: render formatted text so far
    end
    Svc->>DB: save answer, sources, time, score, which provider answered
    View-->>Page: {"type":"done"} with sources + message id
    Page->>User: final answer, sources, rate / save / copy actions
```

---

## 9. Sequence diagram: a question through the API

```mermaid
sequenceDiagram
    autonumber
    participant App as External app
    participant API as REST API
    participant K as API key store
    participant Svc as Chat service
    participant V as Vector store
    participant G as AI providers (in order)
    participant DB as Database

    App->>API: POST /api/v1/chat<br/>Authorization: Bearer key<br/>{"question", "session_id"?}
    API->>K: hash key, look up, check not revoked, org active
    alt invalid or revoked
        API-->>App: 401 {"success": false}
    else valid
        API->>Svc: ask(session for this key, question)
        Svc->>V: retrieve passages (key's organization)
        Svc->>G: generate answer (next provider on failure)
        alt every provider fails and quoting is off
            Svc->>DB: save failed answer + reason
            API-->>App: 503 {"success": false, …}
        else answered (by a provider, or quoted)
            Svc->>DB: save question + answer + provider
            API-->>App: 200 {"answer", "sources", "citations", "session_id"}
        end
    end
```

Other endpoints: `GET /api/v1/documents`, `GET /api/v1/validate-key`, `GET /api/v1/health`. Error codes: 400 bad input, 401 key, 405 method, 429 monthly limit, 503 AI unavailable with quoting off, or maintenance mode.

---

## 10. Class diagram: data model

```mermaid
classDiagram
    direction LR
    class User {
        username
        email
        password: hashed
        is_superuser
        is_active
    }
    class Organization {
        name
        slug
        llm_model
        system_prompt
        answer settings: length, creativity, scope, language
        use_external_ai
        top_k / history_turns / gap_threshold
        monthly_query_limit
        is_active
    }
    class Membership {
        role: owner | admin | member
        menu_prefs
    }
    class AccessRequest {
        status: pending | approved | dismissed
        message
    }
    class Document {
        original_name
        file
        checksum: SHA-256
        status: queued | processing | ready | failed
        page_count
        chunk_count
        error
        is_paused
    }
    class ChatSession {
        public_id: UUID
        channel: web | api
        is_test
        title
    }
    class Message {
        role: user | assistant
        content
        sources: list of file + page
        latency_ms
        retrieval_score
        is_error / error_detail
        answered_by: provider | quoted | cache
        rating / feedback_comment
        intent
    }
    class Topic {
        name
        description
    }
    class SavedAnswer
    class APIKey {
        name
        prefix
        key_hash: SHA-256
        last_used_at
        revoked_at
    }
    class AuditLog {
        action
        target
        detail
    }
    class Passage {
        text
        embedding: vector 384, HNSW cosine
        filename
        page
        position
    }
    class AIProvider {
        kind: gemini | openai | anthropic | groq | …
        model
        base_url
        api_key: encrypted
        priority
        is_enabled
        last_test_ok
    }
    class PlatformSettings {
        <<one row>>
        sign-up, sessions, lockout
        defaults for new organizations
        upload size, quoting, answer reuse
        announcement, maintenance
    }

    User "1" -- "*" Membership
    Organization "1" -- "*" Membership
    Organization "1" -- "*" Document
    Organization "1" -- "*" APIKey
    Organization "1" -- "*" ChatSession
    Organization "1" -- "*" Topic
    Organization "1" -- "*" AccessRequest
    User "1" -- "*" AccessRequest
    ChatSession "1" -- "*" Message
    User "0..1" -- "*" ChatSession : asks in web
    APIKey "0..1" -- "*" ChatSession : asks via API
    Topic "0..1" -- "*" Message : question topic
    User "1" -- "*" SavedAnswer
    Message "1" -- "*" SavedAnswer
    Document "1" -- "*" Passage : indexed as
    Organization "1" -- "*" Passage
    User "1" -- "*" AuditLog : superadmin actions
```

With the pgvector backend, `Passage` is a PostgreSQL table like the rest; with Chroma, the same data lives in one Chroma collection per organization. `AIProvider` and `PlatformSettings` are platform-wide. Every organization-owned record carries its organization, which is how tenants are kept apart.

---

## 11. Data flow: what stays on the server and what leaves

```mermaid
flowchart LR
    subgraph Inside["Stays on your server"]
        F["Original files"]
        T["Extracted text and chunks"]
        Vv["Vectors (embeddings)"]
        D["Users, chats, answers,<br/>ratings, keys, audit log"]
        M["Embedding model"]
    end
    subgraph Outside["Sent to the AI provider in use, per question"]
        P["Instructions + the selected passages<br/>+ recent conversation + the question"]
    end
    F --> T --> M --> Vv
    Vv -- "top matching passages" --> P
    P -- "answer text" --> D
```

- Uploading, extracting, splitting, embedding, searching and categorizing all happen locally.
- For each question, only the small set of retrieved passages and the conversation context are sent to the AI provider that answers. Whole documents are never sent.
- Nothing leaves the server when an organization switches external AI off, when a local (Ollama) provider answers, or when the answer is quoted.
- API keys are stored only as hashes; AI provider keys are encrypted; passwords use Django's salted hashing.

---

## 12. Technology summary

| Layer | Technology | Used for |
| --- | --- | --- |
| Web framework | Django 5.2 (Python 3.10) | Pages, REST API, authentication, permissions, admin |
| Relational database | PostgreSQL 16 (SQLite also works for development) | Users, organizations, documents list, chats, ratings, keys, settings, audit |
| File storage | Local disk (`media/`) | Original uploaded documents |
| Text extraction | pypdf; Python text decoding | Reading PDF pages and text files |
| Chunking | LangChain `RecursiveCharacterTextSplitter` | Overlapping passages with page numbers |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` on PyTorch | Turning text into 384-number vectors, locally |
| Vector search | pgvector 0.8 in PostgreSQL (HNSW, cosine); ChromaDB for local development | Storing and searching passages, filtered by organization |
| Answer generation | AI providers in order: Gemini (`google-generativeai`), OpenAI-compatible APIs (OpenAI, Groq, DeepSeek, OpenRouter, Ollama), Anthropic Claude; quoting as last resort | Rewording the passages found into cited answers; streaming |
| Background work | Python thread pool | Indexing uploads without blocking requests |
| Caching | Django cache framework | Instant answers to repeated questions |
| Front end | Server-rendered HTML, CSS, vanilla JavaScript; marked + DOMPurify | Interface, streaming chat, safe Markdown rendering |
| Secrets | `cryptography` (Fernet) | Encrypting AI provider keys |
| Deployment | Docker Compose: gunicorn web container + `pgvector/pgvector:pg16`; WhiteNoise for static files | Production hosting |
