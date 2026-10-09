# Platform reference (`django_app/`)

The multi-tenant Django platform that replaced the Streamlit app. Each **organization** (tenant) has its own documents, chats, members, API keys, analytics and settings; nothing is shared between organizations. Superadmins run the whole platform from a portal.

Related: [Architecture](ARCHITECTURE.md) · [Configuration](CONFIGURATION.md) · [API](API_REFERENCE.md) · [Deployment](DEPLOYMENT.md) · [Developer setup](SETUP_AND_OPERATIONS.md)

## Running it

- **Docker (recommended):** `cd django_app && docker compose up -d --build` → <http://localhost:8001>. See [DEPLOYMENT.md](DEPLOYMENT.md).
- **Local development:** `python manage.py runserver 127.0.0.1:8123` with the project `venv`. See [SETUP_AND_OPERATIONS.md](SETUP_AND_OPERATIONS.md).

## Roles

| | Member | Admin | Owner | Superadmin |
| --- | :-: | :-: | :-: | :-: |
| Ask questions, history, saved answers, see the document list | ✓ | ✓ | ✓ | in any organization they belong to |
| Upload, pause, re-index, delete documents | | ✓ | ✓ | |
| Analytics, Questions, Feedback, Knowledge gaps, API keys, Settings | | ✓ | ✓ | |
| Add or remove members, change roles | | ✓ (not owners) | ✓ | |
| Make someone an owner | | | ✓ | |
| Platform portal: all organizations and users, platform settings, AI providers, health, audit | | | | ✓ |

One user can belong to several organizations with a different role in each and switches with the organization name at the top left. Superadmin is Django's superuser flag; grant it in the portal (*Users*) or with `createsuperuser`.

## Pages

| URL | Who | What |
| --- | --- | --- |
| `/accounts/login/`, `/accounts/signup/` | anyone | Sign in (with lockout after repeated failures) / create an account and organization (if sign-up is allowed) |
| `/dashboard/` (Home) | members | Questions, answer quality, activity by outcome, busiest times, top topics, most-used documents; admins also see *Needs attention* |
| `/chat/` | members | Chat with streaming answers, sources, rating, save and copy; welcome message and suggested questions |
| `/chat/history/`, `/chat/saved/` | members | Own past chats; own saved answers |
| `/chat/help/` | members | Help page |
| `/documents/` | members (read), admins (manage) | Upload several files; live status; pause/resume, re-index, delete with confirmation |
| `/chat/analytics/` | admins | Questions per day (30 days), answer time, failed answers, top questions, most active people |
| `/chat/questions/` | admins | Every question with filters, 10 per page, CSV export; topics |
| `/chat/gaps/`, `/chat/feedback/` | admins | Knowledge gaps to resolve; 👎 answers to review |
| `/keys/` | admins | Create (shown once) and revoke API keys |
| `/developer/` | members | API playground with generated code |
| `/org/members/` | admins | Members, roles, access requests |
| `/org/settings/` | admins | Organization settings in six tabs |
| `/org/new/` | any user | Create another organization |
| `/platform/` … | console roles | Platform Console: dashboard, organizations, plans, users, AI providers, security centre, audit log, platform settings, system health, guide |
| `/accounts/2fa/` | anyone signing in with two-factor | The second sign-in step |
| `/api/v1/` | API keys | The REST API |
| `/admin/` | superadmins | Django admin for raw data |

Tables and long lists show 10 items per page with search, sorting and a page-size choice (`templates/partials/datatable.js`); the Questions list pages on the server.

## Code layout

```text
django_app/
├── config/        settings (environment-driven), urls, wsgi
├── tenants/       Organization, Membership, AccessRequest; current-org middleware; role decorators;
│                  sign-in with lockout; organization settings; menu; import_legacy command
├── documents/     Document, Passage (pgvector); upload, pause, re-index, background indexing;
│                  reindex and setup_pgvector commands
├── chat/          ChatSession, Message, Topic, SavedAnswer; chat, streaming, history, dashboard,
│                  analytics, questions explorer, categorization, gaps, feedback, Ctrl+K search
├── api/           APIKey; REST API; key pages; Developer playground
├── portal/        AuditLog, AIProvider, PlatformSettings; superadmin pages; maintenance middleware
├── rag/           loaders, chunking, embeddings, vectorstore (pgvector/Chroma), llm (provider
│                  clients), pipeline (answer chain), extractive (quotes), answer_cache, warmup
└── templates/     server-rendered pages; partials/styles.css and partials/datatable.js
```

## Data model

```text
Organization ─┬─< Membership >── User ──< AccessRequest
              ├─< Document ─< Passage            (pgvector backend)
              ├─< APIKey
              ├─< Topic
              └─< ChatSession ─< Message ──< SavedAnswer >── User
Platform-wide:  PlatformSettings (one row) · AIProvider · AuditLog · Plan · PlatformStaff · TwoFactorDevice
```

- **Organization:** name, slug, active flag; model and instructions; answer settings (length, creativity, scope, language, show sources, use external AI); retrieval settings (passages per answer, memory, gap threshold); chat settings (assistant name, welcome message, suggestions); monthly limit; answer reuse.
- **Membership:** user + organization (unique) + role (owner, admin, member) + menu preferences.
- **Document:** file under `media/documents/org_<id>/`, SHA-256 checksum, status (queued, processing, ready, failed), error, page and chunk counts, paused flag.
- **Passage:** organization, document, file name, page, position, text and a 384-number `embedding` with an HNSW cosine index. Removed by the document and organization delete signals.
- **APIKey:** SHA-256 hash and display prefix only; revoked and last-used times.
- **ChatSession:** public UUID, channel (web or API), the user or API key, test flag.
- **Message:** role, content, sources (file and page), timing, error flag and detail, `answered_by` (provider kind, `quoted` or `cache`), match score, rating and comment, type and topic.
- **AIProvider:** kind, name, model, API address, encrypted key, enabled flag, priority, timeout, last test result.
- **PlatformSettings:** the platform settings (one row, cached).
- **AuditLog:** actor, action, organization, target, detail, time.
- **Plan:** name, monthly question limit, document and member caps, cloud AI allowed. Organizations link to one (or none: custom limits) and also carry `suspended_reason`, `suspended_at` and `deletion_scheduled_at`.
- **PlatformStaff:** a console role (support or finance) for someone who isn't a superuser.
- **TwoFactorDevice:** an encrypted authenticator secret, the last accepted time step, and hashed backup codes.

## How tenant isolation works

1. **Web:** `CurrentOrganizationMiddleware` sets `request.organization` from the session, only ever to an organization the user belongs to.
2. **API:** the key decides the organization; a key can only reach its own.
3. **Queries** always filter by the organization, and object lookups include it, so another tenant's IDs give **404**, not 403.
4. **Vectors:** pgvector searches filter by `organization_id`; with Chroma each organization has its own collection.
5. **Tests** check that cross-tenant access fails for documents, chats, members, keys and search results.

## Moving over from the Streamlit app

`python manage.py import_legacy --org-name "<name>" [--source <folder>] [--skip-documents]`

| Legacy | Becomes |
| --- | --- |
| `users.json` | Users. SHA-256 hashes are kept and upgraded to PBKDF2 at each user's next sign-in. The first admin becomes **owner**, other admins **admin**, everyone else **member**. A user still on `admin123` gets no usable password and must be given one. |
| `api_keys.json` | API keys, stored hashed. Existing clients keep working; revoked keys stay revoked. |
| `uploaded_files/*` | Documents, copied into `media/` and indexed. |
| `analytics.json` | Not imported; analytics start fresh. |

Running it again is safe: anything already imported is skipped. **Imported on 2026-10-01** into "ICTA": users `admin` (owner) and `sam` (member), key `ICTA_CHAT`, `doc1.pdf` (2 pages, 6 passages) and `doc2.pdf` (14 pages, 28 passages).

### What changed compared to Streamlit

| Streamlit problem | Platform |
| --- | --- |
| Single tenant, data in JSON files | Organizations, PostgreSQL, migrations |
| Default `admin` / `admin123` | None; superadmin created deliberately |
| Unsalted SHA-256 passwords | PBKDF2, legacy hashes upgraded on sign-in, sign-in lockout |
| API keys in plain text; UI keys rejected by the API | Hashed keys shared by UI and API; revoking works immediately |
| Delete re-indexes everything | Removes only that document's passages |
| Duplicate uploads add duplicate chunks | Checksum check skips duplicates |
| No chat memory | Conversations with history; follow-ups work |
| Upload blocks the page | Background indexing with live status |
| One AI model; failures are errors | Any number of providers in order, then quoting the documents |
| HTML injection through chat | Everything escaped; Markdown sanitised |
| No page numbers in sources | File and page for every answer |
