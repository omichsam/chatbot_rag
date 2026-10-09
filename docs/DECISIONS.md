# Decisions

Why DocAssist is built the way it is. Each entry records a choice, its reasons and its costs, so later changes can weigh the same trade-offs. Add a new entry (with the next number) for any decision with lasting consequences; to change a decision, add a new entry that supersedes the old one rather than editing it.

**Template**

```text
## D<n>. <decision in a few words>  (<date>, <status: accepted / superseded by D<m>>)
Context: what problem or constraint led to it.
Decision: what was chosen.
Consequences: what it costs, what it makes easier, what to watch.
```

---

## D1. Rebuild as a multi-tenant Django app  (2026-10-01, accepted)

**Context:** the Streamlit app was single-tenant, stored data in JSON files, had unsafe passwords and API keys that didn't work with its API.
**Decision:** a Django project with organizations as tenants, a real database, roles, hashed keys and server-rendered pages.
**Consequences:** standard security and admin tooling from Django; every query must filter by organization (enforced in middleware and tested). The Streamlit app stays only as a legacy reference.

## D2. Retrieval, not training  (2026-10-01, accepted)

**Context:** answers must come from each organization's own, changing documents, with sources.
**Decision:** retrieval-augmented generation: find passages by meaning, give only those to an AI model, cite file and page. No fine-tuning.
**Consequences:** new documents work within minutes; answers are checkable; quality depends on retrieval, chunking and the documents themselves, not on training.

## D3. Local embedding model  (2026-10-01, accepted)

**Context:** search needs every passage and question embedded; sending documents to a cloud API for that costs money and leaks content.
**Decision:** `all-MiniLM-L6-v2` (384 dimensions) running on the CPU.
**Consequences:** documents never leave the server for search; ~20 s to load at start (hidden by warm-up); quality is good for English, weaker for other languages. Changing the model means re-indexing, and pgvector's column is sized for 384.

## D4. Server-rendered pages, no front-end framework  (2026-10-01, accepted)

**Context:** a small team, simple deployment, fast pages.
**Decision:** Django templates, a hand-written CSS design system, small vanilla JavaScript (streaming, Ctrl+K, dialogs, table paging).
**Consequences:** no build step; very interactive features take more care to write.

## D5. Any AI provider, in an ordered chain, managed in the portal  (2026-10-04, accepted)

**Context:** Gemini's free quota ran out during testing and 40% of answers failed; organizations want choice of model and some want nothing sent outside.
**Decision:** superadmins add providers (Gemini, OpenAI, Claude, Groq, DeepSeek, OpenRouter, Ollama, any OpenAI-compatible API) with encrypted keys and an order; each answer tries them in turn. Clients use the standard library for HTTP, no vendor SDKs except Gemini's.
**Consequences:** resilience to one provider's outage; keys depend on `DJANGO_SECRET_KEY` staying the same; model names must be kept current by admins (the Test button helps).

## D6. Quote the documents as the last resort  (2026-10-04, accepted)

**Context:** even with several providers, all can fail; "AI service unavailable" is the worst answer.
**Decision:** when every provider fails, pick the 2–3 best-matching sentences with the local embedding model and show them with sources. Never cache these answers.
**Consequences:** people always get an accurate, sourced answer; it reads as quotes rather than prose. Platform setting can turn it off.

## D7. Platform settings in the database  (2026-10-04, accepted)

**Context:** changing sign-up, limits or maintenance needed an `.env` edit and a restart.
**Decision:** one `PlatformSettings` row, seeded from `.env`, edited in the portal, cached for 30 seconds, every change audit-logged.
**Consequences:** no restarts; two sources of truth at first start (the portal wins afterwards); multi-process deployments see changes within 30 s.

## D8. pgvector for production, Chroma for local development  (2026-10-05, accepted)

**Context:** one database is simpler to back up and keep consistent than a database plus a Chroma folder; the Windows PostgreSQL install lacks pgvector.
**Decision:** `RAG_VECTOR_BACKEND` selects pgvector (Docker, production) or Chroma (local). The passage table is created only where the extension exists; passages are removed by delete signals rather than database cascades so the app works without the table.
**Consequences:** both backends must be maintained and tested (pgvector tests run in Docker); switching needs `manage.py reindex`.

## D9. Docker with one worker and eight threads  (2026-10-05, accepted)

**Context:** the embedding model takes memory and ~20 s to load per process.
**Decision:** one gunicorn worker with 8 threads per container; the model is built into the image.
**Consequences:** one model in memory, fast starts, offline-capable image (~2.8 GB). Scaling out means more containers, which then need a shared cache (see KNOWN_ISSUES L1).

## D10. Ten items per page everywhere  (2026-10-04, accepted)

**Context:** long tables and lists made pages scroll endlessly.
**Decision:** every table pages at 10 rows with search, sorting and a page-size choice (client side); large server lists (Questions) page on the server.
**Consequences:** consistent, fast pages; every new table must carry `data-table` (and `data-sort` on date or size cells).
