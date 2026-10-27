# DocAssist documentation

DocAssist answers questions from an organization's own documents. Each answer shows the file and page it came from. This folder documents the current platform (the Django app in `django_app/`) and, separately, the older Streamlit version.

## Start here

| If you are… | Read |
| --- | --- |
| Someone who asks questions | [User guide](USER_GUIDE.md) |
| An organization owner or admin | [Admin guide](ADMIN_GUIDE.md) |
| A superadmin running the platform | [Superadmin guide](SUPERADMIN_GUIDE.md) and [AI providers](AI_PROVIDERS.md) |
| Deploying or operating it | [Server setup with test accounts](SERVER_SETUP.md), [Deployment (Docker + pgvector)](DEPLOYMENT.md) and [Configuration](CONFIGURATION.md) |
| A developer | [Developer setup](SETUP_AND_OPERATIONS.md), [Architecture](ARCHITECTURE.md), [Platform reference](DJANGO_PLATFORM.md), [Decisions](DECISIONS.md) |
| Integrating another system | [API reference](API_REFERENCE.md) |
| Looking for the big picture with diagrams | [Processes and diagrams](PROCESS_DIAGRAMS.md) |

## All documents

| Document | What it covers | Applies to | Last checked |
| --- | --- | --- | --- |
| [USER_GUIDE.md](USER_GUIDE.md) | Asking questions, sources, history, saved answers, feedback | Platform | 2026-10-05 |
| [ADMIN_GUIDE.md](ADMIN_GUIDE.md) | Documents, members, organization settings, analytics, questions, knowledge gaps, API keys | Platform | 2026-10-05 |
| [SUPERADMIN_GUIDE.md](SUPERADMIN_GUIDE.md) | The platform portal: organizations, users, platform settings, AI providers, maintenance, audit | Platform | 2026-10-05 |
| [AI_PROVIDERS.md](AI_PROVIDERS.md) | Gemini, OpenAI, Claude, Groq, DeepSeek, OpenRouter, Ollama; the fallback chain | Platform | 2026-10-05 |
| [DEPLOYMENT.md](DEPLOYMENT.md) | Docker, PostgreSQL + pgvector, backups, upgrades, moving data | Platform | 2026-10-05 |
| [SERVER_SETUP.md](SERVER_SETUP.md) | One-script setup on an Ubuntu/Debian server, with test accounts for every role | Platform | 2026-10-28 |
| [CONFIGURATION.md](CONFIGURATION.md) | Every setting: `.env`, platform settings, organization settings | Platform | 2026-10-05 |
| [ARCHITECTURE.md](ARCHITECTURE.md) | Components, request flow, storage, how answers are produced | Platform | 2026-10-05 |
| [DJANGO_PLATFORM.md](DJANGO_PLATFORM.md) | Pages, roles, data model, tenant isolation, code layout | Platform | 2026-10-05 |
| [API_REFERENCE.md](API_REFERENCE.md) | The public REST API (`/api/v1/`) | Platform | 2026-10-05 |
| [SETUP_AND_OPERATIONS.md](SETUP_AND_OPERATIONS.md) | Local development, tests, management commands, troubleshooting | Platform | 2026-10-05 |
| [PROCESS_DIAGRAMS.md](PROCESS_DIAGRAMS.md) | UML and process diagrams (Mermaid, with SVG copies in `diagrams/`) | Platform | 2026-10-05 |
| [DECISIONS.md](DECISIONS.md) | Why things are built the way they are | Platform | 2026-10-05 |
| [KNOWN_ISSUES.md](KNOWN_ISSUES.md) | Current limitations and the roadmap | Platform | 2026-10-05 |
| [CHANGELOG.md](CHANGELOG.md) | What changed, by version and date | Both | 2026-10-07 |
| [../CONTRIBUTING.md](../CONTRIBUTING.md) | Branches, commit messages, versioning and the release checklist | Both | 2026-10-07 |
| [STREAMLIT_APP.md](STREAMLIT_APP.md) | The old Streamlit app (`shit.py`) | Legacy | 2026-10-01 |

## Keeping the docs current

Documentation drifts when a change lands without its doc update. Use this checklist with every change, and record the change in [CHANGELOG.md](CHANGELOG.md) in the same commit.

| When you change… | Update |
| --- | --- |
| A page, button or workflow people see | The matching guide (user, admin or superadmin), and the pages table in DJANGO_PLATFORM.md |
| An environment variable, platform setting or organization setting | CONFIGURATION.md (and the guide that explains the setting) |
| The API (endpoints, fields, status codes) | API_REFERENCE.md |
| A model, app or the request flow | ARCHITECTURE.md, DJANGO_PLATFORM.md, and the class or sequence diagram in PROCESS_DIAGRAMS.md |
| Docker, the database or deployment steps | DEPLOYMENT.md |
| A decision with lasting consequences (a new dependency, a storage choice, a security trade-off) | Add an entry to DECISIONS.md |
| A limitation found or fixed | KNOWN_ISSUES.md |
| Anything at all | CHANGELOG.md, and the "Last checked" date above for each doc you reviewed |

**Diagrams:** the Mermaid source lives in PROCESS_DIAGRAMS.md. After editing it, regenerate the SVGs in `diagrams/` (the steps are at the top of that file).

**Writing style:** plain language, short sentences, real names and numbers. Lead with what the reader can do, then the details. Mark anything that only applies to the old Streamlit app as *legacy*.
