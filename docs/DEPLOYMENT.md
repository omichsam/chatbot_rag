# Deployment (Docker + PostgreSQL + pgvector)

DocAssist runs as two containers: **web** (the Django app under gunicorn) and **db** (PostgreSQL 16 with the pgvector extension). All data lives in two Docker volumes, so it survives restarts and rebuilds.

```text
browser / API client ──▶ web :8000 ──▶ db :5432 (PostgreSQL 16 + pgvector)
                           │  └─ /data/media  (uploaded files, volume "appdata")
                           └──▶ AI providers over HTTPS (Gemini, OpenAI, Claude, …)
```

## Requirements

- Docker Desktop (Windows/macOS) or Docker Engine with the Compose plugin (Linux).
- About 3.5 GB of disk for the images (web 2.8 GB with PyTorch CPU and the embedding model; database 0.6 GB) plus your data.
- Internet access for the first build and for cloud AI providers.

## First start

```bash
cd django_app
cp .env.docker.example .env.docker      # then edit it, below
docker compose up -d --build            # first build ~10–40 min, depending on the connection
docker compose logs -f web              # watch migrations, then "Listening at: http://0.0.0.0:8000"
docker compose exec web python manage.py createsuperuser
```

Open <http://localhost:8001>.

In `.env.docker`, set at least:

| Variable | Set it to |
| --- | --- |
| `DJANGO_SECRET_KEY` | A long random string (e.g. `python -c "import secrets;print(secrets.token_urlsafe(50))"`). Keep it stable: it also encrypts AI provider keys. |
| `POSTGRES_PASSWORD` | A strong password (used by both containers) |
| `GOOGLE_API_KEY` | Your Gemini key (more providers can be added in the portal) |
| `DJANGO_ALLOWED_HOSTS` | The host names people use, e.g. `docs.example.com` |
| `DJANGO_CSRF_TRUSTED_ORIGINS` | The full origins, e.g. `https://docs.example.com` |

The full list is in [CONFIGURATION.md](CONFIGURATION.md). Keep `.env.docker` private; it's excluded from the image by `.dockerignore`.

## What happens on start

`docker-entrypoint.sh` runs, in order: database migrations; `setup_pgvector` (creates the `vector` extension and the passage table if missing); an optional first superadmin from `DJANGO_SUPERUSER_USERNAME` / `_PASSWORD`; then gunicorn with **one worker and 8 threads**, so the embedding model is loaded into memory once. Both containers have health checks and restart automatically.

## Everyday commands

Run these in `django_app/`.

| Task | Command |
| --- | --- |
| Start / stop | `docker compose up -d` / `docker compose down` (data is kept) |
| Status and health | `docker compose ps` |
| Logs | `docker compose logs -f web` |
| Upgrade after code changes | `docker compose up -d --build` (migrations run automatically) |
| Django shell | `docker compose exec web python manage.py shell` |
| Change a password | `docker compose exec web python manage.py changepassword <username>` |
| Rebuild the search index | `docker compose exec web python manage.py reindex` |
| Run the tests (including pgvector) | `docker compose exec web python manage.py test` |

On Windows Git Bash, prefix commands that pass container paths with `MSYS_NO_PATHCONV=1`, otherwise paths like `/tmp/x` are rewritten into Windows paths.

## Backups

Everything that matters is the database (including all search passages) and the uploaded files.

```bash
# Back up
docker compose exec -T db pg_dump -U docassist -d docassist -Fc > docassist-$(date +%F).dump
docker compose cp web:/data/media ./media-backup-$(date +%F)

# Restore into a fresh setup (stop web first)
docker compose stop web
docker compose exec -T db pg_restore -U docassist -d docassist --clean --if-exists < docassist-2026-10-05.dump
docker compose cp ./media-backup-2026-10-05/. web:/data/media/
docker compose start web
```

Keep backups off the machine. Test a restore occasionally.

## Moving existing data in

To bring data from a local install (SQLite or PostgreSQL without pgvector) into Docker:

```bash
# On the old install (in django_app/)
python manage.py dumpdata --natural-foreign --natural-primary \
  -e contenttypes -e auth.permission -e admin.logentry -e sessions -e documents.passage -o seed.json

# Into Docker
docker compose cp seed.json web:/tmp/seed.json
docker compose exec web python manage.py loaddata /tmp/seed.json
docker compose cp media/. web:/data/media/
docker compose exec -u root web chown -R app /data/media
docker compose exec web python manage.py reindex        # rebuilds passages in pgvector
```

Users keep their passwords. Everyone signs in again (sessions aren't moved). AI provider keys only carry over if `DJANGO_SECRET_KEY` is the same; otherwise re-enter them.

## Going to production

- Put HTTPS in front (nginx, Caddy, Traefik or a cloud load balancer) and forward to port 8000.
- In `.env.docker`: `SESSION_COOKIE_SECURE=true`, `CSRF_COOKIE_SECURE=true`, your domain in `DJANGO_ALLOWED_HOSTS` and `DJANGO_CSRF_TRUSTED_ORIGINS`.
- Don't publish the database port; only `web` exposes a port.
- Running more than one web worker or server: add a shared cache (Redis, or Django's database cache) so sign-in lockout counts and the platform-settings cache are shared. See [KNOWN_ISSUES.md](KNOWN_ISSUES.md).
- Schedule backups and watch `docker compose ps` health.

## Optional: a local AI model in Docker

Add an Ollama service so answers can be written on your own server:

```yaml
  ollama:
    image: ollama/ollama
    volumes: [ollama:/root/.ollama]
    restart: unless-stopped
# and under volumes:  ollama:
```

Then `docker compose exec ollama ollama pull llama3.2:3b`, and in the portal add an **Ollama** provider with the address `http://ollama:11434/v1`. It needs a capable CPU or a GPU to be quick.

## Image details

- Base `python:3.10-slim`; PyTorch 2.8.0 CPU build; dependencies from `requirements.txt`.
- The embedding model `all-MiniLM-L6-v2` is downloaded during the build, and the container runs with Hugging Face offline mode on.
- Runs as a non-root `app` user. Static files are collected at build time and served by WhiteNoise.
