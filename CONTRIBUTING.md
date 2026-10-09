# Contributing to DocAssist

How changes are made, recorded and released. Maintainer: **omichsam**.

## Workflow

1. Create a branch from `main`: `git switch -c feature/short-name` (or `fix/…`, `docs/…`).
2. Make the change with its tests and documentation (see the checklist below).
3. Run the checks:
   ```bash
   cd django_app
   python manage.py test --noinput                 # all tests (pgvector tests run inside Docker)
   python manage.py makemigrations --check --dry-run
   python -m pyflakes rag chat documents tenants portal api
   ```
4. Commit using the message style below, push the branch, and open a pull request into `main`.

## Commit messages

Use [Conventional Commits](https://www.conventionalcommits.org/):

```text
<type>(<area>): <what changed, in the imperative>

<why, if it isn't obvious>
```

| Type | For |
|---|---|
| `feat` | A new feature (bumps MINOR) |
| `fix` | A bug fix (bumps PATCH) |
| `docs` | Documentation only |
| `refactor`, `perf`, `test`, `build`, `chore` | Code changes that don't add features or fix bugs |
| `feat!` / `BREAKING CHANGE:` | A change that breaks existing use (bumps MAJOR) |

Areas: `chat`, `documents`, `rag`, `tenants`, `portal`, `api`, `docker`, `docs`, `legacy`.

## Versioning

[Semantic versioning](https://semver.org/). The version lives in one place, [`django_app/VERSION`](django_app/VERSION); the app reads it (Platform Console header, `GET /api/v1/health`). Every release has a git tag `vX.Y.Z` and an entry in [docs/CHANGELOG.md](docs/CHANGELOG.md).

## Release checklist

1. Move the changes under a new version heading in `docs/CHANGELOG.md` (`## [X.Y.Z] - YYYY-MM-DD`, with Added / Changed / Fixed / Security).
2. Update `django_app/VERSION`.
3. Check the docs listed in [docs/README.md](docs/README.md#keeping-the-docs-current) and update their "Last checked" dates.
4. Run the checks above.
5. Commit: `chore(release): vX.Y.Z`.
6. Tag and push: `git tag -a vX.Y.Z -m "DocAssist X.Y.Z"` then `git push origin main --tags`.
7. Optionally publish a GitHub release from the tag with the changelog section: `gh release create vX.Y.Z --notes-file <notes>`.

## Never commit

Secrets and private data are ignored by `.gitignore`; keep it that way:

- `.env`, `django_app/.env`, `django_app/.env.docker` (keys, passwords, the Django secret key)
- databases (`db.sqlite3`), uploaded documents (`media/`, `uploaded_files/`), vector indexes (`vector_store/`, `chroma_db/`)
- legacy stores with users and keys (`users.json`, `api_keys.json`, `analytics.json`)

If a secret is ever committed, treat it as leaked: rotate it first, then remove it from the history.
