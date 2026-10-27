# Server setup (test environment)

This guide sets up DocAssist on a fresh Ubuntu or Debian server with one script. It installs Docker, runs the app with PostgreSQL + pgvector, and creates **test accounts with known passwords**.

> **Controlled test environments only.** The test passwords are in this file and in the code, so anyone who can reach the server can sign in with them. Before real users get access, change or remove the test accounts (see [Before going live](#before-going-live)).

## Requirements

- Ubuntu 22.04/24.04 or Debian 12, with `sudo` access
- 2 CPU cores, 4 GB RAM, about 10 GB free disk
- Outbound internet access (GitHub, Docker Hub, PyPI, Hugging Face for the first build)
- Port **8001** reachable from your network (also open it in your cloud provider's firewall or security group, if there is one)

## 1. Run the setup script

SSH into the server, then run:

```bash
curl -fsSL https://raw.githubusercontent.com/omichsam/chatbot_rag/main/deploy/server-setup.sh -o server-setup.sh
sudo GOOGLE_API_KEY=your-gemini-key bash server-setup.sh
```

If the repository is private, clone it yourself first (`git clone https://github.com/omichsam/chatbot_rag.git /opt/docassist`, using a GitHub token or deploy key), then run `sudo bash /opt/docassist/deploy/server-setup.sh`.

The script:

1. Installs git, curl, openssl, and Docker Engine with the Compose plugin, if they're missing.
2. Clones the code into `/opt/docassist`, or updates it if it's already there.
3. Writes `django_app/.env.docker` with a random secret key and database password, and the server's IP in the allowed hosts. It only does this on the first run.
4. Builds and starts the containers. The first build takes 10–40 minutes.
5. Waits until the app is healthy, then opens port 8001 if `ufw` is active.
6. Creates the test accounts below and prints them.

When it finishes, open `http://<server-ip>:8001`.

### Options

Set these before `bash`, e.g. `sudo SERVER_HOST=docs.example.com bash server-setup.sh`:

| Variable | Default | Use |
| --- | --- | --- |
| `SERVER_HOST` | The server's first IP | The IP or domain people type in the browser |
| `GOOGLE_API_KEY` | empty | Gemini key. Only used on the first run; later, edit `.env.docker` or add providers in the portal |
| `APP_DIR` | `/opt/docassist` | Where the code is installed |
| `BRANCH` | `main` | The branch to deploy |
| `REPO_URL` | the GitHub repo | Another clone URL (e.g. SSH) |
| `SEED_USERS` | `1` | `0` skips the test accounts |

## 2. Test accounts

| Username | Password | Role | What to test |
| --- | --- | --- | --- |
| `superadmin` | `Test@Super2026` | Superadmin | Platform Console: organizations, users, plans, AI providers, settings, audit log |
| `support` | `Test@Support2026` | Platform staff: Support | The console sections support staff can use |
| `finance` | `Test@Finance2026` | Platform staff: Finance | The console sections finance staff can use |
| `owner` | `Test@Owner2026` | Owner of **Test Organization** | Everything an admin can do, plus organization ownership |
| `orgadmin` | `Test@OrgAdmin2026` | Admin of **Test Organization** | Uploading documents, members, organization settings, analytics, API keys |
| `member` | `Test@Member2026` | Member of **Test Organization** | Asking questions, history, saved answers, feedback |

To create them again, or to reset their passwords and roles at any time:

```bash
cd /opt/docassist/django_app
sudo docker compose exec web python manage.py seed_test_users
# or give every test account the same password:
sudo docker compose exec web python manage.py seed_test_users --password "Your-Pass-123"
```

Re-running also removes any authenticator app (2FA) from the test accounts, so none of them stay locked out.

## 3. Everyday commands

Run these in `/opt/docassist/django_app` with `sudo`:

| Task | Command |
| --- | --- |
| Deploy the latest code from GitHub | `bash /opt/docassist/deploy/server-setup.sh` |
| Follow the app logs | `docker compose logs -f web` |
| Status | `docker compose ps` |
| Restart | `docker compose restart web` |
| Stop / start | `docker compose down` / `docker compose up -d` |
| Change a setting | Edit `.env.docker`, then `docker compose up -d` |
| Add a real superadmin | `docker compose exec web python manage.py createsuperuser` |

Backups, restores and moving data are covered in [DEPLOYMENT.md](DEPLOYMENT.md).

## Troubleshooting

| Problem | Fix |
| --- | --- |
| The page doesn't load from your PC | Open port 8001 in the cloud firewall or security group. Check `docker compose ps` shows `web` as healthy |
| "Bad Request (400)" | The address you typed isn't in `DJANGO_ALLOWED_HOSTS`. Add it in `.env.docker`, then `docker compose up -d` |
| "CSRF verification failed" when signing in | Add the full address (e.g. `http://1.2.3.4:8001`) to `DJANGO_CSRF_TRUSTED_ORIGINS`, then `docker compose up -d` |
| Answers say the AI service is unavailable | No working AI key. Set `GOOGLE_API_KEY` in `.env.docker`, or add a provider in the Platform Console (AI providers) |
| The script stops at "isn't healthy yet" | Read `docker compose logs web`. A slow first start can simply need more time: run the script again |

## Before going live

1. Delete the test accounts in the Platform Console (Users), or give them strong passwords with `docker compose exec web python manage.py changepassword USERNAME`.
2. Create your own superadmin with `createsuperuser`, and turn on two-factor sign-in for the console (Platform Console, Security).
3. Put the app behind a domain with HTTPS (e.g. nginx or Caddy in front of port 8001), then set `SESSION_COOKIE_SECURE=true` and `CSRF_COOKIE_SECURE=true`, and use the `https://` address in `DJANGO_CSRF_TRUSTED_ORIGINS`.
4. Set up backups ([DEPLOYMENT.md](DEPLOYMENT.md#backups)).
