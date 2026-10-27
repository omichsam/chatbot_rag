#!/usr/bin/env bash
# One-shot setup of DocAssist on an Ubuntu/Debian server, with test accounts.
# For a controlled test environment: the test passwords are printed and documented.
#
#   curl -fsSL https://raw.githubusercontent.com/omichsam/chatbot_rag/main/deploy/server-setup.sh -o server-setup.sh
#   sudo GOOGLE_API_KEY=your-key bash server-setup.sh
#
# Re-running it is safe: it pulls the latest code, keeps the existing .env.docker
# and data, rebuilds, and resets the test accounts.
#
# Optional settings (environment variables):
#   APP_DIR         where the code goes            (default /opt/docassist)
#   REPO_URL        git repository                 (default the GitHub repo)
#   BRANCH          branch to deploy               (default main)
#   SERVER_HOST     IP or domain people will use   (default this server's first IP)
#   GOOGLE_API_KEY  Gemini key, written to .env.docker on first setup only
#   SEED_USERS      set to 0 to skip test accounts
set -euo pipefail

APP_DIR="${APP_DIR:-/opt/docassist}"
REPO_URL="${REPO_URL:-https://github.com/omichsam/chatbot_rag.git}"
BRANCH="${BRANCH:-main}"
SERVER_HOST="${SERVER_HOST:-$(hostname -I | awk '{print $1}')}"
SEED_USERS="${SEED_USERS:-1}"
PORT=8001

step() { printf '\n\033[1;34m==> %s\033[0m\n' "$*"; }
die() { printf '\033[1;31mError: %s\033[0m\n' "$*" >&2; exit 1; }

[ "$(id -u)" -eq 0 ] || die "run as root: sudo bash $0"
command -v apt-get >/dev/null || die "this script supports Ubuntu/Debian (apt) only"

step "Installing git, curl and openssl"
apt-get update -qq
apt-get install -y -qq git curl openssl ca-certificates >/dev/null

if ! command -v docker >/dev/null || ! docker compose version >/dev/null 2>&1; then
  step "Installing Docker Engine and the Compose plugin"
  curl -fsSL https://get.docker.com | sh
fi
systemctl enable --now docker >/dev/null

step "Getting the code into $APP_DIR ($BRANCH)"
if [ -d "$APP_DIR/.git" ]; then
  git -C "$APP_DIR" fetch --quiet origin "$BRANCH"
  git -C "$APP_DIR" checkout --quiet "$BRANCH"
  git -C "$APP_DIR" reset --quiet --hard "origin/$BRANCH"
else
  git clone --quiet --branch "$BRANCH" "$REPO_URL" "$APP_DIR"
fi
cd "$APP_DIR/django_app"

if [ ! -f .env.docker ]; then
  step "Writing .env.docker for http://$SERVER_HOST:$PORT"
  cp .env.docker.example .env.docker
  set_env() { sed -i "s|^#\? \?$1=.*|$1=$2|" .env.docker; }
  set_env DJANGO_SECRET_KEY "$(openssl rand -base64 48 | tr -d '\n/+=')"
  set_env POSTGRES_PASSWORD "$(openssl rand -hex 16)"
  set_env DJANGO_ALLOWED_HOSTS "$SERVER_HOST,localhost,127.0.0.1"
  set_env DJANGO_CSRF_TRUSTED_ORIGINS "http://$SERVER_HOST:$PORT,http://localhost:$PORT"
  set_env GOOGLE_API_KEY "${GOOGLE_API_KEY:-}"
  chmod 600 .env.docker
  [ -n "${GOOGLE_API_KEY:-}" ] || echo "No GOOGLE_API_KEY given: add one to $APP_DIR/django_app/.env.docker or in the portal (AI providers)."
else
  step "Keeping the existing .env.docker"
fi

step "Building and starting the containers (the first build takes 10-40 minutes)"
docker compose up -d --build

step "Waiting for the app to become healthy"
for _ in $(seq 1 60); do
  status="$(docker inspect -f '{{.State.Health.Status}}' "$(docker compose ps -q web)" 2>/dev/null || true)"
  [ "$status" = "healthy" ] && break
  sleep 5
done
[ "$status" = "healthy" ] || die "the web container isn't healthy yet; check: cd $APP_DIR/django_app && docker compose logs web"

if command -v ufw >/dev/null && ufw status | grep -q "Status: active"; then
  step "Opening port $PORT in the firewall"
  ufw allow "$PORT/tcp" >/dev/null
fi

if [ "$SEED_USERS" = "1" ]; then
  step "Creating test accounts"
  docker compose exec -T web python manage.py seed_test_users
fi

step "Done"
echo "Open http://$SERVER_HOST:$PORT"
echo "Logs:    cd $APP_DIR/django_app && docker compose logs -f web"
echo "Update:  sudo bash $APP_DIR/deploy/server-setup.sh"
