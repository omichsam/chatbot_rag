#!/bin/sh
# Prepare the database, then start the app (the CMD).
set -e

python manage.py migrate --noinput
if [ "$RAG_VECTOR_BACKEND" = "pgvector" ]; then
  python manage.py setup_pgvector
fi

# Optional first superadmin, from DJANGO_SUPERUSER_USERNAME / _PASSWORD / _EMAIL.
if [ -n "$DJANGO_SUPERUSER_USERNAME" ] && [ -n "$DJANGO_SUPERUSER_PASSWORD" ]; then
  python manage.py createsuperuser --noinput --email "${DJANGO_SUPERUSER_EMAIL:-admin@example.com}" 2>/dev/null \
    && echo "Created superadmin $DJANGO_SUPERUSER_USERNAME" || true
fi

exec "$@"
