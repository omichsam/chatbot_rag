# DocAssist web app. Vectors live in PostgreSQL via pgvector (see docker-compose.yml).
FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    HF_HOME=/opt/hf-cache \
    MEDIA_ROOT=/data/media \
    RAG_VECTOR_DIR=/data/vector_store \
    RAG_VECTOR_BACKEND=pgvector

WORKDIR /app

# CPU-only torch keeps the image ~2 GB smaller than the default CUDA build.
RUN pip install --no-cache-dir "torch==2.8.0" --index-url https://download.pytorch.org/whl/cpu --extra-index-url https://pypi.org/simple
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Bake the embedding model (~90 MB) into the image, so containers start without
# downloading it and work without internet access to Hugging Face.
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
ENV HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1

COPY . .
RUN DJANGO_SECRET_KEY=build-only DJANGO_DEBUG=false python manage.py collectstatic --noinput \
    && chmod +x docker-entrypoint.sh \
    && useradd --create-home --uid 1000 app \
    && mkdir -p /data/media \
    && chown -R app /data /app /opt/hf-cache

USER app
EXPOSE 8000
ENTRYPOINT ["./docker-entrypoint.sh"]
# One worker process so the embedding model is loaded once; threads handle
# concurrent requests and background indexing.
CMD ["gunicorn", "config.wsgi", "--bind", "0.0.0.0:8000", "--workers", "1", "--threads", "8", "--timeout", "180", "--access-logfile", "-"]
