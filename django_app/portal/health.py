"""System health for the platform portal: cheap checks on page load, real tests on demand."""
import time

import django
from django.conf import settings
from django.db import connection
from django.template.defaultfilters import filesizeformat

from documents.models import Document
from rag import embeddings, vectorstore

from .metrics import folder_size
from .models import AIProvider, platform


def _mask(secret):
    return f"…{secret[-4:]}" if secret and len(secret) > 8 else ("set" if secret else "")


def status():
    """Each check: name, ok (True/False/None = unknown), summary, and details rows."""
    checks = []

    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
        db_ok = True
    except Exception as exc:  # pragma: no cover - shown on the page
        db_ok, db_error = False, str(exc)
    checks.append({
        "key": "database", "name": "Database", "ok": db_ok,
        "summary": "Connected" if db_ok else f"Error: {db_error}",
        "details": [("Engine", connection.vendor), ("Name", str(settings.DATABASES["default"]["NAME"])),
                    ("Django", django.get_version())],
    })

    vector_ok, info, vector_error = None, {}, ""
    try:
        info = vectorstore.stats()
        vector_ok = True
    except Exception as exc:
        vector_ok, vector_error = False, str(exc)
    details = [("Backend", info.get("backend", vectorstore.backend())), ("Location", info.get("location", "–"))]
    if vectorstore.backend() == "chroma":
        details.append(("Size on disk", filesizeformat(folder_size(settings.RAG_VECTOR_DIR))))
    checks.append({
        "key": "vectors", "name": f"Vector store ({'pgvector' if vectorstore.backend() == 'pgvector' else 'Chroma'})",
        "ok": vector_ok,
        "summary": (f"{info['groups']} organization{'s' if info['groups'] != 1 else ''} · {info['passages']:,} passages"
                    if vector_ok else f"Error: {vector_error}"),
        "details": details,
    })

    enabled = list(AIProvider.objects.filter(is_enabled=True))
    tested = [p for p in enabled if p.last_test_ok is not None]
    checks.append({
        "key": "llm", "name": "AI answers (providers)",
        # Healthy when at least one provider passed its last test; unknown until one is tested.
        "ok": False if not enabled else (any(p.last_test_ok for p in tested) if tested else None),
        "summary": (f"{len(enabled)} enabled: " + " → ".join(p.name for p in enabled)) if enabled
        else "No provider enabled; answers quote the documents",
        "details": [(f"{i}. {p.name}", f"{p.model_for(None) or '–'} · key {p.key_display} · "
                     + ("last test passed" if p.last_test_ok else "last test failed" if p.last_test_ok is False else "not tested"))
                    for i, p in enumerate(enabled, start=1)] or [("Providers", "Add one on the AI providers page")],
    })

    checks.append({
        "key": "embeddings", "name": "Embeddings (local model)", "ok": True,
        "summary": "Loaded in memory" if embeddings._model is not None else "Loads on first use",
        "details": [("Backend", settings.RAG_EMBEDDING_BACKEND), ("Model", settings.RAG_EMBEDDING_MODEL),
                    ("Runs on", "this server (no external calls)")],
    })

    busy = Document.objects.filter(status__in=[Document.Status.PENDING, Document.Status.PROCESSING]).count()
    failed = Document.objects.filter(status=Document.Status.FAILED).count()
    checks.append({
        "key": "ingest", "name": "Document indexing", "ok": failed == 0,
        "summary": f"{busy} in progress · {failed} failed",
        "details": [("Mode", settings.RAG_INGEST_MODE), ("Workers", settings.RAG_INGEST_WORKERS),
                    ("Chunk size / overlap", f"{settings.RAG_CHUNK_SIZE} / {settings.RAG_CHUNK_OVERLAP}"),
                    ("Max upload", f"{platform().max_upload_mb} MB")],
    })

    checks.append({
        "key": "storage", "name": "File storage", "ok": True,
        "summary": "Uploaded originals",
        "details": [("Folder", str(settings.MEDIA_ROOT)), ("Size on disk", filesizeformat(folder_size(settings.MEDIA_ROOT)))],
    })
    return checks


def configuration():
    return [
        ("Debug mode", "On (development)" if settings.DEBUG else "Off (production)"),
        ("Allowed hosts", ", ".join(settings.ALLOWED_HOSTS) or "–"),
        ("Self sign-up", "Allowed" if platform().allow_signup else "Disabled"),
        ("Maintenance mode", "On" if platform().maintenance_mode else "Off"),
        ("Time zone", settings.TIME_ZONE),
        ("Top passages per answer", settings.RAG_TOP_K),
        ("Conversation turns remembered", settings.RAG_HISTORY_TURNS),
        ("Knowledge-gap threshold", settings.RAG_GAP_SCORE),
        ("Answer writers, in order", " → ".join(
            [p.name for p in AIProvider.objects.filter(is_enabled=True)]
            + (["quoted from documents"] if platform().quoted_fallback else []))),
    ]


def run_test(kind):
    """Run one live test. Returns (ok, message)."""
    started = time.monotonic()
    try:
        if kind == "llm":
            from .ai import test_provider

            provider = AIProvider.objects.filter(is_enabled=True).first()
            if provider is None:
                return False, "No AI provider is enabled."
            ok, message = test_provider(provider)
            return ok, f"{provider.name}: {message}"
        elif kind == "embeddings":
            vector = embeddings.embed_query("health check")
            result = f"Produced a {len(vector)}-number vector"
        elif kind == "vectors":
            info = vectorstore.stats()
            result = f"{info['backend']}: {info['passages']:,} passages"
        elif kind == "database":
            with connection.cursor() as cursor:
                cursor.execute("SELECT 1")
            result = "Query succeeded"
        else:
            return False, "Unknown test."
    except Exception as exc:
        return False, f"Failed after {time.monotonic() - started:.1f} s: {str(exc)[:300]}"
    return True, f"{result} in {time.monotonic() - started:.2f} s"
