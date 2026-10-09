"""Load the embedding model and open each vector index when the server starts.

Without this, the first question after every server start waits ~20 s for the
model to load and ~1.5 s for the index to open.
"""
import logging
import os
import sys
import threading
import time

from django.conf import settings

logger = logging.getLogger(__name__)


def is_serving():
    """True only in a process that will answer web requests."""
    argv = " ".join(sys.argv)
    if "runserver" in sys.argv:
        # With auto-reload, the outer process only watches files; the child (RUN_MAIN) serves.
        return os.environ.get("RUN_MAIN") == "true" or "--noreload" in sys.argv
    return any(server in argv for server in ("gunicorn", "uvicorn", "waitress", "daphne"))


def warm_up():
    from django.db import close_old_connections, connections

    from documents.models import Document

    from . import embeddings, vectorstore

    started = time.monotonic()
    close_old_connections()
    try:
        embeddings.embed_query("warm up")
        org_ids = Document.objects.filter(status=Document.Status.READY).values_list("organization_id", flat=True).distinct()
        for org_id in org_ids:
            vectorstore.search(org_id, "warm up", 1)
        logger.info("RAG warm-up finished in %.1f s (%d organizations)", time.monotonic() - started, len(org_ids))
    except Exception:
        logger.exception("RAG warm-up failed; the first question will load things instead")
    finally:
        connections.close_all()


def start_in_background():
    if settings.RAG_WARMUP and is_serving():
        threading.Thread(target=warm_up, name="rag-warmup", daemon=True).start()
