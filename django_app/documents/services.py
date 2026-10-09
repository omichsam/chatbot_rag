"""Document ingestion: extract text, chunk, embed, and store in the tenant's collection."""
import hashlib
import logging
from concurrent.futures import ThreadPoolExecutor

from django.conf import settings
from django.db import close_old_connections, connections
from django.utils import timezone

from rag import chunking, loaders, vectorstore

from .models import Document

logger = logging.getLogger(__name__)

_executor = None


def compute_checksum(uploaded_file):
    digest = hashlib.sha256()
    for block in uploaded_file.chunks():
        digest.update(block)
    uploaded_file.seek(0)
    return digest.hexdigest()


def enqueue_ingest(document_id):
    """Index a document now ("sync") or in the background thread pool ("thread")."""
    global _executor
    if settings.RAG_INGEST_MODE == "sync":
        ingest_document(document_id)
        return
    if _executor is None:
        _executor = ThreadPoolExecutor(max_workers=settings.RAG_INGEST_WORKERS, thread_name_prefix="ingest")
    _executor.submit(_ingest_in_thread, document_id)


def _ingest_in_thread(document_id):
    close_old_connections()
    try:
        ingest_document(document_id)
    except Exception:
        logger.exception("Ingestion crashed for document %s", document_id)
    finally:
        connections.close_all()


def ingest_document(document_id):
    document = Document.objects.filter(pk=document_id).first()
    if document is None:
        return
    # .update() rather than .save() so a document deleted mid-way isn't recreated.
    Document.objects.filter(pk=document_id).update(status=Document.Status.PROCESSING, error="")

    try:
        pages = loaders.extract_pages(document.file.path, document.original_name)
        chunks = chunking.split_pages(pages)
        if not chunks:
            raise ValueError("No text could be extracted. Scanned PDFs need OCR first.")
        vectorstore.delete_document(document.organization_id, document.pk)
        vectorstore.add_chunks(document.organization_id, document.pk, document.original_name, chunks)
    except Exception as exc:
        logger.warning("Ingestion failed for document %s: %s", document_id, exc)
        Document.objects.filter(pk=document_id).update(
            status=Document.Status.FAILED, error=str(exc)[:1000], processed_at=timezone.now()
        )
        return

    updated = Document.objects.filter(pk=document_id).update(
        status=Document.Status.READY,
        page_count=len(pages),
        chunk_count=len(chunks),
        processed_at=timezone.now(),
    )
    if not updated:  # deleted while we were indexing
        vectorstore.delete_document(document.organization_id, document.pk)


def delete_document(document):
    # Vector and file cleanup happen in the post_delete signal.
    document.delete()
