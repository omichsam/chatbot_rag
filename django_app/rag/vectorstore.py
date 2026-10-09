"""Tenant-isolated vector storage: pgvector inside PostgreSQL, or Chroma.

RAG_VECTOR_BACKEND picks one:

- "pgvector": passages live in the documents_passage table of the main database,
  with an HNSW cosine index. One database holds everything, so one backup covers
  it and deleting a document or organization removes its passages in the same
  transaction. Needs PostgreSQL with the vector extension (the Docker setup).
- "chroma": passages live in a Chroma folder (RAG_VECTOR_DIR), one collection
  per organization. Works with any database, including SQLite.

Either way a search only ever sees one organization's passages, and passages
carry their document id so one document can be removed on its own.
"""
import threading
from dataclasses import dataclass

from django.conf import settings

from .embeddings import embed_query, embed_texts

ADD_BATCH_SIZE = 500

_clients = {}
_clients_lock = threading.Lock()


@dataclass
class Hit:
    text: str
    document_id: int
    filename: str
    page: int
    score: float


def backend():
    return getattr(settings, "RAG_VECTOR_BACKEND", "chroma")


def _pg():
    return backend() == "pgvector"


# --- Chroma -------------------------------------------------------------------------

def _client():
    # RAG_VECTOR_DIR = None keeps vectors in memory only (used by the test suite).
    path = str(settings.RAG_VECTOR_DIR) if settings.RAG_VECTOR_DIR else ":memory:"
    with _clients_lock:
        if path not in _clients:
            import chromadb
            from chromadb.config import Settings as ChromaSettings

            chroma_settings = ChromaSettings(anonymized_telemetry=False)
            _clients[path] = (
                chromadb.EphemeralClient(settings=chroma_settings) if path == ":memory:"
                else chromadb.PersistentClient(path=path, settings=chroma_settings)
            )
        return _clients[path]


def collection_name(organization_id):
    return f"{getattr(settings, 'RAG_COLLECTION_PREFIX', '')}org_{organization_id}"


def _collection(organization_id):
    return _client().get_or_create_collection(
        name=collection_name(organization_id), metadata={"hnsw:space": "cosine"}
    )


# --- pgvector ------------------------------------------------------------------------

def _passages():
    from documents.models import Passage  # documents imports rag, so import here

    return Passage.objects


# --- The interface the rest of the app uses ----------------------------------------------

def add_chunks(organization_id, document_id, filename, chunks):
    embeddings = embed_texts(chunk.text for chunk in chunks)
    if _pg():
        from documents.models import Passage

        _passages().bulk_create(
            [Passage(organization_id=organization_id, document_id=document_id, filename=filename,
                     page=chunk.page, position=i, text=chunk.text, embedding=vector)
             for i, (chunk, vector) in enumerate(zip(chunks, embeddings))],
            batch_size=ADD_BATCH_SIZE,
        )
        return
    collection = _collection(organization_id)
    for start in range(0, len(chunks), ADD_BATCH_SIZE):
        batch = chunks[start : start + ADD_BATCH_SIZE]
        collection.add(
            ids=[f"doc{document_id}-{start + i}" for i in range(len(batch))],
            documents=[chunk.text for chunk in batch],
            embeddings=embeddings[start : start + ADD_BATCH_SIZE],
            metadatas=[
                {"document_id": document_id, "filename": filename, "page": chunk.page}
                for chunk in batch
            ],
        )


def delete_document(organization_id, document_id):
    if _pg():
        _passages().filter(organization_id=organization_id, document_id=document_id).delete()
        return
    _collection(organization_id).delete(where={"document_id": document_id})


def delete_organization(organization_id):
    if _pg():
        _passages().filter(organization_id=organization_id).delete()
        return
    try:
        _client().delete_collection(collection_name(organization_id))
    except Exception:  # collection never created
        pass


def count(organization_id):
    if _pg():
        return _passages().filter(organization_id=organization_id).count()
    return _collection(organization_id).count()


def search(organization_id, query, k, exclude_document_ids=()):
    """The k chunks closest in meaning to query, skipping chunks of excluded (paused) documents."""
    exclude = [int(pk) for pk in exclude_document_ids]
    if _pg():
        from pgvector.django import CosineDistance

        rows = (_passages().filter(organization_id=organization_id).exclude(document_id__in=exclude)
                .annotate(distance=CosineDistance("embedding", embed_query(query)))
                .order_by("distance")
                .values_list("text", "document_id", "filename", "page", "distance")[:k])
        return [Hit(text=t, document_id=d, filename=f, page=p, score=1 - dist) for t, d, f, p, dist in rows]

    collection = _collection(organization_id)
    total = collection.count()
    if total == 0:
        return []
    result = collection.query(
        query_embeddings=[embed_query(query)],
        n_results=min(k, total),
        where={"document_id": {"$nin": exclude}} if exclude else None,
        include=["documents", "metadatas", "distances"],
    )
    return [
        Hit(
            text=text,
            document_id=meta["document_id"],
            filename=meta["filename"],
            page=meta.get("page", 1),
            score=1 - distance,
        )
        for text, meta, distance in zip(
            result["documents"][0], result["metadatas"][0], result["distances"][0]
        )
    ]


def stats():
    """What the health page shows: backend name, groups (organizations), passages, where they live."""
    if _pg():
        from django.db.models import Count

        groups = _passages().values("organization_id").annotate(n=Count("id"))
        return {"backend": "pgvector (PostgreSQL)", "groups": len(groups), "passages": sum(g["n"] for g in groups),
                "location": "documents_passage table, HNSW cosine index"}
    client = _client()
    cols = client.list_collections()
    return {"backend": "Chroma", "groups": len(cols),
            "passages": sum(client.get_collection(getattr(c, "name", c)).count() for c in cols),
            "location": str(settings.RAG_VECTOR_DIR)}
