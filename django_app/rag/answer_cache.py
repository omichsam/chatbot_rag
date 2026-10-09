"""Reuse answers to repeated questions so they come back instantly.

Only the first question of a conversation is cached (follow-ups depend on
history). The key includes the organization's documents and settings, so
uploading, deleting or re-indexing a document, or changing the model or
instructions, starts fresh automatically.
"""
import hashlib
import re

from django.core.cache import cache
from django.db.models import Count, Max

from .vectorstore import Hit


def _seconds():
    from portal.models import platform  # portal imports rag, so import here

    return platform().answer_cache_hours * 3600


def _version(organization):
    from documents.models import Document

    docs = Document.objects.filter(organization=organization).answerable().aggregate(
        n=Count("id"), last=Max("processed_at"))
    # Pausing or resuming changes what can be retrieved, so it must change the key too.
    paused = ",".join(map(str, sorted(Document.objects.paused_ids(organization.pk))))
    raw = f"{docs['n']}|{docs['last']}|{paused}|{organization.answer_fingerprint()}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def _key(organization, question):
    normalized = re.sub(r"\s+", " ", question.strip().lower()).rstrip("?!. ")
    digest = hashlib.sha256(normalized.encode()).hexdigest()[:24]
    return f"answer:{organization.pk}:{_version(organization)}:{digest}"


def get(organization, question):
    """(text, hits) or None."""
    if not _seconds() or not organization.cache_answers:
        return None
    found = cache.get(_key(organization, question))
    if not found:
        return None
    return found["text"], [Hit(**h) for h in found["hits"]]


def put(organization, question, text, hits):
    if _seconds() and organization.cache_answers and hits:
        cache.set(_key(organization, question), {"text": text, "hits": [h.__dict__ for h in hits]},
                  _seconds())
