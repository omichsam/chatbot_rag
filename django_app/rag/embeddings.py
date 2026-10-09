"""Text embeddings.

The sentence-transformers model is loaded once per process, on first use.
The "fake" backend hashes words into a vector so tests run instantly and
offline while still ranking overlapping text as similar.
"""
import hashlib
import math
import re
import threading

from django.conf import settings

FAKE_DIMENSIONS = 384

_model = None
_model_lock = threading.Lock()


def _sentence_transformer():
    global _model
    with _model_lock:
        if _model is None:
            from sentence_transformers import SentenceTransformer

            _model = SentenceTransformer(settings.RAG_EMBEDDING_MODEL)
    return _model


def _fake_embed(text):
    vector = [0.0] * FAKE_DIMENSIONS
    for token in re.findall(r"\w+", text.lower()):
        vector[int(hashlib.md5(token.encode()).hexdigest(), 16) % FAKE_DIMENSIONS] += 1.0
    norm = math.sqrt(sum(v * v for v in vector))
    if norm == 0:
        vector[0] = 1.0
        return vector
    return [v / norm for v in vector]


def embed_texts(texts):
    texts = list(texts)
    if settings.RAG_EMBEDDING_BACKEND == "fake":
        return [_fake_embed(text) for text in texts]
    vectors = _sentence_transformer().encode(
        texts, normalize_embeddings=True, batch_size=32, show_progress_bar=False
    )
    return vectors.tolist()


def embed_query(text):
    return embed_texts([text])[0]
