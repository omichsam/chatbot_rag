import atexit
from contextlib import contextmanager
import shutil
import tempfile
import uuid
from pathlib import Path

from django.conf import settings
from django.test import override_settings

# Vectors live in memory during tests (opening a new on-disk Chroma database per
# test made it intermittently lose indexes). Each test gets fresh collection
# names because organization ids are reused once a test's transaction rolls back.
_TMP = Path(tempfile.mkdtemp(prefix="ragtest_"))
atexit.register(shutil.rmtree, _TMP, ignore_errors=True)


def _fresh_prefix():
    return f"t{uuid.uuid4().hex[:10]}_"


class RagTestMixin:
    """Offline, fast RAG settings: fake embeddings, echo LLM, in-memory vectors."""

    @classmethod
    def setUpClass(cls):
        cls._rag_settings = override_settings(
            RAG_EMBEDDING_BACKEND="fake",
            RAG_LLM_BACKEND="echo",
            RAG_INGEST_MODE="sync",
            RAG_VECTOR_DIR=None,
            RAG_COLLECTION_PREFIX=_fresh_prefix(),
            MEDIA_ROOT=_TMP / "media",
        )
        cls._rag_settings.enable()
        super().setUpClass()

    @classmethod
    def tearDownClass(cls):
        super().tearDownClass()
        cls._rag_settings.disable()

    def tearDown(self):
        # The next test starts with an empty vector store.
        settings.RAG_COLLECTION_PREFIX = _fresh_prefix()
        super().tearDown()


@contextmanager
def platform_settings(**values):
    """Change platform settings (portal.models.PlatformSettings) for the block, then put them back."""
    from portal.models import PlatformSettings, platform

    conf = PlatformSettings.objects.get(pk=platform().pk or 1) if PlatformSettings.objects.exists() else platform()
    before = {name: getattr(conf, name) for name in values}
    for name, value in values.items():
        setattr(conf, name, value)
    conf.save()
    try:
        yield conf
    finally:
        for name, value in before.items():
            setattr(conf, name, value)
        conf.save()
