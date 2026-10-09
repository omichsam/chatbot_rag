import uuid

from django.conf import settings
from django.db import models
from pgvector.django import HnswIndex, VectorField


def document_upload_to(instance, filename):
    return f"documents/org_{instance.organization_id}/{uuid.uuid4().hex}_{filename}"


class DocumentQuerySet(models.QuerySet):
    def answerable(self):
        """Indexed and not paused: the documents answers are drawn from."""
        return self.filter(status=Document.Status.READY, is_paused=False)

    def paused_ids(self, organization_id):
        return list(self.filter(organization_id=organization_id, is_paused=True).values_list("pk", flat=True))


class Document(models.Model):
    class Status(models.TextChoices):
        PENDING = "pending", "Queued"
        PROCESSING = "processing", "Processing"
        READY = "ready", "Ready"
        FAILED = "failed", "Failed"

    organization = models.ForeignKey("tenants.Organization", on_delete=models.CASCADE, related_name="documents")
    uploaded_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    file = models.FileField(upload_to=document_upload_to, max_length=500)
    original_name = models.CharField(max_length=255)
    size = models.PositiveBigIntegerField(default=0)
    checksum = models.CharField(max_length=64, help_text="SHA-256 of the file, used to spot duplicates.")
    status = models.CharField(max_length=12, choices=Status.choices, default=Status.PENDING)
    error = models.TextField(blank=True)
    page_count = models.PositiveIntegerField(default=0)
    chunk_count = models.PositiveIntegerField(default=0)
    created_at = models.DateTimeField(auto_now_add=True)
    processed_at = models.DateTimeField(null=True, blank=True)
    # Paused documents keep their file and vectors but are left out of every search.
    is_paused = models.BooleanField(default=False)
    paused_at = models.DateTimeField(null=True, blank=True)

    objects = DocumentQuerySet.as_manager()

    class Meta:
        ordering = ["-created_at"]
        indexes = [models.Index(fields=["organization", "checksum"])]

    def __str__(self):
        return self.original_name

    @property
    def is_busy(self):
        return self.status in (self.Status.PENDING, self.Status.PROCESSING)


EMBEDDING_DIMENSIONS = 384  # all-MiniLM-L6-v2 (and the test suite's fake embeddings)


class Passage(models.Model):
    """One searchable piece of a document, with its embedding (RAG_VECTOR_BACKEND = "pgvector").

    The table and the vector extension are created by migration 0003 only on a
    PostgreSQL server that has pgvector; elsewhere the Chroma backend is used and
    this table is not needed (`manage.py setup_pgvector` creates it later).
    """

    # No database cascade: the table may not exist (Chroma backend), and Django would
    # query it on every delete. The delete signals remove passages instead, inside the
    # same transaction as the delete (documents/signals.py, tenants/signals.py).
    organization = models.ForeignKey("tenants.Organization", on_delete=models.DO_NOTHING, db_constraint=False,
                                     related_name="+")
    document = models.ForeignKey(Document, on_delete=models.DO_NOTHING, db_constraint=False, related_name="passages")
    filename = models.CharField(max_length=255)
    page = models.PositiveIntegerField(default=1)
    position = models.PositiveIntegerField(default=0)
    text = models.TextField()
    embedding = VectorField(dimensions=EMBEDDING_DIMENSIONS)

    class Meta:
        indexes = [
            models.Index(fields=["organization", "document"]),
            HnswIndex(name="passage_embedding_hnsw", fields=["embedding"], m=16, ef_construction=64,
                      opclasses=["vector_cosine_ops"]),
        ]

    def __str__(self):
        return f"{self.filename} p.{self.page} #{self.position}"
