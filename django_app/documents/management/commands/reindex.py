"""Rebuild the search index from the stored files, e.g. after switching vector backend."""
from django.core.management.base import BaseCommand

from documents.models import Document
from documents.services import ingest_document
from rag import vectorstore


class Command(BaseCommand):
    help = "Re-extract, re-embed and re-store documents' passages in the current vector backend."

    def add_arguments(self, parser):
        parser.add_argument("--organization", type=int, help="Only this organization's documents (id).")
        parser.add_argument("--missing", action="store_true",
                            help="Only documents that have no passages in the current backend yet.")

    def handle(self, *args, **options):
        documents = Document.objects.exclude(status=Document.Status.PENDING).order_by("organization_id", "pk")
        if options["organization"]:
            documents = documents.filter(organization_id=options["organization"])
        self.stdout.write(f"Vector backend: {vectorstore.backend()}")
        done = failed = skipped = 0
        for document in documents:
            if options["missing"] and _has_passages(document):
                skipped += 1
                continue
            ingest_document(document.pk)  # replaces this document's passages
            document.refresh_from_db()
            if document.status == Document.Status.READY:
                done += 1
                self.stdout.write(f"  {document.original_name}: {document.chunk_count} passages")
            else:
                failed += 1
                self.stdout.write(self.style.WARNING(f"  {document.original_name}: failed ({document.error})"))
        self.stdout.write(self.style.SUCCESS(f"Re-indexed {done}, failed {failed}, skipped {skipped}."))


def _has_passages(document):
    if vectorstore.backend() == "pgvector":
        return document.passages.exists()
    return bool(vectorstore._collection(document.organization_id).get(where={"document_id": document.pk}, limit=1)["ids"])
