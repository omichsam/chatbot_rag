"""Make sure the pgvector extension and passage table exist (safe to run on every start)."""
from django.core.management.base import BaseCommand, CommandError
from django.db import connection

from documents.models import Passage
from documents.pgvector_setup import passage_table_ready, pgvector_available


class Command(BaseCommand):
    help = "Create the pgvector extension and the documents_passage table if they are missing."

    def handle(self, *args, **options):
        if not pgvector_available(connection):
            raise CommandError("This database doesn't offer the pgvector extension "
                               "(use PostgreSQL with pgvector, e.g. the pgvector/pgvector Docker image).")
        with connection.cursor() as cursor:
            cursor.execute("CREATE EXTENSION IF NOT EXISTS vector")
        if passage_table_ready(connection):
            self.stdout.write("pgvector ready: extension and passage table already exist.")
            return
        with connection.schema_editor() as editor:
            editor.create_model(Passage)
        self.stdout.write(self.style.SUCCESS("pgvector ready: created the passage table and its HNSW index."))
