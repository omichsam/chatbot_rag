"""Helpers for the pgvector passage table (used by migration 0003 and `manage.py setup_pgvector`)."""


def pgvector_available(connection):
    """True when the database is PostgreSQL and the vector extension can be installed."""
    if connection.vendor != "postgresql":
        return False
    with connection.cursor() as cursor:
        cursor.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'vector'")
        return cursor.fetchone() is not None


def passage_table_ready(connection):
    from .models import Passage

    return Passage._meta.db_table in connection.introspection.table_names()
