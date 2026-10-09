from django.core.files.uploadedfile import SimpleUploadedFile
from django.test import TestCase
from django.urls import reverse

from rag import answer_cache, pipeline, vectorstore
from rag.testing import RagTestMixin
from tenants.models import Membership, Organization
from tenants.tests import make_member

from .models import Document


def text_file(name, content):
    return SimpleUploadedFile(name, content.encode(), content_type="text/plain")


class DocumentTests(RagTestMixin, TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme")
        self.admin = make_member("admin", self.org, Membership.Role.ADMIN)
        self.member = make_member("member", self.org)

    def upload(self, *files):
        with self.captureOnCommitCallbacks(execute=True):
            return self.client.post(reverse("documents:upload"), {"files": list(files)})

    def test_admin_upload_is_indexed(self):
        self.client.force_login(self.admin)
        self.upload(text_file("zebras.txt", "Zebras have black and white stripes. " * 50))
        document = Document.objects.get()
        self.assertEqual(document.status, Document.Status.READY)
        self.assertGreater(document.chunk_count, 1)
        self.assertEqual(vectorstore.count(self.org.pk), document.chunk_count)

    def test_member_cannot_upload(self):
        self.client.force_login(self.member)
        response = self.upload(text_file("x.txt", "hello"))
        self.assertEqual(response.status_code, 403)
        self.assertFalse(Document.objects.exists())

    def test_duplicate_upload_is_skipped(self):
        self.client.force_login(self.admin)
        self.upload(text_file("a.txt", "same content"))
        self.upload(text_file("b.txt", "same content"))
        self.assertEqual(Document.objects.count(), 1)

    def test_unsupported_type_rejected(self):
        self.client.force_login(self.admin)
        self.upload(SimpleUploadedFile("evil.exe", b"MZ"))
        self.assertFalse(Document.objects.exists())

    def test_empty_file_fails_cleanly(self):
        self.client.force_login(self.admin)
        self.upload(text_file("blank.txt", "   \n  "))
        document = Document.objects.get()
        self.assertEqual(document.status, Document.Status.FAILED)
        self.assertIn("No text", document.error)

    def test_delete_removes_only_that_documents_vectors(self):
        self.client.force_login(self.admin)
        self.upload(text_file("a.txt", "alpha content " * 100), text_file("b.txt", "beta content " * 100))
        a, b = Document.objects.get(original_name="a.txt"), Document.objects.get(original_name="b.txt")
        self.client.post(reverse("documents:delete", args=[a.pk]))
        self.assertFalse(Document.objects.filter(pk=a.pk).exists())
        self.assertEqual(vectorstore.count(self.org.pk), b.chunk_count)

    def test_other_tenant_cannot_see_or_delete(self):
        self.client.force_login(self.admin)
        self.upload(text_file("secret.txt", "acme secret plans"))
        document = Document.objects.get()

        other = Organization.objects.create(name="Rival")
        rival_admin = make_member("rival", other, Membership.Role.OWNER)
        self.client.force_login(rival_admin)
        self.assertNotContains(self.client.get(reverse("documents:list")), "secret.txt")
        self.assertEqual(self.client.post(reverse("documents:delete", args=[document.pk])).status_code, 404)
        self.assertTrue(Document.objects.filter(pk=document.pk).exists())

    def test_deleting_organization_drops_its_collection(self):
        self.client.force_login(self.admin)
        self.upload(text_file("a.txt", "alpha content"))
        org_id = self.org.pk
        self.org.delete()
        self.assertEqual(vectorstore.count(org_id), 0)


class PauseTests(RagTestMixin, TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme")
        self.admin = make_member("admin", self.org, Membership.Role.ADMIN)
        self.member = make_member("member", self.org)
        self.client.force_login(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            self.client.post(reverse("documents:upload"), {"files": [
                text_file("zebras.txt", "Zebras have black and white stripes. " * 40),
                text_file("whales.txt", "Blue whales are the largest animals. " * 40),
            ]})
        self.zebras = Document.objects.get(original_name="zebras.txt")
        self.whales = Document.objects.get(original_name="whales.txt")

    def toggle(self, document):
        return self.client.post(reverse("documents:pause", args=[document.pk]))

    def retrieved_files(self):
        return {hit.filename for hit in pipeline.retrieve(self.org, "What stripes do zebras have?")}

    def test_pause_hides_document_from_answers_and_keeps_vectors(self):
        self.assertIn("zebras.txt", self.retrieved_files())
        self.toggle(self.zebras)
        self.zebras.refresh_from_db()
        self.assertTrue(self.zebras.is_paused)
        self.assertNotIn("zebras.txt", self.retrieved_files())
        self.assertEqual(vectorstore.count(self.org.pk), self.zebras.chunk_count + self.whales.chunk_count)

    def test_resume_restores_without_reindexing(self):
        self.toggle(self.zebras)
        self.toggle(self.zebras)
        self.zebras.refresh_from_db()
        self.assertFalse(self.zebras.is_paused)
        self.assertIsNone(self.zebras.paused_at)
        self.assertIn("zebras.txt", self.retrieved_files())

    def test_all_paused_retrieves_nothing(self):
        self.toggle(self.zebras)
        self.toggle(self.whales)
        self.assertEqual(self.retrieved_files(), set())

    def test_pause_changes_answer_cache_key(self):
        before = answer_cache._key(self.org, "zebras?")
        self.toggle(self.zebras)
        paused = answer_cache._key(self.org, "zebras?")
        self.toggle(self.zebras)
        self.assertNotEqual(before, paused)
        self.assertEqual(before, answer_cache._key(self.org, "zebras?"))

    def test_paused_document_left_out_of_api_list(self):
        self.toggle(self.zebras)
        names = {d.original_name for d in Document.objects.filter(organization=self.org).answerable()}
        self.assertEqual(names, {"whales.txt"})

    def test_member_cannot_pause(self):
        self.client.force_login(self.member)
        self.assertEqual(self.toggle(self.zebras).status_code, 403)
        self.zebras.refresh_from_db()
        self.assertFalse(self.zebras.is_paused)

    def test_other_tenant_cannot_pause(self):
        other = Organization.objects.create(name="Rival")
        self.client.force_login(make_member("rival", other, Membership.Role.OWNER))
        self.assertEqual(self.toggle(self.zebras).status_code, 404)

    def test_reupload_of_paused_document_points_to_resume(self):
        self.toggle(self.zebras)
        with self.captureOnCommitCallbacks(execute=True):
            response = self.client.post(reverse("documents:upload"), {
                "files": [text_file("copy.txt", "Zebras have black and white stripes. " * 40)]}, follow=True)
        self.assertContains(response, "which is paused")
        self.assertEqual(Document.objects.count(), 2)

    def test_list_shows_paused_badge(self):
        self.toggle(self.zebras)
        response = self.client.get(reverse("documents:list"))
        self.assertContains(response, "1 paused")
        self.assertContains(response, ">Paused</span>")


class PgvectorTests(RagTestMixin, TestCase):
    """The pgvector backend. Runs where the database has pgvector (the Docker setup); skipped elsewhere."""

    def setUp(self):
        from django.db import connection

        from .pgvector_setup import passage_table_ready

        if not passage_table_ready(connection):
            self.skipTest("this database has no pgvector passage table")
        self.vectors = self.settings(RAG_VECTOR_BACKEND="pgvector")
        self.vectors.enable()
        self.addCleanup(self.vectors.disable)
        self.org = Organization.objects.create(name="Acme")
        self.admin = make_member("admin", self.org, Membership.Role.ADMIN)
        self.client.force_login(self.admin)
        with self.captureOnCommitCallbacks(execute=True):
            self.client.post(reverse("documents:upload"), {"files": [
                text_file("zebras.txt", "Zebras have black and white stripes. " * 40),
                text_file("whales.txt", "Blue whales are the largest animals. " * 40)]})

    def test_passages_are_stored_searched_and_removed(self):
        from .models import Passage

        zebras = Document.objects.get(original_name="zebras.txt")
        self.assertEqual(Passage.objects.filter(document=zebras).count(), zebras.chunk_count)
        self.assertEqual(vectorstore.count(self.org.pk), Passage.objects.filter(organization=self.org).count())
        hits = vectorstore.search(self.org.pk, "What stripes do zebras have?", 3)
        self.assertEqual(hits[0].filename, "zebras.txt")
        self.assertTrue(0 < hits[0].score <= 1)

        excluded = vectorstore.search(self.org.pk, "What stripes do zebras have?", 3, exclude_document_ids=[zebras.pk])
        self.assertNotIn("zebras.txt", {h.filename for h in excluded})

        self.client.post(reverse("documents:delete", args=[zebras.pk]))
        self.assertFalse(Passage.objects.filter(document_id=zebras.pk).exists())

    def test_other_organizations_never_match(self):
        other = Organization.objects.create(name="Rival")
        self.assertEqual(vectorstore.search(other.pk, "zebras", 5), [])
        org_id = self.org.pk
        self.org.delete()
        self.assertEqual(vectorstore.count(org_id), 0)
