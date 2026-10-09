import json

from django.test import TestCase
from django.utils import timezone

from chat.tests import add_document
from rag.testing import RagTestMixin, platform_settings
from tenants.models import Organization

from .models import APIKey


class APITests(RagTestMixin, TestCase):
    def setUp(self):
        self.acme = Organization.objects.create(name="Acme")
        self.rival = Organization.objects.create(name="Rival")
        add_document(self.acme, "zebras.txt", "Zebras have black and white stripes.")
        add_document(self.rival, "volcanoes.txt", "Volcanoes erupt lava.")
        self.key, self.raw = APIKey.create_key(self.acme, "test")

    def post(self, path, body, key=None):
        headers = {"HTTP_AUTHORIZATION": f"Bearer {key}"} if key else {}
        return self.client.post(path, json.dumps(body), content_type="application/json", **headers)

    def test_requires_valid_key(self):
        self.assertEqual(self.post("/api/v1/chat", {"question": "hi"}).status_code, 401)
        self.assertEqual(self.post("/api/v1/chat", {"question": "hi"}, "sk_wrong").status_code, 401)

    def test_chat_is_scoped_to_key_organization(self):
        response = self.post("/api/v1/chat", {"question": "stripes"}, self.raw)
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertTrue(data["success"])
        self.assertEqual(data["sources"], ["zebras.txt"])

        _, rival_raw = APIKey.create_key(self.rival, "rival")
        data = self.post("/api/v1/chat", {"question": "stripes"}, rival_raw).json()
        self.assertEqual(data["sources"], ["volcanoes.txt"])

    def test_session_continues(self):
        first = self.post("/api/v1/chat", {"question": "stripes"}, self.raw).json()
        second = self.post("/api/v1/chat", {"question": "more", "session_id": first["session_id"]}, self.raw).json()
        self.assertEqual(first["session_id"], second["session_id"])

    def test_legacy_paths_and_fields(self):
        data = self.post("/api/chat", {"query": "stripes", "session_id": "user123"}, self.raw).json()
        self.assertTrue(data["success"])
        self.assertEqual(self.client.get("/api/validate-key", HTTP_AUTHORIZATION=f"Bearer {self.raw}").status_code, 200)

    def test_x_api_key_header(self):
        response = self.client.post("/api/v1/chat", json.dumps({"question": "stripes"}),
                                    content_type="application/json", HTTP_X_API_KEY=self.raw)
        self.assertEqual(response.status_code, 200)

    def test_revoked_key_rejected(self):
        self.key.revoked_at = timezone.now()
        self.key.save()
        self.assertEqual(self.post("/api/v1/chat", {"question": "hi"}, self.raw).status_code, 401)

    def test_raw_key_not_stored(self):
        self.assertNotIn(self.raw, {self.key.prefix, self.key.key_hash})
        self.assertEqual(self.key.key_hash, APIKey.hash_key(self.raw))

    def test_health_hides_counts_without_key(self):
        self.assertNotIn("documents_count", self.client.get("/api/v1/health").json())
        data = self.client.get("/api/v1/health", HTTP_AUTHORIZATION=f"Bearer {self.raw}").json()
        self.assertTrue(data["documents_loaded"])

    def test_documents_endpoint(self):
        data = self.client.get("/api/v1/documents", HTTP_AUTHORIZATION=f"Bearer {self.raw}").json()
        self.assertEqual([d["filename"] for d in data["documents"]], ["zebras.txt"])

    def test_ai_outage_still_answers_from_the_backup(self):
        from unittest import mock

        from rag.llm import LLMError

        with mock.patch("rag.llm.generate", side_effect=LLMError("429 quota")):
            response = self.post("/api/v1/chat", {"question": "stripes"}, self.raw)
        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.json()["success"])

    def test_ai_outage_returns_503_without_a_backup(self):
        from unittest import mock

        from rag.llm import LLMError

        with platform_settings(quoted_fallback=False), mock.patch("rag.llm.generate", side_effect=LLMError("429 quota")):
            response = self.post("/api/v1/chat", {"question": "stripes"}, self.raw)
        self.assertEqual(response.status_code, 503)
        self.assertFalse(response.json()["success"])

    def test_validation_errors(self):
        self.assertEqual(self.post("/api/v1/chat", {"question": "  "}, self.raw).status_code, 400)
        response = self.client.post("/api/v1/chat", "not json", content_type="application/json",
                                    HTTP_AUTHORIZATION=f"Bearer {self.raw}")
        self.assertEqual(response.status_code, 400)


class DeveloperPageTests(RagTestMixin, TestCase):
    def test_every_member_can_open_the_playground(self):
        from tenants.models import Membership
        from tenants.tests import make_member

        org = Organization.objects.create(name="Acme")
        for role in (Membership.Role.MEMBER, Membership.Role.ADMIN):
            user = make_member(f"user-{role}", org, role)
            self.client.force_login(user)
            page = self.client.get("/developer/")
            self.assertContains(page, "Send request")
            self.assertContains(page, "/api/v1/")
            self.assertContains(page, 'href="/developer/"')   # in the menu, unlocked

    def test_requires_login(self):
        self.assertEqual(self.client.get("/developer/").status_code, 302)
