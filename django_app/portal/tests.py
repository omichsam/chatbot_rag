import json

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from api.models import APIKey
from chat.tests import add_document
from rag import vectorstore
from rag.testing import RagTestMixin, platform_settings
from tenants.models import Membership, Organization
from tenants.tests import make_member

from .models import AuditLog

User = get_user_model()


class PortalTests(RagTestMixin, TestCase):
    def setUp(self):
        self.root = User.objects.create_superuser("root", password="s3cure-pass-123")
        self.org = Organization.objects.create(name="Acme")
        self.owner = make_member("owner", self.org, Membership.Role.OWNER)
        self.member = make_member("member", self.org)
        add_document(self.org, "zebras.txt", "Zebras have black and white stripes.")
        self.key, self.raw = APIKey.create_key(self.org, "website")
        self.client.force_login(self.root)

    def act(self, action, **data):
        return self.client.post(reverse("portal:organization_action", args=[self.org.pk]), {"action": action, **data})

    def test_only_superusers_can_open_portal(self):
        for name in ("overview", "organizations", "users", "system", "audit", "guide"):
            self.assertEqual(self.client.get(reverse(f"portal:{name}")).status_code, 200, name)
        self.client.force_login(self.owner)  # an org owner is not a platform admin
        self.assertEqual(self.client.get(reverse("portal:overview")).status_code, 403)
        self.assertEqual(self.act("suspend").status_code, 403)

    def test_overview_and_org_pages_show_analytics(self):
        self.client.post("/api/v1/chat", json.dumps({"question": "stripes"}), content_type="application/json",
                         HTTP_AUTHORIZATION=f"Bearer {self.raw}")
        page = self.client.get(reverse("portal:overview"))
        self.assertContains(page, "Questions per day")
        self.assertEqual(page.context["ops"]["questions"], 1)
        self.assertEqual(page.context["source_totals"]["api"], 1)
        detail = self.client.get(reverse("portal:organization", args=[self.org.pk]))
        self.assertContains(detail, "zebras.txt")
        self.assertContains(detail, "website")
        self.assertEqual(detail.context["s"]["questions"], 1)

    def test_suspend_blocks_members_and_keys_then_activate_restores(self):
        self.act("suspend")  # a reason is required
        self.org.refresh_from_db()
        self.assertTrue(self.org.is_active)
        self.act("suspend", reason="Unpaid invoice")
        self.org.refresh_from_db()
        self.assertFalse(self.org.is_active)
        self.assertEqual(self.org.suspended_reason, "Unpaid invoice")
        response = self.client.post("/api/v1/chat", json.dumps({"question": "stripes"}), content_type="application/json",
                                    HTTP_AUTHORIZATION=f"Bearer {self.raw}")
        self.assertEqual(response.status_code, 401)
        self.client.force_login(self.member)
        self.assertRedirects(self.client.get(reverse("chat:index")), reverse("tenants:create"))

        self.client.force_login(self.root)
        self.act("activate")
        self.org.refresh_from_db()
        self.assertTrue(self.org.is_active)
        self.assertEqual(list(AuditLog.objects.values_list("action", flat=True)), ["org.activate", "org.suspend"])

    def test_limits_members_and_keys(self):
        self.act("limits", monthly_query_limit="50", llm_model="gemini-test")
        self.org.refresh_from_db()
        self.assertEqual((self.org.monthly_query_limit, self.org.llm_model), (50, "gemini-test"))

        newcomer = User.objects.create_user("newcomer", password="s3cure-pass-123")
        self.act("add_member", username="newcomer", role="admin")
        self.assertEqual(Membership.objects.get(user=newcomer).role, "admin")

        owner_membership = Membership.objects.get(user=self.owner)
        self.act("set_role", membership=owner_membership.pk, role="member")
        owner_membership.refresh_from_db()
        self.assertEqual(owner_membership.role, "owner")  # last owner can't be demoted
        self.act("remove_member", membership=owner_membership.pk)
        self.assertTrue(Membership.objects.filter(pk=owner_membership.pk).exists())

        self.act("revoke_key", key=self.key.pk)
        self.key.refresh_from_db()
        self.assertFalse(self.key.is_active)

    def test_delete_requires_exact_name_and_cleans_up(self):
        self.act("schedule_delete", confirm_name="acme")  # wrong case
        self.org.refresh_from_db()
        self.assertIsNone(self.org.deletion_scheduled_at)
        self.act("delete_now", confirm_name="Acme")  # must be scheduled first
        self.assertTrue(Organization.objects.filter(pk=self.org.pk).exists())

        self.act("schedule_delete", confirm_name="Acme")
        self.org.refresh_from_db()
        self.assertFalse(self.org.is_active)
        self.assertIsNotNone(self.org.deletion_scheduled_at)
        self.act("cancel_delete")
        self.org.refresh_from_db()
        self.assertTrue(self.org.is_active)
        self.assertIsNone(self.org.deletion_scheduled_at)

        self.act("schedule_delete", confirm_name="Acme")
        org_id = self.org.pk
        response = self.act("delete_now", confirm_name="Acme")
        self.assertRedirects(response, reverse("portal:organizations"))
        self.assertFalse(Organization.objects.filter(pk=org_id).exists())
        self.assertEqual(vectorstore.count(org_id), 0)
        entry = AuditLog.objects.get(action="org.delete")
        self.assertEqual(entry.target, "Acme")

    def test_user_controls(self):
        url = reverse("portal:user_action", args=[self.member.pk])
        self.client.post(url, {"action": "reset_password"})
        page = self.client.get(reverse("portal:users"))
        temp = page.context["temp"]
        self.assertEqual(temp["username"], "member")
        self.assertTrue(self.client.login(username="member", password=temp["password"]))
        self.client.force_login(self.root)
        self.assertIsNone(self.client.get(reverse("portal:users")).context["temp"])  # shown once

        self.client.post(url, {"action": "deactivate"})
        self.member.refresh_from_db()
        self.assertFalse(self.member.is_active)
        self.client.post(url, {"action": "grant_super"})  # needs the username typed
        self.member.refresh_from_db()
        self.assertFalse(self.member.is_superuser)
        self.client.post(url, {"action": "grant_super", "confirm_name": "member"})
        self.member.refresh_from_db()
        self.assertTrue(self.member.is_superuser)

        self_url = reverse("portal:user_action", args=[self.root.pk])
        self.client.post(self_url, {"action": "revoke_super"})
        self.client.post(self_url, {"action": "deactivate"})
        self.root.refresh_from_db()
        self.assertTrue(self.root.is_superuser and self.root.is_active)

    def test_system_tests_run_and_are_logged(self):
        for kind in ("database", "vectors", "embeddings", "llm"):
            self.client.post(reverse("portal:system_test"), {"kind": kind})
            result = self.client.get(reverse("portal:system")).context["result"]
            self.assertTrue(result["ok"], (kind, result))
        self.assertEqual(AuditLog.objects.filter(action="system.test").count(), 4)

    def test_console_is_separate_from_the_workspace(self):
        console = self.client.get(reverse("portal:overview"))
        self.assertContains(console, "Platform Console")
        self.assertContains(console, 'href="/platform/organizations/"')
        self.assertContains(console, "Exit to workspace")
        workspace = self.client.get(reverse("tenants:create"))  # root has no organization of its own
        self.assertContains(workspace, "class=\"console-link\"")       # one link in
        self.assertNotContains(workspace, 'href="/platform/organizations/"')
        self.client.force_login(self.owner)
        self.assertNotContains(self.client.get(reverse("dashboard")), "class=\"console-link\"")



class AIProviderTests(TestCase):
    def setUp(self):
        self.root = User.objects.create_superuser("root", password="s3cure-pass-123")
        self.client.force_login(self.root)

    def add(self, **fields):
        data = {"kind": "groq", "name": "", "model": "llama-3.3-70b-versatile", "base_url": "", "timeout": 60,
                "is_enabled": "on", "api_key": "gsk_secret_1234", "then": "save"}
        data.update(fields)
        return self.client.post(reverse("portal:ai_provider_add"), {k: v for k, v in data.items() if v is not None})

    def test_default_gemini_provider_exists(self):
        from .models import AIProvider

        self.assertEqual(list(AIProvider.objects.values_list("kind", flat=True)), ["gemini"])
        self.assertContains(self.client.get(reverse("portal:ai_providers")), "Google Gemini")

    def test_add_provider_encrypts_key_and_keeps_it_out_of_the_audit_log(self):
        from .models import AIProvider, AuditLog

        self.assertRedirects(self.add(), reverse("portal:ai_providers"))
        groq = AIProvider.objects.get(kind="groq")
        self.assertEqual(groq.name, "Groq")
        self.assertNotIn("gsk_secret_1234", groq.api_key_encrypted)
        self.assertEqual((groq.api_key, groq.key_hint), ("gsk_secret_1234", "1234"))
        self.assertGreater(groq.priority, AIProvider.objects.get(kind="gemini").priority)
        self.assertFalse(AuditLog.objects.filter(detail__contains="gsk_secret").exists())
        page = self.client.get(reverse("portal:ai_providers"))
        self.assertContains(page, "•••• 1234")
        self.assertNotContains(page, "gsk_secret_1234")

    def test_validation(self):
        response = self.add(api_key="", model="")
        self.assertContains(response, "This provider needs an API key.")
        self.assertContains(response, "Enter the model name")
        self.assertContains(self.add(kind="custom", base_url=""), "Enter the API address")

    def test_editing_without_a_key_keeps_the_saved_one(self):
        from .models import AIProvider

        self.add()
        groq = AIProvider.objects.get(kind="groq")
        self.client.post(reverse("portal:ai_provider", args=[groq.pk]), {
            "kind": "groq", "name": "Groq backup", "model": "llama-3.1-8b-instant", "base_url": "", "timeout": 30,
            "is_enabled": "on", "api_key": "", "then": "save"})
        groq.refresh_from_db()
        self.assertEqual((groq.name, groq.model, groq.api_key), ("Groq backup", "llama-3.1-8b-instant", "gsk_secret_1234"))

    def test_test_reorder_disable_delete(self):
        from unittest import mock

        from .models import AIProvider

        self.add()
        groq = AIProvider.objects.get(kind="groq")
        action = reverse("portal:ai_provider_action", args=[groq.pk])
        with mock.patch("rag.llm.chat_generate", return_value="OK") as chat:
            self.client.post(action, {"action": "test"})
        self.assertEqual(chat.call_args.args[0], "Reply with the single word OK.")
        groq.refresh_from_db()
        self.assertTrue(groq.last_test_ok)

        self.client.post(action, {"action": "up"})
        self.assertEqual(list(AIProvider.objects.values_list("kind", flat=True)), ["groq", "gemini"])
        self.client.post(action, {"action": "disable"})
        groq.refresh_from_db()
        self.assertFalse(groq.is_enabled)
        self.client.post(action, {"action": "delete"})
        self.assertFalse(AIProvider.objects.filter(kind="groq").exists())

    def test_only_superadmins(self):
        from tenants.models import Organization
        from tenants.tests import make_member

        org = Organization.objects.create(name="Acme")
        self.client.force_login(make_member("owner", org, "owner"))
        self.assertEqual(self.client.get(reverse("portal:ai_providers")).status_code, 403)
        self.assertEqual(self.add().status_code, 403)


class ProviderClientTests(TestCase):
    """The HTTP clients parse each provider's replies (no network: responses are faked)."""

    class FakeResponse:
        def __init__(self, body=b"", lines=()):
            self.body, self.lines = body, list(lines)

        def read(self):
            return self.body

        def __iter__(self):
            return iter(self.lines)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

    def test_openai_compatible_whole_and_streamed(self):
        import json
        from unittest import mock

        from rag import llm

        whole = self.FakeResponse(json.dumps({"choices": [{"message": {"content": "Hi there"}}]}).encode())
        with self.settings(RAG_LLM_BACKEND="live"), mock.patch("rag.llm._post", return_value=whole) as post:
            self.assertEqual(llm.chat_generate("Q", "gpt-x", base_url="https://api.example/v1", api_key="k"), "Hi there")
        url, headers, body, _ = post.call_args.args
        self.assertEqual((url, headers["Authorization"], body["model"]), ("https://api.example/v1/chat/completions", "Bearer k", "gpt-x"))

        lines = [b'data: {"choices":[{"delta":{"content":"Hel"}}]}\n', b"\n",
                 b'data: {"choices":[{"delta":{"content":"lo"}}]}\n', b"data: [DONE]\n"]
        with self.settings(RAG_LLM_BACKEND="live"), mock.patch("rag.llm._post", return_value=self.FakeResponse(lines=lines)):
            self.assertEqual("".join(llm.chat_stream("Q", "gpt-x", base_url="https://api.example/v1")), "Hello")

    def test_claude_whole_and_streamed(self):
        import json
        from unittest import mock

        from rag import llm

        whole = self.FakeResponse(json.dumps({"content": [{"type": "text", "text": "Hi from Claude"}]}).encode())
        with self.settings(RAG_LLM_BACKEND="live"), mock.patch("rag.llm._post", return_value=whole) as post:
            self.assertEqual(llm.claude_generate("Q", "claude-x", base_url="https://api.anthropic.com/v1", api_key="k"),
                             "Hi from Claude")
        url, headers, body, _ = post.call_args.args
        self.assertEqual((url, headers["x-api-key"], headers["anthropic-version"]),
                         ("https://api.anthropic.com/v1/messages", "k", llm.ANTHROPIC_VERSION))

        lines = [b"event: content_block_delta\n",
                 b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"Hi "}}\n',
                 b'data: {"type":"content_block_delta","delta":{"type":"text_delta","text":"there"}}\n',
                 b'data: {"type":"message_stop"}\n']
        with self.settings(RAG_LLM_BACKEND="live"), mock.patch("rag.llm._post", return_value=self.FakeResponse(lines=lines)):
            self.assertEqual("".join(llm.claude_stream("Q", "claude-x", base_url="https://x/v1", api_key="k")), "Hi there")

    def test_http_errors_become_llm_errors(self):
        import io
        import urllib.error
        from unittest import mock

        from rag import llm

        error = urllib.error.HTTPError("https://x", 429, "Too Many", {}, io.BytesIO(b'{"error":{"message":"Rate limit reached"}}'))
        with mock.patch("urllib.request.urlopen", side_effect=error):
            with self.assertRaisesMessage(llm.LLMError, "429 Rate limit reached"):
                llm._post("https://x/v1/chat/completions", {}, {}, 5)


class PlatformSettingsTests(RagTestMixin, TestCase):
    def setUp(self):
        from django.core.cache import cache

        cache.clear()  # platform settings and sign-in counts are cached
        self.root = User.objects.create_superuser("root", password="s3cure-pass-123")
        self.org = Organization.objects.create(name="Acme")
        self.member = make_member("mia", self.org)

    def tearDown(self):
        from django.core.cache import cache

        cache.clear()
        super().tearDown()

    def form_data(self, **changes):
        from .forms import PlatformSettingsForm
        from .models import platform

        conf = platform()
        data = {}
        for name in PlatformSettingsForm.Meta.fields:
            value = getattr(conf, name)
            if isinstance(value, bool):
                if value:
                    data[name] = "on"
            else:
                data[name] = value
        for name, value in changes.items():
            if value is None:
                data.pop(name, None)
            else:
                data[name] = value
        return data

    def save(self, **changes):
        self.client.force_login(self.root)
        return self.client.post(reverse("portal:settings"), self.form_data(**changes))

    def test_superadmin_saves_settings_and_the_change_is_audited(self):
        from .models import platform

        self.assertRedirects(self.save(max_upload_mb=50, default_monthly_limit=100), reverse("portal:settings"))
        self.assertEqual((platform().max_upload_mb, platform().default_monthly_limit), (50, 100))
        entry = AuditLog.objects.get(action="platform.settings")
        self.assertIn("Largest upload (MB): 25 → 50", entry.detail)
        # New organizations start from the platform defaults.
        self.assertEqual(Organization.objects.create(name="Newco").monthly_query_limit, 100)

    def test_validation_and_access(self):
        response = self.save(max_upload_mb=9999, maintenance_mode="on", maintenance_message="")
        self.assertContains(response, "Tell people why DocAssist is offline.")
        self.assertContains(response, "Ensure this value is less than or equal to 500")
        self.client.force_login(self.member)
        self.assertEqual(self.client.get(reverse("portal:settings")).status_code, 403)

    def test_restore_defaults_for_one_tab(self):
        from .models import platform

        self.save(max_upload_mb=5, answer_cache_hours=1)
        self.client.post(reverse("portal:settings"), {"action": "reset", "section": "answers"})
        self.assertEqual((platform().max_upload_mb, platform().answer_cache_hours), (25, 24))
        self.assertTrue(AuditLog.objects.filter(action="platform.reset", target="answers").exists())

    def test_maintenance_mode_blocks_everyone_but_superadmins(self):
        self.save(maintenance_mode="on", maintenance_message="Back at noon.")
        self.client.force_login(self.member)
        page = self.client.get(reverse("dashboard"))
        self.assertEqual(page.status_code, 503)
        self.assertContains(page, "Back at noon.", status_code=503)
        self.assertEqual(self.client.get(reverse("login")).status_code, 200)
        api = self.client.post("/api/v1/chat", "{}", content_type="application/json")
        self.assertEqual((api.status_code, api.json()["error"]), (503, "Back at noon."))
        self.client.force_login(self.root)
        self.assertEqual(self.client.get(reverse("portal:overview")).status_code, 200)

    def test_announcement_and_signup_switch(self):
        self.save(announcement="Planned upgrade on Saturday.", announcement_tone="warning", allow_signup=None)
        self.client.force_login(self.member)
        self.assertContains(self.client.get(reverse("dashboard")), "Planned upgrade on Saturday.")
        self.client.logout()
        self.assertEqual(self.client.get(reverse("signup")).status_code, 404)

    def test_repeated_wrong_passwords_pause_sign_in(self):
        with platform_settings(login_max_attempts=3, login_lockout_minutes=10):
            for _ in range(3):
                self.client.post(reverse("login"), {"username": "mia", "password": "wrong"})
            response = self.client.post(reverse("login"), {"username": "mia", "password": "s3cure-pass-123"})
        self.assertContains(response, "Too many failed sign-ins. Please wait 10 minutes")
        self.assertFalse(response.wsgi_request.user.is_authenticated)
        self.assertTrue(AuditLog.objects.filter(action="security.lockout", target="mia").exists())



class ConsoleTests(RagTestMixin, TestCase):
    """The Platform Console: roles, support mode, two-factor sign-in, plans, lifecycle, security, audit, search."""

    def setUp(self):
        from django.core.cache import cache

        cache.clear()
        self.root = User.objects.create_superuser("root", password="s3cure-pass-123")
        self.org = Organization.objects.create(name="Acme")
        self.owner = make_member("owner", self.org, Membership.Role.OWNER)
        add_document(self.org, "zebras.txt", "Zebras have black and white stripes.")
        self.client.force_login(self.root)

    def tearDown(self):
        from django.core.cache import cache

        cache.clear()
        super().tearDown()

    def staff(self, username, role):
        from .models import PlatformStaff

        person = make_member(username, Organization.objects.create(name=f"{username} org"))
        PlatformStaff.objects.create(user=person, role=role)
        return person

    def test_roles_limit_what_staff_can_see_and_do(self):
        support = self.staff("sue", "support")
        self.client.force_login(support)
        for name in ("overview", "organizations", "users", "security", "audit", "system"):
            self.assertEqual(self.client.get(reverse(f"portal:{name}")).status_code, 200, name)
        for name in ("settings", "ai_providers", "plans"):
            self.assertEqual(self.client.get(reverse(f"portal:{name}")).status_code, 403, name)
        act = reverse("portal:organization_action", args=[self.org.pk])
        self.assertEqual(self.client.post(act, {"action": "suspend", "reason": "x"}).status_code, 403)

        finance = self.staff("fin", "finance")
        self.client.force_login(finance)
        self.assertEqual(self.client.get(reverse("portal:plans")).status_code, 200)
        self.assertEqual(self.client.get(reverse("portal:usage_export")).status_code, 200)
        self.assertEqual(self.client.get(reverse("portal:users")).status_code, 403)
        self.assertEqual(self.client.post(reverse("portal:support_start", args=[self.org.pk])).status_code, 403)

    def test_owner_assigns_console_roles(self):
        person = make_member("pat", self.org)
        url = reverse("portal:user_action", args=[person.pk])
        self.client.post(url, {"action": "set_staff_role", "role": "support"})
        self.assertEqual(person.platform_staff.role, "support")
        self.client.post(url, {"action": "set_staff_role", "role": ""})
        from .models import PlatformStaff

        self.assertFalse(PlatformStaff.objects.filter(user=person).exists())

    def test_support_mode_is_read_only_and_audited(self):
        self.client.post(reverse("portal:support_start", args=[self.org.pk]))
        page = self.client.get(reverse("documents:list"))
        self.assertContains(page, "Support mode")
        self.assertContains(page, "zebras.txt")
        # Any change is refused while in support mode.
        document = self.org.documents.get()
        response = self.client.post(reverse("documents:delete", args=[document.pk]))
        self.assertEqual(response.status_code, 302)
        self.assertTrue(self.org.documents.filter(pk=document.pk).exists())
        blocked = self.client.post(reverse("chat:send"), json.dumps({"question": "zebras?"}), content_type="application/json")
        self.assertEqual(blocked.status_code, 403)
        self.client.post(reverse("portal:support_exit"))
        self.assertNotContains(self.client.get(reverse("portal:overview")), "Support mode:")
        self.assertEqual(list(AuditLog.objects.filter(action__startswith="support.").values_list("action", flat=True)),
                         ["support.exit", "support.start"])

    def test_two_factor_setup_sign_in_and_backup_codes(self):
        from . import totp
        from .models import TwoFactorDevice

        self.client.post(reverse("portal:two_factor"), {"action": "start"})
        device = TwoFactorDevice.objects.get(user=self.root)
        self.client.post(reverse("portal:two_factor"), {"action": "confirm", "code": "000000"})
        self.assertFalse(TwoFactorDevice.objects.get(user=self.root).is_confirmed)
        self.client.post(reverse("portal:two_factor"), {"action": "confirm", "code": totp.code_at(device.secret)})
        device.refresh_from_db()
        self.assertTrue(device.is_confirmed)
        self.assertEqual(len(device.backup_codes), 10)
        codes = self.client.get(reverse("portal:two_factor")).context["backup_codes"]
        self.assertEqual(len(codes), 10)

        # Signing in now takes a second step.
        self.client.logout()
        response = self.client.post(reverse("login"), {"username": "root", "password": "s3cure-pass-123"})
        self.assertRedirects(response, reverse("two_factor_verify"), fetch_redirect_response=False)
        self.assertFalse(self.client.get(reverse("portal:overview")).wsgi_request.user.is_authenticated)
        self.client.post(reverse("two_factor_verify"), {"code": "111111"})
        self.client.post(reverse("two_factor_verify"), {"code": codes[0]})  # a backup code works once
        self.assertEqual(self.client.get(reverse("portal:overview")).status_code, 200)
        device.refresh_from_db()
        self.assertEqual(len(device.backup_codes), 9)
        self.assertTrue(AuditLog.objects.filter(action="security.2fa_failed", target="root").exists())

    def test_requiring_two_factor_sends_staff_to_set_it_up(self):
        from .models import PlatformSettings

        support = self.staff("sue", "support")
        with platform_settings(require_2fa=True):
            self.client.force_login(support)
            self.assertRedirects(self.client.get(reverse("portal:overview")), reverse("portal:two_factor"))
        # The owner can't turn the requirement on before setting up their own authenticator.
        self.client.force_login(self.root)
        self.client.post(reverse("portal:security_action"), {"action": "require_2fa_on"})
        self.assertFalse(PlatformSettings.objects.get(pk=1).require_2fa)

    def test_plans_set_limits_and_are_enforced(self):
        from tenants.models import Plan

        free = Plan.objects.get(name="Free")
        free.max_documents, free.max_members, free.allow_external_ai = 1, 1, False
        free.save()
        self.client.post(reverse("portal:organization_action", args=[self.org.pk]), {"action": "set_plan", "plan": free.pk})
        self.org.refresh_from_db()
        self.assertEqual((self.org.plan, self.org.monthly_query_limit), (free, free.monthly_query_limit))
        self.assertTrue(self.org.document_limit_reached())
        self.assertTrue(self.org.member_limit_reached())
        self.assertFalse(self.org.external_ai_allowed)

        from rag import pipeline

        self.assertEqual(pipeline.providers(self.org), [])  # Gemini is a cloud provider

        self.client.force_login(self.owner)
        with self.captureOnCommitCallbacks(execute=True):
            from django.core.files.uploadedfile import SimpleUploadedFile

            self.client.post(reverse("documents:upload"), {"files": [SimpleUploadedFile("more.txt", b"more text")]})
        self.assertEqual(self.org.documents.count(), 1)

    def test_scheduled_deletion_is_purged_after_the_grace_period(self):
        from datetime import timedelta

        from django.utils import timezone

        from .lifecycle import purge_due

        self.client.post(reverse("portal:organization_action", args=[self.org.pk]),
                         {"action": "schedule_delete", "confirm_name": "Acme"})
        self.assertEqual(purge_due(), [])
        self.assertEqual(purge_due(now=timezone.now() + timedelta(days=31)), ["Acme"])
        self.assertFalse(Organization.objects.filter(pk=self.org.pk).exists())
        self.assertTrue(AuditLog.objects.filter(action="org.purge", target="Acme").exists())

    def test_export_contains_data_and_files(self):
        import io
        import zipfile

        response = self.client.get(reverse("portal:organization_export", args=[self.org.pk]))
        archive = zipfile.ZipFile(io.BytesIO(response.content))
        names = archive.namelist()
        self.assertIn("organization.json", names)
        self.assertTrue(any(n.endswith("zebras.txt") for n in names))
        data = json.loads(archive.read("organization.json"))
        self.assertEqual(data["organization"]["name"], "Acme")
        self.assertEqual(data["members"][0]["username"], "owner")

    def test_security_centre_flags_keys_and_lists_failed_sign_ins(self):
        from datetime import timedelta

        from django.utils import timezone

        key, _ = APIKey.create_key(self.org, "old integration")
        APIKey.objects.filter(pk=key.pk).update(created_at=timezone.now() - timedelta(days=200))
        self.client.logout()
        self.client.post(reverse("login"), {"username": "owner", "password": "wrong"})
        self.client.force_login(self.root)
        page = self.client.get(reverse("portal:security"))
        self.assertContains(page, "old integration")
        self.assertContains(page, "Never used")
        self.assertContains(page, "rotate")
        self.assertEqual(page.context["failed_30d"], 1)

    def test_audit_filters_changes_and_csv(self):
        self.client.post(reverse("portal:organization_action", args=[self.org.pk]),
                         {"action": "limits", "monthly_query_limit": "70", "llm_model": ""})
        page = self.client.get(reverse("portal:audit") + "?action=org.&actor=root")
        self.assertEqual(page.context["page"].paginator.count, 1)
        self.assertEqual(page.context["page"][0].changes[0], ("Monthly limit", "0", "70"))
        csv_response = self.client.get(reverse("portal:audit") + "?format=csv")
        self.assertIn("org.limits", csv_response.content.decode())

    def test_console_search_finds_orgs_users_and_keys(self):
        APIKey.create_key(self.org, "website widget")
        groups = {g["title"]: g for g in self.client.get(reverse("portal:search") + "?q=acm").json()["groups"]}
        self.assertEqual(groups["Organizations"]["items"][0]["label"], "Acme")
        groups = {g["title"]: g for g in self.client.get(reverse("portal:search") + "?q=widget").json()["groups"]}
        self.assertIn("API keys", groups)
        groups = {g["title"]: g for g in self.client.get(reverse("portal:search") + "?q=owner").json()["groups"]}
        self.assertEqual(groups["Users"]["items"][0]["label"], "owner")

    def test_dashboard_shows_provider_insights(self):
        from unittest import mock

        from rag.llm import LLMError

        self.client.force_login(self.owner)
        with mock.patch("rag.llm.generate", side_effect=LLMError("429 quota")):
            self.client.post(reverse("chat:send"), json.dumps({"question": "zebras?"}), content_type="application/json")
        self.client.force_login(self.root)
        page = self.client.get(reverse("portal:overview"))
        insights = page.context["providers"]
        self.assertEqual(insights["quoted"], 1)
        self.assertEqual(insights["rows"][0]["failures"], 1)
        self.assertEqual(page.context["ops"]["backup_pct"], 100.0)
