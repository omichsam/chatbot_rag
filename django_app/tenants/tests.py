import json
import hashlib

from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from rag.testing import RagTestMixin

from .middleware import SESSION_KEY
from .models import Membership, Organization

User = get_user_model()


def make_member(username, organization, role=Membership.Role.MEMBER, password="s3cure-pass-123"):
    user = User.objects.create_user(username=username, password=password)
    Membership.objects.create(user=user, organization=organization, role=role)
    return user


class SignupTests(RagTestMixin, TestCase):
    def test_signup_creates_owned_organization_and_logs_in(self):
        response = self.client.post(reverse("signup"), {
            "username": "alice", "email": "", "organization_name": "Acme Ltd",
            "password1": "s3cure-pass-123", "password2": "s3cure-pass-123",
        })
        self.assertRedirects(response, reverse("documents:list"))
        membership = Membership.objects.get(user__username="alice")
        self.assertEqual(membership.role, Membership.Role.OWNER)
        self.assertEqual(membership.organization.slug, "acme-ltd")

    def test_slugs_are_unique(self):
        self.assertEqual(Organization.objects.create(name="Acme").slug, "acme")
        self.assertEqual(Organization.objects.create(name="Acme").slug, "acme-2")


class RememberMeTests(RagTestMixin, TestCase):
    def setUp(self):
        make_member("bob", Organization.objects.create(name="Acme"))

    def test_remember_me_keeps_session(self):
        response = self.client.post(reverse("login"), {"username": "bob", "password": "s3cure-pass-123", "remember": "1"})
        self.assertRedirects(response, reverse("dashboard"))
        self.assertFalse(self.client.session.get_expire_at_browser_close())

    def test_without_remember_me_session_ends_with_browser(self):
        self.client.post(reverse("login"), {"username": "bob", "password": "s3cure-pass-123"})
        self.assertTrue(self.client.session.get_expire_at_browser_close())


class LegacyPasswordTests(TestCase):
    def test_legacy_sha256_hash_logs_in_and_is_upgraded(self):
        user = User.objects.create(username="sam")
        user.password = "legacy_sha256$$" + hashlib.sha256(b"old-password-1").hexdigest()
        user.save()

        self.assertTrue(self.client.login(username="sam", password="old-password-1"))
        user.refresh_from_db()
        self.assertTrue(user.password.startswith("pbkdf2_sha256$"))

    def test_wrong_password_fails(self):
        user = User.objects.create(username="sam")
        user.password = "legacy_sha256$$" + hashlib.sha256(b"old-password-1").hexdigest()
        user.save()
        self.assertFalse(self.client.login(username="sam", password="nope"))


class MembershipTests(RagTestMixin, TestCase):
    def setUp(self):
        self.org = Organization.objects.create(name="Acme")
        self.owner = make_member("owner", self.org, Membership.Role.OWNER)
        self.admin = make_member("admin", self.org, Membership.Role.ADMIN)
        self.member = make_member("member", self.org)

    def test_members_page_requires_admin(self):
        self.client.force_login(self.member)
        self.assertEqual(self.client.get(reverse("tenants:members")).status_code, 403)
        self.client.force_login(self.admin)
        self.assertEqual(self.client.get(reverse("tenants:members")).status_code, 200)

    def test_admin_can_create_member_account(self):
        self.client.force_login(self.admin)
        self.client.post(reverse("tenants:members"), {"username": "newbie", "password": "s3cure-pass-123", "role": "member"})
        self.assertTrue(Membership.objects.filter(user__username="newbie", organization=self.org).exists())

    def test_admin_cannot_remove_or_create_owner(self):
        self.client.force_login(self.admin)
        owner_membership = Membership.objects.get(user=self.owner)
        response = self.client.post(reverse("tenants:update_member", args=[owner_membership.pk]), {"action": "remove"})
        self.assertEqual(response.status_code, 403)
        member_membership = Membership.objects.get(user=self.member)
        response = self.client.post(reverse("tenants:update_member", args=[member_membership.pk]), {"action": "role", "role": "owner"})
        self.assertEqual(response.status_code, 403)

    def test_cannot_manage_other_organizations_members(self):
        other = Organization.objects.create(name="Other")
        stranger = make_member("stranger", other)
        self.client.force_login(self.owner)
        response = self.client.post(
            reverse("tenants:update_member", args=[Membership.objects.get(user=stranger).pk]), {"action": "remove"}
        )
        self.assertEqual(response.status_code, 404)
        self.assertTrue(Membership.objects.filter(user=stranger).exists())


class SwitchOrganizationTests(RagTestMixin, TestCase):
    def test_switch_only_to_own_organizations(self):
        first = Organization.objects.create(name="First")
        second = Organization.objects.create(name="Second")
        foreign = Organization.objects.create(name="Foreign")
        user = make_member("u", first)
        Membership.objects.create(user=user, organization=second)
        self.client.force_login(user)

        self.client.post(reverse("tenants:switch"), {"organization": second.pk})
        self.assertEqual(self.client.session[SESSION_KEY], second.pk)

        response = self.client.post(reverse("tenants:switch"), {"organization": foreign.pk})
        self.assertEqual(response.status_code, 404)
        self.assertEqual(self.client.session[SESSION_KEY], second.pk)

    def test_user_without_organization_is_sent_to_create_one(self):
        user = User.objects.create_user("lonely", password="s3cure-pass-123")
        self.client.force_login(user)
        self.assertRedirects(self.client.get(reverse("chat:index")), reverse("tenants:create"))


class SuperadminTests(TestCase):
    def test_superadmin_without_organization_lands_in_admin(self):
        User.objects.create_superuser("root", password="s3cure-pass-123")
        self.client.login(username="root", password="s3cure-pass-123")
        self.assertRedirects(self.client.get(reverse("dashboard")), reverse("portal:overview"))
        self.assertEqual(self.client.get(reverse("admin:index")).status_code, 200)


class ShellTests(RagTestMixin, TestCase):
    """The redesigned menu, access requests, notifications and sign-in/out notices."""

    def setUp(self):
        self.org = Organization.objects.create(name="Acme")
        self.owner = make_member("owner", self.org, Membership.Role.OWNER)
        self.member = make_member("member", self.org)

    def test_admin_menu_has_all_sections_member_gets_admin_tools_card(self):
        self.client.force_login(self.owner)
        page = self.client.get(reverse("dashboard"))
        titles = [s["title"] for s in page.context["menu"]["sections"]]
        self.assertEqual(titles, ["Workspace", "Insights", "Build", "Admin"])
        self.assertNotContains(page, "Admin tools ·")

        self.client.force_login(self.member)
        page = self.client.get(reverse("dashboard"))
        self.assertEqual([s["title"] for s in page.context["menu"]["sections"]], ["Workspace", "Build"])
        self.assertContains(page, "Admin tools · 7")
        self.assertContains(page, "Request access")

    def test_menu_prefs_pin_hide_order_are_per_person(self):
        self.client.force_login(self.owner)
        url = reverse("tenants:menu_prefs")
        body = {"order": ["chat", "home"], "hidden": ["developer", "home"], "pinned": ["gaps", "nope"]}
        self.client.post(url, json.dumps(body), content_type="application/json")
        menu = self.client.get(reverse("dashboard")).context["menu"]
        self.assertEqual([i["id"] for i in menu["pinned"]], ["gaps"])
        workspace = [i["id"] for i in menu["sections"][0]["items"]]
        self.assertEqual(workspace[:2], ["chat", "home"])             # reordered, home can't be hidden
        build = next(s for s in menu["sections"] if s["id"] == "build")
        self.assertNotIn("developer", [i["id"] for i in build["items"]])

        self.client.force_login(self.member)                          # someone else's menu is untouched
        member_menu = self.client.get(reverse("dashboard")).context["menu"]
        self.assertEqual(member_menu["pinned"], [])
        self.client.post(url, json.dumps({"pinned": ["settings"]}), content_type="application/json")
        self.assertEqual(Membership.objects.get(user=self.member).menu_prefs["pinned"], [])  # not allowed

        self.client.force_login(self.owner)
        self.client.post(url, json.dumps({"reset": True}), content_type="application/json")
        self.assertEqual(Membership.objects.get(user=self.owner).menu_prefs, {})

    def test_access_request_flow(self):
        self.client.force_login(self.member)
        self.client.post(reverse("tenants:request_access"), {"next": "/dashboard/"})
        self.assertContains(self.client.get(reverse("dashboard")), "Request sent")

        self.client.force_login(self.owner)
        home = self.client.get(reverse("dashboard"))
        self.assertContains(home, "member asked for admin access")
        self.assertEqual(home.context["menu"]["sections"][3]["items"][0]["badge"]["kind"], "alert")
        from .models import AccessRequest
        request_id = AccessRequest.objects.get().pk
        self.client.post(reverse("tenants:decide_access", args=[request_id]), {"decision": "approve"})
        self.assertEqual(Membership.objects.get(user=self.member).role, Membership.Role.ADMIN)

    def test_sign_in_and_out_notices(self):
        response = self.client.post(reverse("login"), {"username": "owner", "password": "s3cure-pass-123", "remember": "1"}, follow=True)
        self.assertContains(response, "Welcome back, owner.")
        response = self.client.post(reverse("logout"), follow=True)
        self.assertContains(response, "You&#x27;ve been signed out")
        self.assertRedirects(response, reverse("login"))

    def test_major_actions_ask_for_confirmation(self):
        self.client.force_login(self.owner)
        page = self.client.get(reverse("tenants:members"))
        self.assertContains(page, 'data-confirm="Remove member from Acme?"')
        self.assertContains(page, 'data-confirm-title="Sign out?"')
        self.assertNotContains(page, "return confirm(")

    def test_search_and_history(self):
        self.client.force_login(self.member)
        groups = self.client.get(reverse("chat:search") + "?q=docu").json()["groups"]
        titles = [g["title"] for g in groups]
        self.assertIn("Ask", titles)
        self.assertIn("Go to", titles)                               # Documents page
        self.assertEqual(self.client.get(reverse("chat:history")).status_code, 200)


class MemberVisibilityTests(RagTestMixin, TestCase):
    def test_answer_time_and_ai_usage_are_admin_only(self):
        org = Organization.objects.create(name="Acme")
        owner = make_member("owner", org, Membership.Role.OWNER)
        member = make_member("member", org)
        self.client.force_login(owner)
        page = self.client.get(reverse("dashboard"))
        self.assertContains(page, "Avg. answer time")
        self.assertContains(page, "AI usage today")
        self.client.force_login(member)
        page = self.client.get(reverse("dashboard"))
        self.assertNotContains(page, "Avg. answer time")
        self.assertNotContains(page, "AI usage today")
        self.assertIsNone(page.context["ai_status"])


class SeedTestUsersTests(TestCase):
    def test_creates_every_role_and_is_safe_to_rerun(self):
        from io import StringIO

        from django.core.management import call_command
        from portal.models import PlatformStaff

        call_command("seed_test_users", stdout=StringIO())
        User.objects.filter(username="member").update(is_active=False)
        call_command("seed_test_users", password="Other-pass-123", stdout=StringIO())

        self.assertTrue(User.objects.get(username="superadmin").is_superuser)
        self.assertEqual(PlatformStaff.objects.get(user__username="support").role, PlatformStaff.Role.SUPPORT)
        org = Organization.objects.get(name="Test Organization")
        self.assertEqual(org.memberships.get(user__username="owner").role, Membership.Role.OWNER)
        self.assertEqual(org.memberships.count(), 3)
        member = User.objects.get(username="member")
        self.assertTrue(member.is_active)
        self.assertTrue(member.check_password("Other-pass-123"))
