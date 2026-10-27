"""Create a fixed set of test accounts, one per role, for a controlled test environment.

    python manage.py seed_test_users
    python manage.py seed_test_users --password "Something-Else-1"

Safe to re-run: existing accounts get their password, role and organization reset.
Never run this on a server real users can reach; the passwords below are public.
"""
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand
from django.db import transaction

from portal.models import AuditLog, PlatformStaff, TwoFactorDevice
from tenants.models import Membership, Organization

User = get_user_model()

TEST_ORG_NAME = "Test Organization"

# username, password, what they can do: (superuser, platform staff role, organization role)
TEST_USERS = [
    ("superadmin", "Test@Super2026", True, None, None),
    ("support", "Test@Support2026", False, PlatformStaff.Role.SUPPORT, None),
    ("finance", "Test@Finance2026", False, PlatformStaff.Role.FINANCE, None),
    ("owner", "Test@Owner2026", False, None, Membership.Role.OWNER),
    ("orgadmin", "Test@OrgAdmin2026", False, None, Membership.Role.ADMIN),
    ("member", "Test@Member2026", False, None, Membership.Role.MEMBER),
]


def _describe(superuser, staff_role, org_role):
    if superuser:
        return "Superadmin (Platform Console, everything)"
    if staff_role:
        return f"Platform staff: {PlatformStaff.Role(staff_role).label}"
    return f"{Membership.Role(org_role).label} of {TEST_ORG_NAME}"


class Command(BaseCommand):
    help = "Create or reset test accounts for every role (controlled test environments only)."

    def add_arguments(self, parser):
        parser.add_argument("--password", help="Use this password for every test account instead of the defaults.")

    @transaction.atomic
    def handle(self, *args, password=None, **options):
        organization, _ = Organization.objects.get_or_create(name=TEST_ORG_NAME)
        rows = []
        for username, default_password, superuser, staff_role, org_role in TEST_USERS:
            user, created = User.objects.get_or_create(username=username)
            user.email = f"{username}@test.local"
            user.is_active = True
            user.is_superuser = user.is_staff = superuser
            user.set_password(password or default_password)
            user.save()
            # A forgotten authenticator would lock a test account out.
            TwoFactorDevice.objects.filter(user=user).delete()

            if staff_role:
                PlatformStaff.objects.update_or_create(user=user, defaults={"role": staff_role})
            else:
                PlatformStaff.objects.filter(user=user).delete()
            if org_role:
                Membership.objects.update_or_create(user=user, organization=organization, defaults={"role": org_role})

            rows.append((username, password or default_password, _describe(superuser, staff_role, org_role),
                         "created" if created else "reset"))

        AuditLog.objects.create(action="security.test_users", target=", ".join(r[0] for r in rows),
                                detail="seed_test_users from the command line")

        self.stdout.write(self.style.WARNING("Test accounts. For a controlled test environment only."))
        self.stdout.write(f"{'Username':<12} {'Password':<20} Role")
        for username, pw, role, state in rows:
            self.stdout.write(f"{username:<12} {pw:<20} {role} ({state})")
