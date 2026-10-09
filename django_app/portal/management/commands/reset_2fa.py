"""Remove someone's authenticator app, e.g. when they lost their phone and backup codes."""
from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError

from portal.models import AuditLog, TwoFactorDevice


class Command(BaseCommand):
    help = "Remove a user's two-factor authenticator so they can set it up again."

    def add_arguments(self, parser):
        parser.add_argument("username")

    def handle(self, *args, username, **options):
        user = get_user_model().objects.filter(username=username).first()
        if user is None:
            raise CommandError(f"No user called {username}.")
        deleted, _ = TwoFactorDevice.objects.filter(user=user).delete()
        AuditLog.objects.create(action="security.2fa_reset", target=username, detail="from the command line")
        self.stdout.write(f"Two-factor sign-in {'removed' if deleted else 'was not set up'} for {username}.")
