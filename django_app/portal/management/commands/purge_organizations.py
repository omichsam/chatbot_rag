"""Delete organizations whose 30-day deletion grace period has ended (run daily, e.g. from cron)."""
from django.core.management.base import BaseCommand

from portal.lifecycle import purge_due


class Command(BaseCommand):
    help = "Delete organizations whose scheduled deletion date has passed."

    def handle(self, *args, **options):
        names = purge_due()
        self.stdout.write(f"Deleted {len(names)} organization(s){': ' + ', '.join(names) if names else ''}.")
