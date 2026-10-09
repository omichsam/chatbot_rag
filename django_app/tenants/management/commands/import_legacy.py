"""Import the Streamlit app's data (users.json, api_keys.json, uploaded_files/) into one organization.

    python manage.py import_legacy --org-name "ICTA"

Safe to re-run: existing users, keys and documents (by checksum) are skipped.
"""
import hashlib
import json
from datetime import datetime
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.core.files import File
from django.core.management.base import BaseCommand, CommandError
from django.db import transaction
from django.utils import timezone

from api.models import APIKey
from documents import services
from documents.models import Document
from rag.loaders import SUPPORTED_EXTENSIONS
from tenants.models import Membership, Organization

User = get_user_model()

# The Streamlit app auto-created admin/admin123; never carry that password over.
DEFAULT_ADMIN_HASH = hashlib.sha256(b"admin123").hexdigest()


def _parse_time(value):
    try:
        parsed = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    return timezone.make_aware(parsed) if timezone.is_naive(parsed) else parsed


class Command(BaseCommand):
    help = "Import users, API keys and documents from the legacy Streamlit app into an organization."

    def add_arguments(self, parser):
        parser.add_argument("--source", default=str(settings.BASE_DIR.parent),
                            help="Folder containing users.json, api_keys.json and uploaded_files/ (default: the parent folder).")
        parser.add_argument("--org-name", default="Default Organization", help="Organization to import into (created if missing).")
        parser.add_argument("--skip-documents", action="store_true", help="Don't import or index documents.")

    def handle(self, *args, **options):
        source = Path(options["source"])
        if not source.is_dir():
            raise CommandError(f"{source} is not a folder.")

        organization, created = Organization.objects.get_or_create(name=options["org_name"])
        self.stdout.write(f"{'Created' if created else 'Using'} organization: {organization.name} (id {organization.pk})")

        users = self._import_users(source / "users.json", organization)
        self._import_keys(source / "api_keys.json", organization, users)
        if not options["skip_documents"]:
            self._import_documents(source / "uploaded_files", organization)
        self.stdout.write(self.style.SUCCESS("Import finished."))

    @transaction.atomic
    def _import_users(self, path, organization):
        if not path.exists():
            self.stdout.write(f"No {path.name}; skipping users.")
            return {}
        data = json.loads(path.read_text(encoding="utf-8"))
        users = {}
        has_owner = organization.memberships.filter(role=Membership.Role.OWNER).exists()

        # Admins first so the first admin becomes the owner.
        for username, record in sorted(data.items(), key=lambda item: not item[1].get("is_admin")):
            user = User.objects.filter(username=username).first()
            if user is None:
                user = User(username=username)
                legacy_hash = record.get("password_hash", "")
                if legacy_hash == DEFAULT_ADMIN_HASH:
                    user.set_unusable_password()
                    self.stdout.write(self.style.WARNING(
                        f"  {username}: had the default password admin123, so it was NOT imported. "
                        f"Set one with: python manage.py changepassword {username}"
                    ))
                elif legacy_hash:
                    user.password = f"legacy_sha256$${legacy_hash}"
                else:
                    user.set_unusable_password()
                user.save()
                joined = _parse_time(record.get("created_at"))
                if joined:
                    User.objects.filter(pk=user.pk).update(date_joined=joined)
                self.stdout.write(f"  user {username}: created")
            else:
                self.stdout.write(f"  user {username}: already exists")
            users[username] = user

            if record.get("is_admin"):
                role = Membership.Role.ADMIN if has_owner else Membership.Role.OWNER
                has_owner = True
            else:
                role = Membership.Role.MEMBER
            _membership, made = Membership.objects.get_or_create(user=user, organization=organization, defaults={"role": role})
            if made:
                self.stdout.write(f"    -> {role} of {organization.name}")
        return users

    @transaction.atomic
    def _import_keys(self, path, organization, users):
        if not path.exists():
            self.stdout.write(f"No {path.name}; skipping API keys.")
            return
        data = json.loads(path.read_text(encoding="utf-8"))
        for raw_key, record in data.items():
            if APIKey.objects.filter(key_hash=APIKey.hash_key(raw_key)).exists():
                self.stdout.write(f"  key {record.get('name')}: already imported")
                continue
            api_key, _ = APIKey.create_key(
                organization, record.get("name") or "Imported key",
                created_by=users.get(record.get("username")), raw_key=raw_key,
            )
            if not record.get("is_active", True):
                api_key.revoked_at = timezone.now()
                api_key.save(update_fields=["revoked_at"])
            self.stdout.write(f"  key {api_key.name}: imported ({'active' if api_key.is_active else 'revoked'})")

    def _import_documents(self, folder, organization):
        if not folder.is_dir():
            self.stdout.write(f"No {folder.name}/; skipping documents.")
            return
        for path in sorted(folder.iterdir()):
            if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
                continue
            checksum = hashlib.sha256(path.read_bytes()).hexdigest()
            if Document.objects.filter(organization=organization, checksum=checksum).exclude(status=Document.Status.FAILED).exists():
                self.stdout.write(f"  document {path.name}: already imported")
                continue
            document = Document(organization=organization, original_name=path.name, size=path.stat().st_size, checksum=checksum)
            with path.open("rb") as handle:
                document.file.save(path.name, File(handle), save=True)
            self.stdout.write(f"  document {path.name}: indexing...")
            # Always index inline: background threads would die with this command.
            services.ingest_document(document.pk)
            document.refresh_from_db()
            status = self.style.SUCCESS(f"{document.chunk_count} chunks") if document.status == Document.Status.READY else self.style.ERROR(f"failed: {document.error}")
            self.stdout.write(f"    -> {status}")
