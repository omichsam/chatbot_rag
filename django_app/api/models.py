import hashlib
import secrets

from django.conf import settings
from django.db import models


class APIKey(models.Model):
    """An organization-scoped API key. Only a SHA-256 of the key is stored."""

    KEY_PREFIX = "sk_"
    DISPLAY_CHARS = 12

    organization = models.ForeignKey("tenants.Organization", on_delete=models.CASCADE, related_name="api_keys")
    name = models.CharField(max_length=100)
    prefix = models.CharField(max_length=16, help_text="First characters of the key, for recognizing it.")
    key_hash = models.CharField(max_length=64, unique=True)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    created_at = models.DateTimeField(auto_now_add=True)
    last_used_at = models.DateTimeField(null=True, blank=True)
    revoked_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]
        verbose_name = "API key"

    def __str__(self):
        return f"{self.name} ({self.prefix}…)"

    @property
    def is_active(self):
        return self.revoked_at is None

    @staticmethod
    def hash_key(raw_key):
        return hashlib.sha256(raw_key.encode()).hexdigest()

    @classmethod
    def create_key(cls, organization, name, created_by=None, raw_key=None):
        """Return (api_key, raw_key). The raw key is never stored; show it once."""
        raw_key = raw_key or cls.KEY_PREFIX + secrets.token_urlsafe(32)
        api_key = cls.objects.create(
            organization=organization,
            name=name,
            prefix=raw_key[: cls.DISPLAY_CHARS],
            key_hash=cls.hash_key(raw_key),
            created_by=created_by,
        )
        return api_key, raw_key

    @classmethod
    def authenticate(cls, raw_key):
        if not raw_key:
            return None
        return (
            cls.objects.select_related("organization")
            .filter(key_hash=cls.hash_key(raw_key), revoked_at__isnull=True, organization__is_active=True)
            .first()
        )
