from django.apps import AppConfig


class TenantsConfig(AppConfig):
    name = "tenants"
    verbose_name = "Organizations"

    def ready(self):
        from . import signals  # noqa: F401
