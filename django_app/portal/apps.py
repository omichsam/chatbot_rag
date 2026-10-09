from django.apps import AppConfig


class PortalConfig(AppConfig):
    name = "portal"
    verbose_name = "Platform portal"

    def ready(self):
        from . import signals  # noqa: F401  (connects the failed sign-in recorder)
