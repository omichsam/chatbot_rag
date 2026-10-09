"""Failed sign-ins go to the audit log, for the security centre."""
from django.contrib.auth.signals import user_login_failed
from django.dispatch import receiver


@receiver(user_login_failed)
def record_failed_sign_in(sender, credentials, request=None, **kwargs):
    from .models import AuditLog

    address = request.META.get("REMOTE_ADDR", "") if request is not None else ""
    AuditLog.objects.create(action="security.login_failed", target=str(credentials.get("username", ""))[:200],
                            detail=f"from {address or 'unknown address'}")
