"""Who may use the Platform Console, and for what.

Superusers are **owners** with full control. Platform staff (portal.PlatformStaff)
get a narrower role: **support** can look at organizations and users, read the
audit log and system health, and enter read-only support mode; **finance** can
see usage and plans. Changes (any POST) are owner-only unless a view says
otherwise. When two-factor sign-in is required, or the person has set it up,
the console also needs this session to have passed the second step.
"""
from functools import wraps

from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import redirect

OWNER, SUPPORT, FINANCE = "owner", "support", "finance"
ROLE_LABELS = {OWNER: "Owner", SUPPORT: "Support", FINANCE: "Finance"}
ALL = (OWNER, SUPPORT, FINANCE)
SESSION_2FA = "two_factor_verified"


def portal_role(user):
    """The user's console role, or None."""
    if user is None or not user.is_authenticated or not user.is_active:
        return None
    if user.is_superuser:
        return OWNER
    staff = getattr(user, "platform_staff", None)
    return staff.role if staff is not None else None


def two_factor_needed(request):
    """'setup' (must enrol first), 'verify' (must enter a code), or None."""
    from .models import TwoFactorDevice, platform

    device = TwoFactorDevice.objects.filter(user=request.user, confirmed_at__isnull=False).first()
    if device is None:
        return "setup" if platform().require_2fa else None
    return None if request.session.get(SESSION_2FA) else "verify"


def portal_view(*roles, write=(OWNER,), two_factor=True):
    """Allow GET for `roles` and POST for `write`; enforce two-factor sign-in."""
    roles = roles or ALL

    def decorator(view):
        @wraps(view)
        def wrapped(request, *args, **kwargs):
            role = portal_role(request.user)
            allowed = write if request.method == "POST" else roles
            if role not in allowed:
                raise PermissionDenied
            if two_factor:
                needed = two_factor_needed(request)
                if needed == "setup":
                    return redirect("portal:two_factor")
                if needed == "verify":
                    return redirect(f"/accounts/2fa/?next={request.get_full_path()}")
            request.portal_role = role
            return view(request, *args, **kwargs)

        return login_required(wrapped)

    return decorator


def require_role(request, *roles):
    if portal_role(request.user) not in roles:
        raise PermissionDenied
