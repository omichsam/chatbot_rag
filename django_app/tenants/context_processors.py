from django.conf import settings

from portal.models import platform


def organization(request):
    user = getattr(request, "user", None)
    org = getattr(request, "organization", None)
    membership = getattr(request, "membership", None)
    memberships = []
    badges = {}
    if user is not None and user.is_authenticated:
        memberships = list(
            user.memberships.select_related("organization").filter(organization__is_active=True)
        )
        if org is not None and membership is not None:
            badges = _menu_badges(org, membership)
    return {
        "organization": org,
        "membership": membership,
        "user_memberships": memberships,
        "allow_signup": platform().allow_signup,
        "announcement": _announcement(user),
        "platform_maintenance": platform().maintenance_mode,
        "menu_badges": badges,
        "portal_role": _portal_role(user),
        "platform_badges": _platform_badges() if _portal_role(user) else {},
        "console": _console() if request.path.startswith("/platform/") and _portal_role(user) else None,
        "support_mode": getattr(request, "support_mode", False),
        **_shell(request, org, membership),
    }


def _announcement(user):
    conf = platform()
    if not (user is not None and user.is_authenticated):
        return None
    if conf.maintenance_mode and user.is_superuser:
        return {"text": "Maintenance mode is on: only superadmins can use DocAssist.", "tone": "warning"}
    if conf.announcement:
        return {"text": conf.announcement, "tone": conf.announcement_tone}
    return None


def _shell(request, org, membership):
    """Sidebar menu, AI meter and notifications for pages inside an organization."""
    if org is None or membership is None:
        return {"menu": None, "ai_status": None, "notifications": []}
    from .menu import ai_status, attention_items, build_menu

    return {
        "menu": build_menu(request),
        # Operational details (AI usage, quota) are for admins only.
        "ai_status": ai_status(org) if membership.is_admin else None,
        "notifications": attention_items(org, membership),
        "member_count": org.memberships.count(),
        "access_pending": not membership.is_admin and membership.user.access_requests.filter(
            organization=org, status="pending").exists(),
    }


def _menu_badges(org, membership):
    """Small counts shown next to sidebar items."""
    from chat.insights import gap_answers, pending_feedback
    from documents.models import Document

    badges = {
        "documents_busy": Document.objects.filter(
            organization=org, status__in=[Document.Status.PENDING, Document.Status.PROCESSING]
        ).count(),
        "saved": membership.user.saved_answers.filter(message__organization=org).count(),
    }
    if membership.is_admin:
        badges["feedback"] = pending_feedback(org).count()
        badges["gaps"] = gap_answers(org).count()
    return badges


def _portal_role(user):
    from portal.access import portal_role

    return portal_role(user)


def _console():
    """Status shown in the Platform Console header."""
    from portal.models import AIProvider

    enabled = list(AIProvider.objects.filter(is_enabled=True).only("name", "last_test_ok"))
    failing = [p.name for p in enabled if p.last_test_ok is False]
    return {
        "environment": settings.PLATFORM_ENVIRONMENT,
        "version": settings.APP_VERSION,
        "maintenance": platform().maintenance_mode,
        "providers": len(enabled), "providers_failing": failing,
    }


def _platform_badges():
    """Counts for the superadmin's Platform menu section."""
    from documents.models import Document
    from tenants.models import Organization

    return {
        "deleting": Organization.objects.filter(deletion_scheduled_at__isnull=False).count(),
        "suspended": Organization.objects.filter(is_active=False).count(),
        "problems": Document.objects.filter(status=Document.Status.FAILED).count(),
    }
