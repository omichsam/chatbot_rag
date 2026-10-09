"""Organization lifecycle: scheduled deletion and data export."""
import io
import json
import zipfile
from datetime import timedelta

from django.utils import timezone

from api.models import APIKey
from chat.models import ChatSession, Message, Topic
from tenants.models import Organization

from .models import AuditLog

GRACE_DAYS = 30


def schedule_deletion(org, reason=""):
    org.deletion_scheduled_at = timezone.now() + timedelta(days=GRACE_DAYS)
    org.is_active = False
    org.suspended_reason = reason or "Scheduled for deletion"
    org.suspended_at = org.suspended_at or timezone.now()
    org.save(update_fields=["deletion_scheduled_at", "is_active", "suspended_reason", "suspended_at"])


def cancel_deletion(org):
    org.deletion_scheduled_at = None
    org.is_active = True
    org.suspended_reason, org.suspended_at = "", None
    org.save(update_fields=["deletion_scheduled_at", "is_active", "suspended_reason", "suspended_at"])


def purge_due(now=None):
    """Delete organizations whose grace period has ended. Returns their names."""
    now = now or timezone.now()
    names = []
    for org in Organization.objects.filter(deletion_scheduled_at__lte=now):
        names.append(org.name)
        AuditLog.objects.create(action="org.purge", organization_name=org.name, target=org.name,
                                detail=f"Grace period ended {org.deletion_scheduled_at:%d %b %Y}")
        org.delete()  # signals remove files and passages
    return names


def _iso(value):
    return value.isoformat() if value else None


def export_organization(org):
    """A zip of everything the organization owns: its data as JSON, plus the original files."""
    data = {
        "exported_at": timezone.now().isoformat(),
        "organization": {
            "name": org.name, "slug": org.slug, "created_at": _iso(org.created_at), "is_active": org.is_active,
            "plan": org.plan.name if org.plan else None, "monthly_query_limit": org.monthly_query_limit,
            "assistant_name": org.assistant_name, "system_prompt": org.system_prompt,
        },
        "members": [{"username": m.user.username, "email": m.user.email, "role": m.role, "joined": _iso(m.created_at)}
                    for m in org.memberships.select_related("user")],
        "documents": [{"id": d.pk, "file": d.original_name, "status": d.status, "pages": d.page_count,
                       "passages": d.chunk_count, "size": d.size, "paused": d.is_paused, "uploaded": _iso(d.created_at)}
                      for d in org.documents.all()],
        "api_keys": [{"name": k.name, "prefix": k.prefix, "created": _iso(k.created_at), "revoked": _iso(k.revoked_at),
                      "last_used": _iso(k.last_used_at)} for k in APIKey.objects.filter(organization=org)],
        "topics": [{"name": t.name, "description": t.description} for t in Topic.objects.filter(organization=org)],
        "conversations": [
            {"id": str(s.public_id), "channel": s.channel, "user": s.user.username if s.user else None,
             "title": s.title, "created": _iso(s.created_at),
             "messages": [{"role": m.role, "content": m.content, "sources": m.sources, "created": _iso(m.created_at),
                           "rating": m.rating, "answered_by": m.answered_by}
                          for m in Message.objects.filter(session=s).order_by("created_at", "id")]}
            for s in ChatSession.objects.filter(organization=org).select_related("user").order_by("created_at")
        ],
    }
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("organization.json", json.dumps(data, indent=2, ensure_ascii=False))
        for document in org.documents.all():
            try:
                with document.file.open("rb") as f:
                    archive.writestr(f"documents/{document.pk}_{document.original_name}", f.read())
            except (FileNotFoundError, ValueError):
                pass  # the file is missing on disk; the JSON still lists the document
    return buffer.getvalue()
