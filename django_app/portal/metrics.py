"""Platform-wide numbers for the superadmin portal.

Each metric is one grouped query across all organizations (not one query per
organization), so the pages stay fast as tenants are added.
"""
import os
from collections import defaultdict
from datetime import timedelta
from pathlib import Path

from django.conf import settings
from django.contrib.auth import get_user_model
from django.db.models import Count, Max, Q, Sum
from django.db.models.functions import TruncDate
from django.utils import timezone

from api.models import APIKey
from chat.models import ChatSession, Message
from documents.models import Document
from tenants.models import Membership, Organization

User = get_user_model()


def _by_org(queryset, **aggregates):
    return {row.pop("organization"): row for row in queryset.values("organization").annotate(**aggregates)}


def organization_stats(days=30):
    """{org_id: {...}} with members, documents, questions, failures, keys and last activity."""
    since = timezone.now() - timedelta(days=days)
    members = _by_org(Membership.objects.all(), members=Count("id"))
    docs = _by_org(
        Document.objects.all(),
        docs_ready=Count("id", filter=Q(status=Document.Status.READY)),
        docs_failed=Count("id", filter=Q(status=Document.Status.FAILED)),
        docs_busy=Count("id", filter=Q(status__in=[Document.Status.PENDING, Document.Status.PROCESSING])),
        storage=Sum("size"),
        pages=Sum("page_count"),
    )
    questions = _by_org(
        Message.objects.filter(role=Message.Role.USER, created_at__gte=since),
        questions=Count("id"),
        api_questions=Count("id", filter=Q(session__channel=ChatSession.Channel.API)),
    )
    answers = _by_org(
        Message.objects.filter(role=Message.Role.ASSISTANT, created_at__gte=since),
        answers=Count("id"), failed=Count("id", filter=Q(is_error=True)),
        disliked=Count("id", filter=Q(rating=-1)),
    )
    keys = _by_org(APIKey.objects.filter(revoked_at__isnull=True), keys=Count("id"))
    last = _by_org(Message.objects.all(), last_activity=Max("created_at"))
    owners = {}
    for m in Membership.objects.filter(role=Membership.Role.OWNER).select_related("user").order_by("created_at"):
        owners.setdefault(m.organization_id, m.user.username)

    stats = {}
    for org_id in Organization.objects.values_list("id", flat=True):
        row = {"members": 0, "docs_ready": 0, "docs_failed": 0, "docs_busy": 0, "storage": 0, "pages": 0,
               "questions": 0, "api_questions": 0, "answers": 0, "failed": 0, "disliked": 0, "keys": 0,
               "last_activity": None}
        for source in (members, docs, questions, answers, keys, last):
            row.update({k: v for k, v in source.get(org_id, {}).items() if v is not None})
        row["success_pct"] = round(100 * (row["answers"] - row["failed"]) / row["answers"]) if row["answers"] else None
        row["owner"] = owners.get(org_id)
        stats[org_id] = row
    return stats


def platform_totals(days=30):
    since = timezone.now() - timedelta(days=days)
    answers = Message.objects.filter(role=Message.Role.ASSISTANT, created_at__gte=since)
    answered = answers.count()
    failed = answers.filter(is_error=True).count()
    docs = Document.objects.aggregate(
        ready=Count("id", filter=Q(status=Document.Status.READY)),
        failed=Count("id", filter=Q(status=Document.Status.FAILED)),
        busy=Count("id", filter=Q(status__in=[Document.Status.PENDING, Document.Status.PROCESSING])),
        bytes=Sum("size"), chunks=Sum("chunk_count"),
    )
    return {
        "orgs": Organization.objects.count(),
        "orgs_active": Organization.objects.filter(is_active=True).count(),
        "users": User.objects.count(),
        "users_active_30d": User.objects.filter(last_login__gte=since).count(),
        "superusers": User.objects.filter(is_superuser=True).count(),
        "questions": Message.objects.filter(role=Message.Role.USER, created_at__gte=since).count(),
        "answered": answered,
        "failed": failed,
        "success_pct": round(100 * (answered - failed) / answered) if answered else None,
        "docs_ready": docs["ready"] or 0, "docs_failed": docs["failed"] or 0, "docs_busy": docs["busy"] or 0,
        "doc_bytes": docs["bytes"] or 0, "chunks": docs["chunks"] or 0,
        "keys_active": APIKey.objects.filter(revoked_at__isnull=True).count(),
        "last_ai_error": answers.filter(is_error=True).aggregate(v=Max("created_at"))["v"],
    }


def daily_by_source(days=30):
    """Questions per day across all organizations, split into web / api / test."""
    today = timezone.localdate()
    start = today - timedelta(days=days - 1)
    rows = (
        Message.objects.filter(role=Message.Role.USER, created_at__date__gte=start)
        .annotate(day=TruncDate("created_at"))
        .values("day", "session__channel", "session__is_test")
        .annotate(n=Count("id"))
    )
    per_day = defaultdict(lambda: {"web": 0, "api": 0, "test": 0})
    for r in rows:
        source = "test" if r["session__is_test"] else r["session__channel"]
        per_day[r["day"]][source] += r["n"]
    series = []
    for offset in range(days):
        day = start + timedelta(days=offset)
        counts = per_day.get(day, {"web": 0, "api": 0, "test": 0})
        series.append({"day": day, **counts, "total": sum(counts.values())})
    peak = max((d["total"] for d in series), default=0)
    for d in series:
        for key in ("web", "api", "test"):
            d[f"{key}_pct"] = round(100 * d[key] / peak, 1) if peak else 0
    return series, peak


def folder_size(path):
    total = 0
    path = Path(path) if path else None
    if not path or not path.exists():
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            try:
                total += os.path.getsize(os.path.join(root, name))
            except OSError:
                pass
    return total


def needs_attention(stats, orgs):
    """Organizations with something an operator should look at, most urgent first."""
    items = []
    for org in orgs:
        s = stats.get(org.pk, {})
        reasons = []
        if s.get("failed"):
            reasons.append(f"{s['failed']} failed answer{'s' if s['failed'] != 1 else ''}")
        if s.get("docs_failed"):
            reasons.append(f"{s['docs_failed']} document{'s' if s['docs_failed'] != 1 else ''} failed to index")
        if org.monthly_query_limit and org.queries_this_month() >= org.monthly_query_limit * 0.9:
            reasons.append("near its monthly limit")
        if not org.is_active:
            reasons.append("suspended")
        if reasons:
            items.append({"org": org, "reasons": reasons, "weight": s.get("failed", 0) + s.get("docs_failed", 0)})
    return sorted(items, key=lambda i: -i["weight"])


# --- Operations dashboard ----------------------------------------------------------------

def operations(days=30):
    """Headline numbers for the console dashboard, with the previous period for comparison."""
    from django.db.models import Avg

    now = timezone.now()
    since, before = now - timedelta(days=days), now - timedelta(days=2 * days)

    def period(start, end):
        questions = Message.objects.filter(role=Message.Role.USER, created_at__gte=start, created_at__lt=end)
        answers = Message.objects.filter(role=Message.Role.ASSISTANT, created_at__gte=start, created_at__lt=end)
        n_answers = answers.count()
        failed = answers.filter(is_error=True).count()
        backup = answers.filter(is_error=False).exclude(error_detail="").count()
        avg_ms = answers.filter(is_error=False).aggregate(v=Avg("latency_ms"))["v"]
        return {
            "questions": questions.count(),
            "active_orgs": questions.values("organization").distinct().count(),
            "active_people": questions.exclude(session__user=None).values("session__user").distinct().count(),
            "answers": n_answers,
            "failure_pct": round(100 * failed / n_answers, 1) if n_answers else None,
            "backup_pct": round(100 * backup / n_answers, 1) if n_answers else None,
            "avg_seconds": avg_ms / 1000 if avg_ms else None,
        }

    current, previous = period(since, now), period(before, since)
    for key in ("questions", "active_orgs", "active_people"):
        old = previous[key]
        current[f"{key}_delta"] = round(100 * (current[key] - old) / old) if old else None
    current["orgs_total"] = Organization.objects.count()
    current["orgs_suspended"] = Organization.objects.filter(is_active=False, deletion_scheduled_at__isnull=True).count()
    current["orgs_deleting"] = Organization.objects.filter(deletion_scheduled_at__isnull=False).count()
    current["storage"] = Document.objects.aggregate(v=Sum("size"))["v"] or 0
    current["passages"] = Document.objects.aggregate(v=Sum("chunk_count"))["v"] or 0
    return current


def provider_insights(days=30):
    """Per provider: answers written, share, speed, failures and the last error. Plus quoted and cached answers."""
    from django.db.models import Avg

    from .models import AIProvider

    since = timezone.now() - timedelta(days=days)
    answers = Message.objects.filter(role=Message.Role.ASSISTANT, created_at__gte=since)
    total = answers.exclude(answered_by="").count()
    written = {r["answered_by"]: r for r in answers.exclude(answered_by="").values("answered_by")
               .annotate(n=Count("id"), avg_ms=Avg("latency_ms"))}
    # Failures are recorded on the answer as "Provider name: error; Next name: error".
    failures, last_error = defaultdict(int), {}
    for detail, created in answers.exclude(error_detail="").order_by("created_at").values_list("error_detail", "created_at"):
        for part in detail.split("; "):
            name, _, error = part.partition(": ")
            if name and error:
                failures[name] += 1
                last_error[name] = (error[:160], created)

    rows, kinds_seen = [], set()
    for provider in AIProvider.objects.all():
        # Answers are recorded by provider kind; credit them to the first provider of that kind.
        stats = written.get(provider.kind, {}) if provider.kind not in kinds_seen else {}
        kinds_seen.add(provider.kind)
        n = stats.get("n", 0)
        avg_ms = stats.get("avg_ms")
        rows.append({
            "provider": provider, "answers": n, "share": round(100 * n / total) if total else 0,
            "avg_seconds": avg_ms / 1000 if avg_ms else None,
            "failures": failures.get(provider.name, 0),
            "last_error": last_error.get(provider.name),
        })
    quoted, cache = written.get("quoted", {}).get("n", 0), written.get("cache", {}).get("n", 0)
    return {
        "rows": rows, "total": total,
        "quoted": quoted, "quoted_share": round(100 * quoted / total) if total else 0,
        "cache": cache, "cache_share": round(100 * cache / total) if total else 0,
    }


def api_key_inventory(stale_days=90, old_days=180):
    """Every active API key across organizations, with the reasons it may need attention."""
    now = timezone.now()
    rows = []
    for key in APIKey.objects.filter(revoked_at__isnull=True).select_related("organization", "created_by"):
        flags = []
        if key.last_used_at is None and key.created_at < now - timedelta(days=30):
            flags.append("Never used")
        elif key.last_used_at and key.last_used_at < now - timedelta(days=stale_days):
            flags.append(f"Unused for {stale_days}+ days")
        if key.created_at < now - timedelta(days=old_days):
            flags.append(f"Older than {old_days} days: rotate")
        rows.append({"key": key, "flags": flags})
    rows.sort(key=lambda r: (-len(r["flags"]), r["key"].organization.name))
    return rows
