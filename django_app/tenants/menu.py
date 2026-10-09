"""The sidebar menu, "needs attention" items and AI status, built per request.

One registry describes every menu item. Each person's own arrangement
(Membership.menu_prefs: pinned / hidden / order) is applied on top, and
role decides what is shown: members see admin areas folded into a single
"Admin tools" card instead of locked items.
"""
from collections import Counter
from datetime import timedelta

from django.urls import reverse
from django.utils import timezone

SECTIONS = [
    ("workspace", "Workspace", False),
    ("insights", "Insights", True),
    ("build", "Build", False),
    ("admin", "Admin", True),
]

# id, label, icon, url name, section, admin only, (namespace, url names) that mark it active
ITEMS = [
    ("home", "Home", "dashboard", "dashboard", "workspace", False, ("", ("dashboard",))),
    ("chat", "Chat", "chat", "chat:index", "workspace", False, ("chat", ("index", "session"))),
    ("documents", "Documents", "file", "documents:list", "workspace", False, ("documents", None)),
    ("saved", "Saved answers", "bookmark", "chat:saved", "workspace", False, ("chat", ("saved",))),
    ("history", "My history", "history", "chat:history", "workspace", False, ("chat", ("history",))),
    ("analytics", "Analytics", "chart", "chat:analytics", "insights", True, ("chat", ("analytics",))),
    ("questions", "Questions", "search", "chat:questions", "insights", True, ("chat", ("questions",))),
    ("feedback", "Feedback", "inbox", "chat:feedback", "insights", True, ("chat", ("feedback",))),
    ("gaps", "Knowledge gaps", "target", "chat:gaps", "insights", True, ("chat", ("gaps",))),
    ("developer", "Developer", "code", "developer", "build", False, ("", ("developer",))),
    ("keys", "API keys", "key", "keys:list", "build", True, ("keys", None)),
    ("members", "Members & roles", "users", "tenants:members", "admin", True, ("tenants", ("members",))),
    ("settings", "Settings", "settings", "tenants:settings", "admin", True, ("tenants", ("settings",))),
]
ITEM_IDS = [i[0] for i in ITEMS]
ALWAYS_VISIBLE = {"home"}


def _badges(org, membership):
    from api.models import APIKey
    from chat.insights import gap_answers, pending_feedback
    from documents.models import Document

    from .models import AccessRequest

    busy = Document.objects.filter(organization=org, status__in=[Document.Status.PENDING, Document.Status.PROCESSING]).count()
    badges = {
        "documents": {"text": f"{busy} indexing", "kind": "progress"} if busy else None,
        "saved": _count(membership.user.saved_answers.filter(message__organization=org).count()),
    }
    if membership.is_admin:
        feedback = pending_feedback(org).count()
        requests = AccessRequest.objects.filter(organization=org, status=AccessRequest.Status.PENDING).count()
        badges.update({
            "feedback": {"text": str(feedback), "kind": "alert"} if feedback else None,
            "gaps": _count(gap_answers(org).count()),
            "keys": _count(APIKey.objects.filter(organization=org, revoked_at__isnull=True).count()),
            "members": {"text": str(requests), "kind": "alert"} if requests else None,
        })
    return badges


def _count(n):
    return {"text": str(n), "kind": "info"} if n else None


def build_menu(request):
    """Sections and pinned items for the current person, or None outside an organization."""
    org, membership = getattr(request, "organization", None), getattr(request, "membership", None)
    if org is None or membership is None:
        return None
    match = getattr(request, "resolver_match", None)
    ns, current = (match.namespace, match.url_name) if match else ("", "")
    prefs = membership.menu_prefs or {}
    order = {item_id: i for i, item_id in enumerate(prefs.get("order", []))}
    hidden = set(prefs.get("hidden", [])) - ALWAYS_VISIBLE
    pinned_ids = [p for p in prefs.get("pinned", []) if p in ITEM_IDS]
    badges = _badges(org, membership)
    is_admin = membership.is_admin

    items = {}
    for item_id, label, icon, url_name, section, admin_only, (active_ns, active_names) in ITEMS:
        items[item_id] = {
            "id": item_id, "label": label, "icon": icon, "url": reverse(url_name), "section": section,
            "admin_only": admin_only, "badge": badges.get(item_id),
            "active": ns == active_ns and (active_names is None or current in active_names),
            "allowed": is_admin or not admin_only,
        }

    def visible(item):
        return item["allowed"] and (item["id"] not in hidden or item["active"])

    pinned = [items[p] for p in pinned_ids if visible(items[p])]
    pinned_set = {i["id"] for i in pinned}
    sections = []
    for key, title, admin_section in SECTIONS:
        section_items = sorted(
            (i for i in items.values() if i["section"] == key and visible(i) and i["id"] not in pinned_set),
            key=lambda i: order.get(i["id"], 100 + ITEM_IDS.index(i["id"])),
        )
        if section_items:
            sections.append({"id": key, "title": title, "items": section_items})
    locked = [i for i in items.values() if not i["allowed"]]
    return {
        "pinned": pinned,
        "sections": sections,
        "locked_count": len(locked),
        "locked_labels": [i["label"] for i in locked],
        # Everything this person may arrange, in their current order, for the Customize dialog.
        "customizable": [
            {**i, "hidden": i["id"] in hidden, "pinned": i["id"] in pinned_set}
            for i in sorted((i for i in items.values() if i["allowed"]),
                            key=lambda i: (i["id"] not in pinned_set, order.get(i["id"], 100 + ITEM_IDS.index(i["id"]))))
        ],
    }


def clean_prefs(data, membership):
    """Validate menu preferences posted by the Customize dialog."""
    allowed = {i[0] for i in ITEMS if membership.is_admin or not i[5]}

    def ids(key):
        values = data.get(key, []) if isinstance(data, dict) else []
        return [v for v in dict.fromkeys(values) if isinstance(v, str) and v in allowed]

    return {"order": ids("order"), "hidden": [h for h in ids("hidden") if h not in ALWAYS_VISIBLE], "pinned": ids("pinned")[:5]}


# --- AI status (sidebar meter) ---------------------------------------------------

QUOTA_MARKERS = ("429", "quota", "rate limit", "resource exhausted")


def _failing_provider(error_detail):
    """The first provider named in a fallback reason ("Google Gemini: 429 ...")."""
    return error_detail.split(":", 1)[0].strip() or "The AI provider"


def ai_status(org):
    from chat.models import Message

    today_start = timezone.localtime().replace(hour=0, minute=0, second=0, microsecond=0)
    answers = Message.objects.filter(organization=org, role=Message.Role.ASSISTANT)
    today = Message.objects.filter(organization=org, role=Message.Role.USER, created_at__gte=today_start).count()
    last = answers.order_by("-created_at", "-id").first()
    status = {"today": today, "label": "Healthy", "tone": "good", "pct": None, "note": f"{today} question{'s' if today != 1 else ''} today"}
    if last and last.is_error:
        quota = any(m in last.error_detail.lower() for m in QUOTA_MARKERS)
        status.update({
            "label": "Quota reached" if quota else "Answers failing",
            "tone": "bad", "pct": 100,
            "note": "Gemini usually resets daily" if quota else "The AI service is returning errors",
        })
    elif last and last.error_detail:
        quota = any(m in last.error_detail.lower() for m in QUOTA_MARKERS)
        name = _failing_provider(last.error_detail)
        status.update({
            "label": "Using backup", "tone": "warn", "pct": 100,
            "note": f"{name} {'quota reached' if quota else 'unavailable'}; answers still work",
        })
    elif org.monthly_query_limit:
        used = org.queries_this_month()
        pct = min(100, round(100 * used / org.monthly_query_limit))
        status.update({"pct": pct, "tone": "bad" if pct >= 100 else ("warn" if pct >= 80 else "good"),
                       "label": f"{pct}% of monthly limit", "note": f"{used:,} of {org.monthly_query_limit:,} questions this month"})
    return status


# --- Needs attention (Home page + notifications bell) -------------------------------

def attention_items(org, membership, days=7):
    """Things an admin should act on, most urgent first. Members get an empty list."""
    if membership is None or not membership.is_admin:
        return []
    from chat.insights import _questions_for, gap_answers, pending_feedback
    from chat.models import Message
    from documents.models import Document

    from .models import AccessRequest

    since = timezone.now() - timedelta(days=days)
    items = []

    failed = Message.objects.filter(organization=org, role=Message.Role.ASSISTANT, is_error=True, created_at__gte=since)
    n_failed = failed.count()
    if n_failed:
        first = failed.order_by("created_at").select_related("session__api_key").first()
        latest = failed.order_by("-created_at").first()
        quota = any(m in latest.error_detail.lower() for m in QUOTA_MARKERS)
        keys = sorted({k for k in failed.values_list("session__api_key__name", flat=True) if k})
        where = f"Through API key {', '.join(keys)}" if keys else "In the web app"
        items.append({
            "tone": "bad", "icon": "alert",
            "title": f"{'Gemini quota reached: ' if quota else ''}{n_failed} answer{'s' if n_failed != 1 else ''} failed",
            "detail": f"{where} · since {timezone.localtime(first.created_at):%d %b %H:%M}",
            "url": reverse("chat:questions") + "?outcome=failed", "action": "Review",
        })

    backup = Message.objects.filter(organization=org, role=Message.Role.ASSISTANT, is_error=False,
                                    created_at__gte=since).exclude(error_detail="")
    n_backup = backup.count()
    if n_backup:
        latest = backup.order_by("-created_at").first()
        quota = any(m in latest.error_detail.lower() for m in QUOTA_MARKERS)
        name = _failing_provider(latest.error_detail)
        items.append({
            "tone": "warn", "icon": "zap",
            "title": f"{name} {'quota reached' if quota else 'unavailable'}: "
                     f"{n_backup} answer{'s' if n_backup != 1 else ''} came from the backup",
            "detail": f"Latest {timezone.localtime(latest.created_at):%d %b %H:%M} · people still got answers with sources",
            "url": reverse("chat:questions"), "action": "View",
        })

    requests = list(AccessRequest.objects.filter(organization=org, status=AccessRequest.Status.PENDING).select_related("user")[:3])
    for r in requests:
        items.append({
            "tone": "bad", "icon": "users", "title": f"{r.user.username} asked for admin access",
            "detail": r.message or f"Requested {timezone.localtime(r.created_at):%d %b %H:%M}",
            "url": reverse("tenants:members") + "#requests", "action": "Decide",
        })

    gap_list = list(gap_answers(org).order_by("-created_at")[:200])
    if gap_list:
        questions = _questions_for(gap_list)
        counts = Counter(q.content.strip() for q in questions.values() if q)
        top = counts.most_common(1)[0] if counts else None
        items.append({
            "tone": "info", "icon": "target",
            "title": f"{len(gap_list)} question{'s' if len(gap_list) != 1 else ''} your documents can't answer",
            "detail": f'Most asked: "{top[0][:70]}" ×{top[1]}' if top else "Weak or no matches in your documents",
            "url": reverse("chat:gaps"), "action": "Open gaps",
        })

    disliked = list(pending_feedback(org).order_by("-rated_at")[:1])
    n_disliked = pending_feedback(org).count()
    if n_disliked:
        latest = disliked[0]
        asker = latest.session.user.username if latest.session.user else "API"
        comment = f': "{latest.feedback_comment[:60]}"' if latest.feedback_comment else ""
        items.append({
            "tone": "info", "icon": "thumbs-down",
            "title": f"{n_disliked} answer{'s' if n_disliked != 1 else ''} rated not helpful",
            "detail": f"{asker}{comment} · {timezone.localtime(latest.rated_at):%d %b %H:%M}" if latest.rated_at else asker,
            "url": reverse("chat:feedback"), "action": "Review",
        })

    failed_docs = Document.objects.filter(organization=org, status=Document.Status.FAILED).count()
    if failed_docs:
        items.append({
            "tone": "bad", "icon": "file", "title": f"{failed_docs} document{'s' if failed_docs != 1 else ''} failed to index",
            "detail": "Often a scanned PDF with no text, or a damaged file", "url": reverse("documents:list"), "action": "Open",
        })
    return items
