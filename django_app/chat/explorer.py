"""Questions explorer: every question with its source, type, topic and outcome."""
import csv
import json
from collections import Counter, defaultdict
from datetime import timedelta
from urllib.parse import urlencode

from django.contrib import messages
from django.core.paginator import Paginator
from django.db.models import OuterRef, Subquery
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from api.models import APIKey
from rag.pipeline import NO_DOCUMENTS_ANSWER
from tenants.decorators import org_admin_required

from .categorize import Intent, recategorize
from .models import ChatSession, Message, Topic

PERIODS = {"7": 7, "30": 30, "90": 90, "all": None}
PAGE_SIZE = 10  # one screen; the pager below the list moves through the rest
ROW_LIMIT = 10000


class Outcome:
    ANSWERED, WEAK, FAILED, DISLIKED = "answered", "weak", "failed", "disliked"
    LABELS = {ANSWERED: "Answered", WEAK: "Weak match", FAILED: "Failed", DISLIKED: "Disliked"}
    ORDER = [ANSWERED, WEAK, DISLIKED, FAILED]


class Source:
    WEB, API, TEST = "web", "api", "test"
    LABELS = {WEB: "Web", API: "API", TEST: "Test"}


def _outcome(row, threshold):
    if row["reply_id"] is None or row["reply_error"]:
        return Outcome.FAILED
    if row["reply_rating"] == -1:
        return Outcome.DISLIKED
    weak = row["reply_content"] == NO_DOCUMENTS_ANSWER or (
        row["reply_score"] is not None and row["reply_score"] < threshold
    )
    # Greetings and questions about the assistant naturally match no document.
    if weak and row["intent"] in ("", Intent.DOCUMENT):
        return Outcome.WEAK
    return Outcome.ANSWERED


def _source(row):
    if row["session__channel"] == ChatSession.Channel.WEB:
        return Source.WEB, row["session__user__username"] or "deleted user"
    name = row["session__api_key__name"] or "deleted key"
    return (Source.TEST if row["session__is_test"] else Source.API), name


def question_rows(organization, days=30, filters=None):
    """Questions (newest first) joined with their answer, as plain dicts."""
    filters = filters or {}
    reply = Message.objects.filter(
        session=OuterRef("session"), role=Message.Role.ASSISTANT, id__gt=OuterRef("id")
    ).order_by("id")
    qs = Message.objects.filter(organization=organization, role=Message.Role.USER)
    if days:
        qs = qs.filter(created_at__gte=timezone.now() - timedelta(days=days))
    source = filters.get("source")
    if source == Source.WEB:
        qs = qs.filter(session__channel=ChatSession.Channel.WEB)
    elif source == Source.API:
        qs = qs.filter(session__channel=ChatSession.Channel.API, session__is_test=False)
    elif source == Source.TEST:
        qs = qs.filter(session__is_test=True)
    if filters.get("key"):
        qs = qs.filter(session__api_key_id=filters["key"])
    if filters.get("user"):
        qs = qs.filter(session__user__username=filters["user"], session__channel=ChatSession.Channel.WEB)
    if filters.get("topic") == "none":
        qs = qs.filter(topic__isnull=True, intent__in=["", Intent.DOCUMENT])
    elif filters.get("topic"):
        qs = qs.filter(topic_id=filters["topic"])
    if filters.get("intent"):
        qs = qs.filter(intent=filters["intent"])
    if filters.get("q"):
        qs = qs.filter(content__icontains=filters["q"])

    rows = list(
        qs.annotate(
            reply_id=Subquery(reply.values("id")[:1]),
            reply_content=Subquery(reply.values("content")[:1]),
            reply_error=Subquery(reply.values("is_error")[:1]),
            reply_score=Subquery(reply.values("retrieval_score")[:1]),
            reply_rating=Subquery(reply.values("rating")[:1]),
            reply_comment=Subquery(reply.values("feedback_comment")[:1]),
            reply_latency=Subquery(reply.values("latency_ms")[:1]),
            reply_sources=Subquery(reply.values("sources")[:1]),
        )
        .order_by("-created_at", "-id")
        .values(
            "id", "content", "created_at", "intent", "topic_id", "topic__name",
            "session__channel", "session__is_test", "session__api_key_id", "session__api_key__name",
            "session__user__username", "reply_id", "reply_content", "reply_error", "reply_score",
            "reply_rating", "reply_comment", "reply_latency", "reply_sources",
        )[:ROW_LIMIT]
    )
    for row in rows:
        row["outcome"] = _outcome(row, organization.gap_threshold)
        row["source"], row["source_name"] = _source(row)
        row["intent"] = row["intent"] or Intent.DOCUMENT
        row["intent_label"] = Intent.LABELS.get(row["intent"], "Document question")
        sources = row["reply_sources"]
        if isinstance(sources, str):  # SQLite returns JSON from a subquery as text
            try:
                sources = json.loads(sources)
            except ValueError:
                sources = []
        row["reply_sources"] = sources or []
    return rows


def _rate(part, whole):
    return round(100 * part / whole) if whole else 0


def _source_table(rows, keys):
    """One line per API key (live + test traffic) and per web user."""
    stats = defaultdict(lambda: {"questions": 0, "tests": 0, "answered": 0, "weak": 0, "failed": 0,
                                 "disliked": 0, "latency": [], "topics": Counter(), "last": None})
    for row in rows:
        ident = ("key", row["session__api_key_id"]) if row["session__channel"] == ChatSession.Channel.API else ("user", row["source_name"])
        s = stats[ident]
        if row["source"] == Source.TEST:
            s["tests"] += 1
        else:
            s["questions"] += 1
        s[row["outcome"]] += 1
        if row["reply_latency"]:
            s["latency"].append(row["reply_latency"])
        if row["topic__name"]:
            s["topics"][row["topic__name"]] += 1
        s["last"] = max(s["last"], row["created_at"]) if s["last"] else row["created_at"]

    table = []
    for key in keys:  # every key, even unused ones
        s = stats.pop(("key", key.pk), None) or stats.default_factory()
        table.append(_table_row("api", key.name, s, key=key))
    for (kind, ident), s in stats.items():
        if kind == "user":
            table.append(_table_row("web", ident, s))
        else:  # key since deleted
            table.append(_table_row("api", "deleted key", s))
    return sorted(table, key=lambda r: (-(r["total"]), r["name"].lower()))


def _table_row(kind, name, s, key=None):
    total = s["questions"] + s["tests"]
    return {
        "kind": kind, "name": name, "key": key, "total": total, "questions": s["questions"], "tests": s["tests"],
        "answered_pct": _rate(s["answered"], total), "failed": s["failed"], "weak": s["weak"],
        "avg_s": (sum(s["latency"]) / len(s["latency"]) / 1000) if s["latency"] else None,
        "top_topic": s["topics"].most_common(1)[0][0] if s["topics"] else None, "last": s["last"],
    }


def _bars(counter, total, labels=None):
    peak = max(counter.values(), default=0)
    return [
        {"key": k, "label": (labels or {}).get(k, k), "count": n, "pct": _rate(n, total), "width": _rate(n, peak)}
        for k, n in counter.most_common()
    ]


@org_admin_required
def questions(request):
    org = request.organization
    period = request.GET.get("period", "30")
    days = PERIODS.get(period, 30)
    filters = {k: request.GET.get(k, "").strip() for k in ("source", "key", "user", "topic", "intent", "q")}
    outcome = request.GET.get("outcome", "")
    if filters["key"] and not filters["key"].isdigit():
        filters["key"] = ""

    rows = question_rows(org, days, filters)
    outcome_counts = Counter(r["outcome"] for r in rows)
    shown = [r for r in rows if r["outcome"] == outcome] if outcome in Outcome.LABELS else rows

    if request.GET.get("format") == "csv":
        return _csv(shown, org)

    keys = list(APIKey.objects.filter(organization=org).order_by("name"))
    period_rows = question_rows(org, days)  # unfiltered, for the per-source table
    topics = list(Topic.objects.filter(organization=org))
    topic_counts = Counter(r["topic__name"] or ("Other" if r["intent"] == Intent.DOCUMENT else None) for r in shown)
    topic_counts.pop(None, None)

    topic_question_counts = Counter(r["topic_id"] for r in period_rows)
    page = Paginator(shown, PAGE_SIZE).get_page(request.GET.get("page"))
    page_numbers = page.paginator.get_elided_page_range(page.number, on_each_side=1, on_ends=1)
    keep = {k: v for k, v in request.GET.items() if v and k not in ("page", "outcome", "format")}
    keep["period"] = period
    return render(request, "chat/questions.html", {
        "rows": page, "page": page, "page_numbers": page_numbers, "ellipsis": Paginator.ELLIPSIS, "total": len(shown), "all_total": len(rows),
        "period": period, "filters": filters, "outcome": outcome,
        "outcomes": [{"key": k, "label": Outcome.LABELS[k], "count": outcome_counts.get(k, 0)} for k in Outcome.ORDER],
        "filter_qs": urlencode(keep), "page_qs": urlencode({**keep, **({"outcome": outcome} if outcome else {})}),
        "by_topic": _bars(topic_counts, len(shown)),
        "by_intent": _bars(Counter(r["intent"] for r in shown), len(shown), Intent.LABELS),
        "by_source": _bars(Counter(r["source"] for r in shown), len(shown), Source.LABELS),
        "source_table": _source_table(period_rows, keys),
        "keys": keys, "topics": topics, "intents": Intent.LABELS.items(),
        "topic_rows": [(t, topic_question_counts.get(t.pk, 0)) for t in topics],
        "clear_user_qs": urlencode({k: v for k, v in keep.items() if k != "user"}),
        "answered_pct": _rate(outcome_counts.get(Outcome.ANSWERED, 0), len(rows)),
    })


def _csv(rows, org):
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="{org.slug}-questions.csv"'
    writer = csv.writer(response)
    writer.writerow(["asked_at", "source", "who", "question", "type", "topic", "outcome", "answer_seconds",
                     "match_score", "rating", "comment", "sources", "answer"])
    for r in rows:
        writer.writerow([
            r["created_at"].isoformat(), Source.LABELS[r["source"]], r["source_name"], r["content"], r["intent_label"],
            r["topic__name"] or "", Outcome.LABELS[r["outcome"]],
            f'{r["reply_latency"] / 1000:.1f}' if r["reply_latency"] else "",
            f'{r["reply_score"]:.2f}' if r["reply_score"] is not None else "", r["reply_rating"] or "", r["reply_comment"] or "",
            "; ".join(f'{s.get("filename")} p.{s.get("page")}' for s in r["reply_sources"]), r["reply_content"] or "",
        ])
    return response


@org_admin_required
@require_POST
def topic_add(request):
    name = request.POST.get("name", "").strip()[:60]
    description = request.POST.get("description", "").strip()[:300]
    if not name:
        messages.error(request, "Give the topic a name.")
    elif Topic.objects.filter(organization=request.organization, name__iexact=name).exists():
        messages.error(request, f"There's already a topic called {name}.")
    else:
        Topic.objects.create(organization=request.organization, name=name, description=description)
        count = recategorize(request.organization)
        messages.success(request, f"Added {name} and re-sorted {count} questions.")
    return redirect(f"{_questions_url()}#topics")


@org_admin_required
@require_POST
def topic_delete(request, pk):
    topic = get_object_or_404(Topic, pk=pk, organization=request.organization)
    topic.delete()
    count = recategorize(request.organization)
    messages.success(request, f"Removed {topic.name} and re-sorted {count} questions.")
    return redirect(f"{_questions_url()}#topics")


@org_admin_required
@require_POST
def topic_resort(request):
    count = recategorize(request.organization)
    messages.success(request, f"Re-sorted {count} questions into topics.")
    return redirect(f"{_questions_url()}#topics")


def _questions_url():
    from django.urls import reverse

    return reverse("chat:questions")
