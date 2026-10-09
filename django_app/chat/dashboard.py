"""The organization Home page: what needs attention, and how the assistant is being used."""
from collections import Counter
from datetime import timedelta

from django.db.models import Avg, BooleanField, Count, ExpressionWrapper, Q, Sum
from django.shortcuts import render
from django.utils import timezone

from documents.models import Document
from rag.pipeline import NO_DOCUMENTS_ANSWER
from tenants.decorators import org_member_required
from tenants.menu import attention_items

from .models import ChatSession, Message

PERIODS = (7, 30, 90)
MAX_SOURCES = 4
MAX_DOCUMENTS = 6
WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")

# How an answer turned out. "Not covered" means the documents had nothing close
# enough to the question (the same test the Knowledge gaps page uses).
ANSWERED, NOT_COVERED, FAILED = "answered", "not_covered", "failed"
OUTCOMES = ((ANSWERED, "Answered"), (NOT_COVERED, "Not covered by documents"), (FAILED, "Failed"))


def _greeting():
    hour = timezone.localtime().hour
    return "Good morning" if hour < 12 else "Good afternoon" if hour < 17 else "Good evening"


def _avg_seconds(qs):
    value = qs.aggregate(v=Avg("latency_ms"))["v"]
    return value / 1000 if value else None


def _outcome(is_error, score, no_documents, threshold):
    if is_error:
        return FAILED
    if no_documents or (score is not None and score < threshold):
        return NOT_COVERED
    return ANSWERED


def _answer_rows(answers, threshold):
    """(created_at, outcome) for each answer, without loading answer texts."""
    no_docs = ExpressionWrapper(Q(content=NO_DOCUMENTS_ANSWER), output_field=BooleanField())
    rows = answers.annotate(no_docs=no_docs).values_list("created_at", "is_error", "retrieval_score", "no_docs")
    return [(created, _outcome(err, score, nd, threshold)) for created, err, score, nd in rows]


def _delta(now, before, kind="pct", better="up"):
    """A change badge versus the previous period: kind is "pct" (relative), "pts" or "s" (absolute)."""
    if now is None or before is None or (kind == "pct" and not before):
        return None
    change = round(100 * (now - before) / before) if kind == "pct" else round(now - before, 1)
    if change == 0:
        return {"text": "No change", "tone": "flat", "dir": "flat"}
    suffix = {"pct": "%", "pts": " pts", "s": " s"}[kind]
    good = (change > 0) == (better == "up")
    return {"text": f"{'+' if change > 0 else '−'}{abs(change):g}{suffix}", "tone": "good" if good else "bad",
            "dir": "up" if change > 0 else "down"}


def _period_numbers(org, start, end):
    """The headline numbers for one period, so the current and previous periods compare like for like."""
    questions = Message.objects.filter(organization=org, role=Message.Role.USER, created_at__gte=start, created_at__lt=end)
    answers = Message.objects.filter(organization=org, role=Message.Role.ASSISTANT, created_at__gte=start, created_at__lt=end)
    rows = _answer_rows(answers, org.gap_threshold)
    answered = sum(1 for _, o in rows if o == ANSWERED)
    ratings = answers.exclude(rating=None).aggregate(n=Count("id"), up=Count("id", filter=Q(rating=1)))
    return {
        "questions": questions, "answers": answers, "rows": rows,
        "total": questions.count(),
        "answered": answered,
        "answered_pct": round(100 * answered / len(rows)) if rows else None,
        "rated": ratings["n"],
        "helpful_pct": round(100 * ratings["up"] / ratings["n"]) if ratings["n"] else None,
        # Successful answers only: failures return fast and would flatter the number.
        "avg": _avg_seconds(answers.filter(is_error=False)),
    }


def _activity(rows, days):
    """Bars for the activity chart: one per day, or per week for the 90-day view, oldest first."""
    today = timezone.localdate()
    step = 7 if days > 31 else 1
    count = -(-days // step)
    bars = []
    for i in range(count):
        last = today - timedelta(days=step * (count - 1 - i))
        first = last - timedelta(days=step - 1)
        label = f"{first:%a %d %b}" if step == 1 else f"{first:%d %b} – {last:%d %b}"
        bars.append({"first": first, "label": label, ANSWERED: 0, NOT_COVERED: 0, FAILED: 0})
    for created, outcome in rows:
        index = count - 1 - (today - timezone.localdate(created)).days // step
        if 0 <= index < count:
            bars[index][outcome] += 1
    for b in bars:
        b["total"] = b[ANSWERED] + b[NOT_COVERED] + b[FAILED]
    peak = max((b["total"] for b in bars), default=0)
    # Round the axis up to an even number so the middle gridline is a whole number.
    top = max(2, peak + peak % 2)
    for b in bars:
        b["h"] = round(100 * b["total"] / top, 2)
        # Segment heights are shares of their own bar, stacked answered → not covered → failed.
        b["segments"] = [{"key": key, "n": b[key], "h": round(100 * b[key] / b["total"], 2)}
                         for key, _ in OUTCOMES if b[key]]
    busiest = max(bars, key=lambda b: b["total"]) if peak else None
    ticks = [bars[0]["first"], bars[len(bars) // 2]["first"], bars[-1]["first"]] if bars else []
    return {"bars": bars, "peak": peak, "top": top, "mid": top // 2,
            "ticks": [f"{d:%d %b}" for d in ticks], "weekly": step == 7,
            "total": sum(b["total"] for b in bars), "active": sum(1 for b in bars if b["total"]),
            "busiest": busiest}


def _sparkline(bars):
    """SVG polyline points (viewBox 0 0 100 32) for the question volume trend."""
    values = [b["total"] for b in bars]
    if sum(1 for v in values if v) < 3:  # too little history for a trend line
        return ""
    top = max(values)
    return " ".join(f"{100 * i / (len(values) - 1):.1f},{30 - 28 * v / top:.1f}" for i, v in enumerate(values))


def _quality(rows):
    counts = Counter(o for _, o in rows)
    total = sum(counts.values())
    parts, offset = [], 0.0
    for key, label in OUTCOMES:
        pct = 100 * counts[key] / total if total else 0
        parts.append({"key": key, "label": label, "n": counts[key], "pct": round(pct), "dash": round(pct, 2), "rest": round(100 - pct, 2),
                      "offset": round(25 - offset, 2)})  # 25 starts the ring at 12 o'clock
        offset += pct
    return {"total": total, "parts": parts}


def _busiest(questions):
    """A weekday × hour grid of question counts, shaded in five steps."""
    grid = [[0] * 24 for _ in WEEKDAYS]
    for created in questions.values_list("created_at", flat=True):
        local = timezone.localtime(created)
        grid[local.weekday()][local.hour] += 1
    peak = max(max(row) for row in grid)
    rows = [{"day": day, "cells": [{"n": n, "level": 0 if not n else 1 + min(3, int(4 * n / (peak + 1))),
                                    "tip": f"{day} {h:02d}:00–{h + 1:02d}:00 · {n} question{'s' if n != 1 else ''}"}
                                   for h, n in enumerate(row)]} for day, row in zip(WEEKDAYS, grid)]
    best = None
    if peak:
        d, h = max(((d, h) for d in range(7) for h in range(24)), key=lambda dh: grid[dh[0]][dh[1]])
        best = f"{WEEKDAYS[d]} {h:02d}:00–{h + 1:02d}:00"
    return {"rows": rows, "peak": peak, "best": best}


def _document_usage(org, answers):
    """How many answers cited each document in the period, so unused documents stand out."""
    cited = Counter()
    for sources in answers.filter(is_error=False).values_list("sources", flat=True):
        for doc_id in {s.get("document_id") for s in sources or [] if isinstance(s, dict)}:
            cited[doc_id] += 1
    documents = Document.objects.filter(organization=org, status=Document.Status.READY)
    usage = sorted(({"name": d.original_name, "n": cited[d.pk], "paused": d.is_paused} for d in documents),
                   key=lambda u: (-u["n"], u["name"]))
    peak = usage[0]["n"] if usage else 0
    for u in usage:
        u["width"] = round(100 * u["n"] / peak) if peak else 0
    return {"items": usage[:MAX_DOCUMENTS], "more": max(0, len(usage) - MAX_DOCUMENTS),
            "unused": sum(1 for u in usage if not u["n"] and not u["paused"])}


@org_member_required
def dashboard(request):
    org = request.organization
    is_admin = request.membership.is_admin
    try:
        days = int(request.GET.get("period", 30))
    except ValueError:
        days = 30
    days = days if days in PERIODS else 30
    now = timezone.now()
    since = now - timedelta(days=days)

    current = _period_numbers(org, since, now)
    previous = _period_numbers(org, since - timedelta(days=days), since)
    questions, answers, total = current["questions"], current["answers"], current["total"]
    api = questions.filter(session__channel=ChatSession.Channel.API).count()
    people = questions.exclude(session__user=None).values("session__user").distinct().count()
    apps = questions.exclude(session__api_key=None).values("session__api_key").distinct().count()

    ready = Document.objects.filter(organization=org).answerable()
    kb = ready.aggregate(pages=Sum("page_count"), chunks=Sum("chunk_count"))
    paused = Document.objects.filter(organization=org, is_paused=True).count()

    # Where questions come from: each API key, plus the web app.
    by_source = list(
        questions.values("session__channel", "session__api_key__name").annotate(n=Count("id")).order_by("-n")
    )
    sources = [{"name": (r["session__api_key__name"] or "deleted key") if r["session__channel"] == ChatSession.Channel.API
                else "Web app", "n": r["n"]} for r in by_source]
    if len(sources) > MAX_SOURCES:
        sources = sources[:MAX_SOURCES - 1] + [{"name": "Other", "n": sum(s["n"] for s in sources[MAX_SOURCES - 1:])}]
    for i, s in enumerate(sources):
        s["shade"] = i + 1
        s["pct"] = round(100 * s["n"] / total, 1) if total else 0

    topics = list(questions.exclude(topic=None).values("topic__name").annotate(n=Count("id")).order_by("-n")[:5])
    peak = topics[0]["n"] if topics else 0
    for t in topics:
        t["width"] = round(100 * t["n"] / peak) if peak else 0
        t["pct"] = round(100 * t["n"] / total) if total else 0

    activity = _activity(current["rows"], days)
    name = request.user.get_full_name() or request.user.username
    return render(request, "chat/dashboard.html", {
        "days": days, "periods": PERIODS, "greeting": f"{_greeting()}, {name}", "greeting_name": name,
        "now": timezone.localtime(),
        "total": total, "api": api, "web": total - api, "people": people, "apps": apps,
        "questions_delta": _delta(total, previous["total"]),
        "sparkline": _sparkline(activity["bars"]),
        "success_pct": current["answered_pct"], "answered_ok": current["answered"], "answer_count": len(current["rows"]),
        "success_delta": _delta(current["answered_pct"], previous["answered_pct"], "pts"),
        "helpful_pct": current["helpful_pct"], "rated": current["rated"],
        "helpful_delta": _delta(current["helpful_pct"], previous["helpful_pct"], "pts"),
        # Answer times are for admins only.
        "avg_all": current["avg"] if is_admin else None,
        "avg_delta": _delta(current["avg"], previous["avg"], "s", better="down") if is_admin else None,
        "avg_web": _avg_seconds(answers.filter(is_error=False, session__channel=ChatSession.Channel.WEB)) if is_admin else None,
        "avg_api": _avg_seconds(answers.filter(is_error=False, session__channel=ChatSession.Channel.API)) if is_admin else None,
        "docs": ready.count(), "pages": kb["pages"] or 0, "chunks": kb["chunks"] or 0, "paused_docs": paused,
        "attention": attention_items(org, request.membership, days=days),
        "activity": activity, "quality": _quality(current["rows"]),
        "busiest": _busiest(questions) if is_admin else None,
        "doc_usage": _document_usage(org, answers),
        "sources": sources, "topics": topics,
        "recent_chats": ChatSession.objects.filter(
            organization=org, user=request.user, channel=ChatSession.Channel.WEB
        ).annotate(n=Count("messages"))[:5],
    })
