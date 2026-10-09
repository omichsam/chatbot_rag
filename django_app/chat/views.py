import json
from datetime import timedelta

from django.conf import settings
from django.db.models import Avg, Count, Max, Sum
from django.db.models.functions import Lower, TruncDate
from django.http import JsonResponse, StreamingHttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from documents.models import Document
from tenants.decorators import org_admin_required, org_member_required

from . import services
from .explorer import question_rows
from .models import ChatSession, Message, SavedAnswer

ANALYTICS_DAYS = 30


def _user_sessions(request):
    return ChatSession.objects.filter(
        organization=request.organization, user=request.user, channel=ChatSession.Channel.WEB
    )


@org_member_required
def index(request, session_id=None):
    sessions = _user_sessions(request)
    current = get_object_or_404(sessions, public_id=session_id) if session_id else None
    ready = Document.objects.filter(organization=request.organization).answerable()
    return render(request, "chat/chat.html", {
        "sessions": sessions[:30],
        "current": current,
        "chat_messages": current.messages.all() if current else [],
        "has_documents": ready.exists(),
        "suggestion_docs": list(ready.values_list("original_name", flat=True)[:2]),
        # From the top-bar "Ask your documents" box or Ctrl K: asked automatically on load.
        "prefill": request.GET.get("q", "").strip()[:settings.RAG_MAX_QUESTION_CHARS] if not current else "",
        "saved_ids": set(
            SavedAnswer.objects.filter(user=request.user, message__session=current).values_list("message_id", flat=True)
        ) if current else set(),
    })


def _parse_question(request):
    """(question, session) or a JsonResponse error."""
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    question = str(payload.get("question", "")).strip()
    if not question:
        return JsonResponse({"error": "Please type a question."}, status=400)
    if len(question) > settings.RAG_MAX_QUESTION_CHARS:
        return JsonResponse({"error": "That question is too long."}, status=400)
    session_id = payload.get("session_id")
    if session_id:
        session = get_object_or_404(_user_sessions(request), public_id=session_id)
    else:
        session = ChatSession.objects.create(organization=request.organization, user=request.user)
    return question, session


def _reply_payload(session, reply):
    return {
        "session_id": str(session.public_id),
        "message_id": reply.pk,
        "title": session.title,
        "answer": reply.content,
        "sources": reply.sources,
        "is_error": reply.is_error,
    }


@org_member_required
@require_POST
def send(request):
    parsed = _parse_question(request)
    if isinstance(parsed, JsonResponse):
        return parsed
    question, session = parsed
    try:
        reply = services.ask(session, question)
    except services.QuotaExceeded as exc:
        return JsonResponse({"error": str(exc)}, status=429)
    return JsonResponse(_reply_payload(session, reply))


@org_member_required
@require_POST
def send_stream(request):
    """Like send(), but streams the answer as it's written: one JSON object per line."""
    parsed = _parse_question(request)
    if isinstance(parsed, JsonResponse):
        return parsed
    question, session = parsed
    try:
        events = services.ask_stream(session, question)
    except services.QuotaExceeded as exc:
        return JsonResponse({"error": str(exc)}, status=429)

    def lines():
        yield json.dumps({"type": "start", "session_id": str(session.public_id)}) + "\n"
        for kind, value in events:
            if kind == "done":
                yield json.dumps({"type": "done", **_reply_payload(session, value)}) + "\n"
            else:
                yield json.dumps({"type": kind, "text": value}) + "\n"

    response = StreamingHttpResponse(lines(), content_type="application/x-ndjson")
    response["Cache-Control"] = "no-cache"
    response["X-Accel-Buffering"] = "no"  # tell proxies such as nginx not to hold the stream back
    return response


@org_member_required
@require_POST
def delete_session(request, session_id):
    get_object_or_404(_user_sessions(request), public_id=session_id).delete()
    return redirect("chat:index")


@org_admin_required
def analytics(request):
    organization = request.organization
    now = timezone.now()
    since = now - timedelta(days=ANALYTICS_DAYS - 1)
    questions = Message.objects.filter(organization=organization, role=Message.Role.USER)
    answers = Message.objects.filter(organization=organization, role=Message.Role.ASSISTANT)

    per_day = dict(
        questions.filter(created_at__gte=since.replace(hour=0, minute=0, second=0, microsecond=0))
        .annotate(day=TruncDate("created_at"))
        .values_list("day")
        .annotate(n=Count("id"))
    )
    today = timezone.localdate()
    days = [today - timedelta(days=offset) for offset in range(ANALYTICS_DAYS - 1, -1, -1)]
    peak = max(per_day.values(), default=0)
    daily = [
        {"day": day, "count": per_day.get(day, 0), "pct": round(100 * per_day.get(day, 0) / peak) if peak else 0}
        for day in days
    ]

    recent = question_rows(organization, ANALYTICS_DAYS)[:8]

    documents = Document.objects.filter(organization=organization)
    avg_latency = answers.filter(created_at__gte=since).aggregate(v=Avg("latency_ms"))["v"]
    return render(request, "chat/analytics.html", {
        "days": ANALYTICS_DAYS,
        "total_questions": questions.count(),
        "questions_period": questions.filter(created_at__gte=since).count(),
        "questions_month": organization.queries_this_month(),
        "failed_period": answers.filter(is_error=True, created_at__gte=since).count(),
        "avg_seconds": avg_latency / 1000 if avg_latency else None,
        "by_channel": questions.values("session__channel").annotate(n=Count("id")).order_by("-n"),
        "daily": daily,
        "peak": peak,
        "top_users": questions.exclude(session__user=None)
            .values("session__user__username")
            .annotate(n=Count("id"), last=Max("created_at"))
            .order_by("-n")[:10],
        "top_questions": questions.annotate(q=Lower("content")).values("q").annotate(n=Count("id")).order_by("-n", "q")[:10],
        "recent": recent,
        "document_count": documents.answerable().count(),
        "document_bytes": documents.aggregate(v=Sum("size"))["v"] or 0,
    })
