"""Answer feedback, saved answers, knowledge gaps and the help page."""
import json

from django.contrib import messages
from django.db.models import Q
from django.http import JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from rag.pipeline import NO_DOCUMENTS_ANSWER
from tenants.decorators import org_admin_required, org_member_required

from .models import ChatSession, Message, SavedAnswer

GAP_LIMIT = 300


def _own_answer(request, message_id):
    """An assistant message from one of the current user's web chats in this organization."""
    return get_object_or_404(
        Message,
        pk=message_id,
        role=Message.Role.ASSISTANT,
        organization=request.organization,
        session__user=request.user,
        session__channel=ChatSession.Channel.WEB,
    )


def gap_answers(organization):
    """Answers that suggest the documents don't cover the question."""
    return Message.objects.filter(
        organization=organization, role=Message.Role.ASSISTANT, is_error=False, gap_resolved=False,
    ).filter(
        Q(retrieval_score__lt=organization.gap_threshold) | Q(content=NO_DOCUMENTS_ANSWER) | Q(rating=-1)
    )


def pending_feedback(organization):
    return Message.objects.filter(organization=organization, rating=-1, feedback_reviewed=False)


def _questions_for(answers):
    """Map answer id -> the question message it replied to, in two queries."""
    session_ids = {a.session_id for a in answers}
    questions = list(
        Message.objects.filter(session_id__in=session_ids, role=Message.Role.USER)
        .select_related("session__user", "session__api_key").order_by("id")
    )
    by_session = {}
    for q in questions:
        by_session.setdefault(q.session_id, []).append(q)
    result = {}
    for a in answers:
        earlier = [q for q in by_session.get(a.session_id, []) if q.id < a.id]
        result[a.id] = earlier[-1] if earlier else None
    return result


def _asker(question):
    if question is None:
        return "Unknown"
    session = question.session
    if session.user:
        return session.user.username
    return f"API · {session.api_key.name}" if session.api_key else "API"


# --- Rating and saving (from the chat page) ------------------------------------

@org_member_required
@require_POST
def rate(request, message_id):
    answer = _own_answer(request, message_id)
    try:
        payload = json.loads(request.body or "{}")
        rating = int(payload.get("rating", 0))
    except (ValueError, TypeError, json.JSONDecodeError):
        return JsonResponse({"error": "Invalid rating."}, status=400)
    if rating not in (-1, 0, 1):
        return JsonResponse({"error": "Invalid rating."}, status=400)

    answer.rating = rating or None
    answer.feedback_comment = str(payload.get("comment", ""))[:2000] if rating == -1 else ""
    answer.rated_at = timezone.now() if rating else None
    answer.feedback_reviewed = False
    answer.save(update_fields=["rating", "feedback_comment", "rated_at", "feedback_reviewed"])
    return JsonResponse({"rating": answer.rating})


@org_member_required
@require_POST
def toggle_save(request, message_id):
    answer = _own_answer(request, message_id)
    saved, created = SavedAnswer.objects.get_or_create(user=request.user, message=answer)
    if not created:
        saved.delete()
    if request.headers.get("Accept", "").startswith("application/json") or request.content_type == "application/json":
        return JsonResponse({"saved": created})
    messages.success(request, "Saved." if created else "Removed from saved answers.")
    return redirect(request.POST.get("next") or "chat:saved")


# --- Pages ---------------------------------------------------------------------

@org_member_required
def saved_answers(request):
    saves = list(
        SavedAnswer.objects.filter(user=request.user, message__organization=request.organization)
        .select_related("message__session")
    )
    questions = _questions_for([s.message for s in saves])
    return render(request, "chat/saved.html", {
        "items": [{"save": s, "answer": s.message, "question": questions.get(s.message_id)} for s in saves],
    })


@org_admin_required
def feedback(request):
    tab = request.GET.get("tab", "negative")
    answers = Message.objects.filter(organization=request.organization, rating__isnull=False)
    counts = {
        "negative": answers.filter(rating=-1, feedback_reviewed=False).count(),
        "positive": answers.filter(rating=1).count(),
        "reviewed": answers.filter(rating=-1, feedback_reviewed=True).count(),
    }
    if tab == "positive":
        answers = answers.filter(rating=1)
    elif tab == "reviewed":
        answers = answers.filter(rating=-1, feedback_reviewed=True)
    else:
        tab = "negative"
        answers = answers.filter(rating=-1, feedback_reviewed=False)
    answers = list(answers.select_related("session").order_by("-rated_at")[:100])
    questions = _questions_for(answers)
    return render(request, "chat/feedback.html", {
        "tab": tab,
        "counts": counts,
        "items": [{"answer": a, "question": questions.get(a.id), "asker": _asker(questions.get(a.id))} for a in answers],
    })


@org_admin_required
@require_POST
def mark_reviewed(request, message_id):
    answer = get_object_or_404(Message, pk=message_id, organization=request.organization, rating=-1)
    answer.feedback_reviewed = True
    answer.save(update_fields=["feedback_reviewed"])
    messages.success(request, "Marked as reviewed.")
    return redirect("chat:feedback")


@org_admin_required
def knowledge_gaps(request):
    answers = list(gap_answers(request.organization).order_by("-created_at")[:GAP_LIMIT])
    questions = _questions_for(answers)

    groups = {}
    for answer in answers:
        question = questions.get(answer.id)
        text = question.content.strip() if question else "(question not found)"
        group = groups.setdefault(text.lower(), {
            "question": text, "count": 0, "last": answer.created_at, "ids": [], "reasons": set(), "askers": set(),
        })
        group["count"] += 1
        group["ids"].append(answer.id)
        group["askers"].add(_asker(question))
        if answer.content == NO_DOCUMENTS_ANSWER:
            group["reasons"].add("No documents matched")
        elif answer.rating == -1:
            group["reasons"].add("Rated not helpful")
        if answer.retrieval_score is not None and answer.retrieval_score < request.organization.gap_threshold:
            group["reasons"].add(f"Weak match ({answer.retrieval_score:.0%})")

    ordered = sorted(groups.values(), key=lambda g: (-g["count"], -g["last"].timestamp()))
    return render(request, "chat/gaps.html", {"groups": ordered, "threshold": request.organization.gap_threshold})


@org_admin_required
@require_POST
def resolve_gap(request):
    ids = [int(i) for i in request.POST.getlist("ids") if i.isdigit()]
    updated = Message.objects.filter(
        organization=request.organization, role=Message.Role.ASSISTANT, pk__in=ids
    ).update(gap_resolved=True)
    messages.success(request, f"Marked {updated} answer{'s' if updated != 1 else ''} as resolved.")
    return redirect("chat:gaps")


@org_member_required
def help_page(request):
    return render(request, "chat/help.html")
