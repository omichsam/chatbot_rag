import logging
import time


from rag import answer_cache
from rag.llm import LLMError
from rag.pipeline import CACHE, NO_DOCUMENTS_ANSWER, QUOTED, Answer, answer_question, stream_answer

from .categorize import categorize_question
from .models import Message

logger = logging.getLogger(__name__)

LLM_UNAVAILABLE_ANSWER = "Sorry, the AI service is unavailable right now. Please try again in a moment."


class QuotaExceeded(Exception):
    pass


def _start(session, question):
    """Checks, history and the stored question shared by ask() and ask_stream()."""
    organization = session.organization
    if organization.quota_exceeded():
        raise QuotaExceeded(f"{organization.name} has used its {organization.monthly_query_limit} questions for this month.")
    recent = session.messages.order_by("-created_at", "-id")[: organization.history_turns * 2]
    history = [(m.role, m.content) for m in reversed(recent) if not m.is_error]
    asked = Message.objects.create(session=session, organization=organization, role=Message.Role.USER, content=question)
    categorize_question(asked)
    return organization, history


def _finish(session, question, answer, started, error_detail=""):
    reply = Message.objects.create(
        session=session,
        organization=session.organization,
        role=Message.Role.ASSISTANT,
        content=answer.text,
        sources=answer.citations,
        is_error=bool(error_detail),
        retrieval_score=answer.top_score,
        # A failure, or why a backup writer answered instead of Gemini.
        error_detail=(error_detail or answer.fallback_reason)[:500],
        answered_by=answer.answered_by,
        latency_ms=int((time.monotonic() - started) * 1000),
    )
    if not session.title:
        session.title = question[:80]
    session.save(update_fields=["title", "updated_at"])
    return reply


def _cached(organization, question, history):
    if history:  # follow-ups depend on the conversation, never reuse them
        return None
    found = answer_cache.get(organization, question)
    return Answer(found[0], found[1], CACHE) if found else None


def _cacheable(answer, history):
    # Backup answers aren't kept: once Gemini is back, the question deserves its full answer.
    return not history and answer.hits and not answer.fallback_reason and answer.answered_by != QUOTED


def ask(session, question):
    """Answer `question` in `session`, store both messages, and return the answer Message."""
    organization, history = _start(session, question)
    started = time.monotonic()
    answer = _cached(organization, question, history)
    if answer is not None:
        return _finish(session, question, answer, started)
    try:
        answer = answer_question(organization, question, history)
    except LLMError as exc:
        logger.warning("LLM error for organization %s: %s", organization.pk, exc)
        return _finish(session, question, Answer(LLM_UNAVAILABLE_ANSWER), started, str(exc) or "AI error")
    if _cacheable(answer, history):
        answer_cache.put(organization, question, answer.text, answer.hits)
    return _finish(session, question, answer, started)


def ask_stream(session, question):
    """Like ask(), but yields ("delta", text) pieces as they arrive, then ("done", reply Message).

    QuotaExceeded is raised before anything is yielded.
    """
    organization, history = _start(session, question)

    def events():
        started = time.monotonic()
        answer = _cached(organization, question, history)
        if answer is not None:
            yield "delta", answer.text
            yield "done", _finish(session, question, answer, started)
            return
        pieces, hits = [], []
        try:
            hits, stream, outcome = stream_answer(organization, question, history)
            for piece in stream:
                pieces.append(piece)
                yield "delta", piece
        except LLMError as exc:
            logger.warning("LLM error for organization %s: %s", organization.pk, exc)
            # Anything already shown is replaced by the error message in the stored answer.
            yield "error", LLM_UNAVAILABLE_ANSWER
            yield "done", _finish(session, question, Answer(LLM_UNAVAILABLE_ANSWER), started, str(exc) or "AI error")
            return
        text = "".join(pieces).strip() or NO_DOCUMENTS_ANSWER
        answer = Answer(text, hits, outcome["answered_by"], outcome["fallback_reason"])
        if _cacheable(answer, history):
            answer_cache.put(organization, question, text, hits)
        yield "done", _finish(session, question, answer, started)

    return events()
