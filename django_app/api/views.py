"""Public JSON API, authenticated by organization API keys.

Response shapes match the Streamlit-era shit_api.py so existing clients keep
working; /api/chat, /api/health and /api/validate-key remain as aliases.
"""
import json
import uuid
from datetime import timedelta
from functools import wraps

from django.conf import settings
from django.http import JsonResponse
from django.utils import timezone
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_GET, require_POST

from chat import services
from chat.models import ChatSession
from documents.models import Document
from rag import vectorstore

from .models import APIKey

API_VERSION = settings.APP_VERSION
# Sent by the in-app Developer playground so its requests show up as "Test" in analytics.
TEST_CLIENT_HEADER = "X-DocAssist-Client"
LAST_USED_RESOLUTION = timedelta(minutes=1)


def error(status, message, detail=None):
    return JsonResponse({"success": False, "error": message, "detail": detail}, status=status)


def _key_from_request(request):
    header = request.headers.get("Authorization", "")
    if header.lower().startswith("bearer "):
        return header[7:].strip()
    return request.headers.get("X-API-Key", "").strip()


def _authenticate(request):
    api_key = APIKey.authenticate(_key_from_request(request))
    if api_key is not None:
        now = timezone.now()
        if api_key.last_used_at is None or now - api_key.last_used_at > LAST_USED_RESOLUTION:
            APIKey.objects.filter(pk=api_key.pk).update(last_used_at=now)
        request.api_key = api_key
        request.organization = api_key.organization
    return api_key


def api_key_required(view):
    @wraps(view)
    def wrapped(request, *args, **kwargs):
        if _authenticate(request) is None:
            return error(401, "Missing or invalid API key.",
                         "Send it as 'Authorization: Bearer <key>' or 'X-API-Key: <key>'.")
        return view(request, *args, **kwargs)

    return csrf_exempt(wrapped)


@require_POST
@api_key_required
def chat(request):
    try:
        payload = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return error(400, "Request body must be JSON.")
    if not isinstance(payload, dict):
        return error(400, "Request body must be a JSON object.")

    # "query" is accepted for clients of the original api.py.
    question = str(payload.get("question") or payload.get("query") or "").strip()
    if not question:
        return error(400, "'question' is required.")
    if len(question) > settings.RAG_MAX_QUESTION_CHARS:
        return error(400, f"'question' must be at most {settings.RAG_MAX_QUESTION_CHARS} characters.")

    session = None
    session_id = payload.get("session_id")
    if session_id:
        try:
            session = ChatSession.objects.filter(
                public_id=uuid.UUID(str(session_id)), organization=request.organization,
                channel=ChatSession.Channel.API,
            ).first()
        except ValueError:
            session = None  # unknown/legacy id: start a fresh session
    if session is None:
        session = ChatSession.objects.create(
            organization=request.organization, api_key=request.api_key, channel=ChatSession.Channel.API,
            is_test=request.headers.get(TEST_CLIENT_HEADER, "").lower() == "playground",
        )

    try:
        reply = services.ask(session, question)
    except services.QuotaExceeded as exc:
        return error(429, "Monthly question limit reached.", str(exc))

    return JsonResponse({
        "success": not reply.is_error,
        "answer": reply.content,
        "session_id": str(session.public_id),
        "sources": list(dict.fromkeys(s["filename"] for s in reply.sources)),
        "citations": reply.sources,
        "timestamp": reply.created_at.isoformat(),
    }, status=503 if reply.is_error else 200)


@csrf_exempt
@require_GET
def health(request):
    data = {"status": "healthy", "timestamp": timezone.now().isoformat(), "version": API_VERSION}
    # Document stats are tenant data, so only include them for a valid key.
    if _authenticate(request) is not None:
        count = vectorstore.count(request.organization.pk)
        data.update({"documents_loaded": count > 0, "documents_count": count})
    return JsonResponse(data)


@require_GET
@api_key_required
def validate_key(request):
    return JsonResponse({
        "success": True,
        "message": "API key is valid",
        "organization": request.organization.name,
    })


@require_GET
@api_key_required
def documents(request):
    ready = Document.objects.filter(organization=request.organization).answerable()
    return JsonResponse({
        "success": True,
        "documents": [
            {
                "id": d.pk,
                "filename": d.original_name,
                "pages": d.page_count,
                "chunks": d.chunk_count,
                "size": d.size,
                "uploaded_at": d.created_at.isoformat(),
            }
            for d in ready
        ],
    })
