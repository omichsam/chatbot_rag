"""My history page and the Ctrl K search used by the command palette."""
from collections import Counter
from urllib.parse import urlencode

from django.db.models import Count
from django.http import JsonResponse
from django.shortcuts import render
from django.urls import reverse
from django.utils import timezone

from documents.models import Document
from rag import vectorstore
from tenants.decorators import org_member_required
from tenants.menu import build_menu

from .models import ChatSession, Message

PASSAGE_MIN_CHARS = 3


@org_member_required
def history(request):
    q = request.GET.get("q", "").strip()
    sessions = (
        ChatSession.objects.filter(organization=request.organization, user=request.user, channel=ChatSession.Channel.WEB)
        .annotate(n=Count("messages")).order_by("-updated_at")
    )
    if q:
        sessions = sessions.filter(messages__content__icontains=q).distinct()
    today = timezone.localdate()
    groups = {}
    for s in sessions[:200]:
        day = timezone.localtime(s.updated_at).date()
        label = "Today" if day == today else "Yesterday" if (today - day).days == 1 else (
            "This week" if (today - day).days < 7 else day.strftime("%B %Y"))
        groups.setdefault(label, []).append(s)
    return render(request, "chat/history.html", {"groups": groups.items(), "q": q})


@org_member_required
def search(request):
    """Results for the command palette, grouped. Only the current organization is searched."""
    q = request.GET.get("q", "").strip()[:200]
    org, membership = request.organization, request.membership
    lower = q.lower()
    groups = []

    if q:
        groups.append({"title": "Ask", "items": [{
            "label": f'Ask the documents: "{q}"', "icon": "sparkles",
            "url": reverse("chat:index") + "?" + urlencode({"q": q}), "hint": "Enter"}]})

    menu = build_menu(request) or {"pinned": [], "sections": []}
    pages = [i for i in menu["pinned"] + [i for s in menu["sections"] for i in s["items"]]
             if not q or lower in i["label"].lower()]
    if pages:
        groups.append({"title": "Go to", "items": [{"label": i["label"], "icon": i["icon"], "url": i["url"]} for i in pages[:6]]})

    if len(q) >= PASSAGE_MIN_CHARS:
        try:
            hits = vectorstore.search(org.pk, q, 3, exclude_document_ids=Document.objects.paused_ids(org.pk))
        except Exception:
            hits = []
        if hits:
            groups.append({"title": "In your documents", "items": [{
                "label": " ".join(h.text.split())[:110], "icon": "file", "meta": f"{h.filename} · p.{h.page}",
                "url": reverse("chat:index") + "?" + urlencode({"q": q})} for h in hits]})

        docs = Document.objects.filter(organization=org, original_name__icontains=q)[:4]
        if docs:
            groups.append({"title": "Documents", "items": [{
                "label": d.original_name, "icon": "file", "meta": "Paused" if d.is_paused else d.get_status_display(),
                "url": reverse("documents:list")} for d in docs]})

        asked = Message.objects.filter(organization=org, role=Message.Role.USER, content__icontains=q)
        if not membership.is_admin:
            asked = asked.filter(session__user=request.user)
        counts = Counter(m.strip() for m in asked.values_list("content", flat=True)[:300])
        if counts:
            target = reverse("chat:questions") if membership.is_admin else reverse("chat:history")
            groups.append({"title": "Questions people asked" if membership.is_admin else "Your questions", "items": [{
                "label": text[:110], "icon": "chat", "meta": f"asked {n}×" if n > 1 else "",
                "url": target + "?" + urlencode({"q": text[:60]})} for text, n in counts.most_common(4)]})

    actions = [{"label": "New chat", "icon": "plus", "url": reverse("chat:index")},
               {"label": "Customize menu", "icon": "settings", "action": "customize"}]
    if membership.is_admin:
        actions += [{"label": "Upload a document", "icon": "upload", "url": reverse("documents:list")},
                    {"label": "Create an API key", "icon": "key", "url": reverse("keys:list")}]
    actions = [a for a in actions if not q or lower in a["label"].lower()]
    if actions:
        groups.append({"title": "Actions", "items": actions})
    return JsonResponse({"groups": groups})
