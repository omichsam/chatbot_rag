import json

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import login
from django.contrib.auth import views as auth_views
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.db import transaction
from django.http import Http404, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from .decorators import org_admin_required, org_member_required
from portal.models import platform

from .forms import AddMemberForm, LockoutAuthenticationForm, OrganizationCreateForm, OrganizationSettingsForm, SignupForm
from .middleware import SESSION_KEY
from .models import AccessRequest, Membership, Organization


class LoginView(auth_views.LoginView):
    """Django's login, plus "Remember me" (unticked: the session ends when the browser closes; ticked: it
    lasts the platform's session length) and a pause after repeated failed attempts."""

    form_class = LockoutAuthenticationForm

    def form_valid(self, form):
        from portal.models import TwoFactorDevice

        user = form.get_user()
        if TwoFactorDevice.objects.filter(user=user, confirmed_at__isnull=False).exists():
            # Password was right; finish signing in only after the authenticator code.
            self.request.session["pending_2fa"] = {
                "user": user.pk, "remember": bool(self.request.POST.get("remember")), "next": self.get_success_url()}
            return redirect("two_factor_verify")
        response = super().form_valid(form)
        if self.request.POST.get("remember"):
            self.request.session.set_expiry(platform().session_hours * 3600)
        else:
            self.request.session.set_expiry(0)
        user = form.get_user()
        messages.success(self.request, f"Welcome back, {user.get_full_name() or user.username}.")
        return response


class LogoutView(auth_views.LogoutView):
    """Django's logout, plus a notice on the sign-in page (messages use a cookie, so they survive the session ending)."""

    def post(self, request, *args, **kwargs):
        response = super().post(request, *args, **kwargs)
        messages.success(request, "You've been signed out. See you soon.")
        return response


def signup(request):
    if not platform().allow_signup:
        raise Http404
    if request.user.is_authenticated:
        return redirect("chat:index")

    form = SignupForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            user = form.save()
            organization = Organization.objects.create(name=form.cleaned_data["organization_name"])
            Membership.objects.create(user=user, organization=organization, role=Membership.Role.OWNER)
        login(request, user, backend="django.contrib.auth.backends.ModelBackend")
        request.session[SESSION_KEY] = organization.pk
        messages.success(request, f"Welcome! {organization.name} is ready. Upload a document to get started.")
        return redirect("documents:list")
    return render(request, "tenants/signup.html", {"form": form})


@login_required
def create_organization(request):
    form = OrganizationCreateForm(request.POST or None)
    if request.method == "POST" and form.is_valid():
        with transaction.atomic():
            organization = form.save()
            Membership.objects.create(user=request.user, organization=organization, role=Membership.Role.OWNER)
        request.session[SESSION_KEY] = organization.pk
        messages.success(request, f"Created {organization.name}.")
        return redirect("documents:list")
    return render(request, "tenants/create.html", {"form": form})


@login_required
@require_POST
def switch_organization(request):
    membership = get_object_or_404(
        Membership, user=request.user, organization_id=request.POST.get("organization"),
        organization__is_active=True,
    )
    request.session[SESSION_KEY] = membership.organization_id
    next_url = request.POST.get("next", "")
    if not url_has_allowed_host_and_scheme(next_url, allowed_hosts={request.get_host()}):
        next_url = "chat:index"
    return redirect(next_url)


@org_admin_required
def organization_settings(request):
    from rag import pipeline
    from rag.pipeline import system_instructions

    organization = request.organization
    form = OrganizationSettingsForm(request.POST or None, instance=organization)
    if request.method == "POST":
        if form.is_valid():
            form.save()
            messages.success(request, "Settings saved. New answers use them right away.")
            return redirect("tenants:settings")
        messages.error(request, "Some settings need fixing. See the highlighted fields.")
        organization.refresh_from_db()  # the preview below should show what is saved, not the rejected input
    return render(request, "tenants/settings.html", {
        "form": form,
        "default_model": platform().default_model,
        "queries_this_month": organization.queries_this_month(),
        "instructions_preview": system_instructions(organization),
        "ai_chain": pipeline.providers(organization),
        "choice_help": CHOICE_HELP,
        "platform_defaults": {
            "top_k": settings.RAG_TOP_K, "history_turns": settings.RAG_HISTORY_TURNS, "gap_threshold": settings.RAG_GAP_SCORE,
        },
    })


# One line under each option of the choice settings.
CHOICE_HELP = {
    "answer_length": {
        "short": "1–3 sentences. Quickest to read and to generate.",
        "balanced": "2–6 sentences or a short list. Recommended.",
        "detailed": "Thorough, with headings or lists. Slower.",
    },
    "creativity": {
        "precise": "Sticks closely to the wording of the documents. Recommended.",
        "balanced": "More natural phrasing, still grounded.",
        "flexible": "Freer wording and summaries. May drift.",
    },
    "knowledge_scope": {
        "strict": "Answers only from your documents and says so when they don't cover a question.",
        "assisted": "Uses your documents first and may add general knowledge, clearly labelled.",
    },
}


@org_admin_required
def members(request):
    is_owner = request.membership.role == Membership.Role.OWNER
    form = AddMemberForm(
        request.POST or None, organization=request.organization, can_grant_owner=is_owner
    )
    if request.method == "POST" and form.is_valid():
        membership = form.save()
        messages.success(request, f"Added {membership.user.username} as {membership.get_role_display()}.")
        return redirect("tenants:members")

    return render(request, "tenants/members.html", {
        "form": form,
        "memberships": request.organization.memberships.select_related("user"),
        "access_requests": AccessRequest.objects.filter(
            organization=request.organization, status=AccessRequest.Status.PENDING).select_related("user"),
        "roles": Membership.Role.choices,
        "is_owner": is_owner,
    })


@org_admin_required
@require_POST
def update_member(request, membership_id):
    target = get_object_or_404(Membership, pk=membership_id, organization=request.organization)
    is_owner = request.membership.role == Membership.Role.OWNER
    action = request.POST.get("action")
    new_role = request.POST.get("role")

    if target.pk == request.membership.pk:
        messages.error(request, "You can't change your own membership here.")
        return redirect("tenants:members")
    # Only owners may touch owners or create new ones.
    if not is_owner and (target.role == Membership.Role.OWNER or new_role == Membership.Role.OWNER):
        raise PermissionDenied

    if action == "remove":
        target.delete()
        messages.success(request, f"Removed {target.user.username}.")
    elif action == "role" and new_role in Membership.Role.values:
        target.role = new_role
        target.save(update_fields=["role"])
        messages.success(request, f"{target.user.username} is now {target.get_role_display()}.")
    return redirect("tenants:members")


# --- Menu preferences and access requests -----------------------------------------

@org_member_required
@require_POST
def save_menu_prefs(request):
    from .menu import clean_prefs

    try:
        data = json.loads(request.body or "{}")
    except json.JSONDecodeError:
        return JsonResponse({"error": "Invalid JSON."}, status=400)
    if data.get("reset"):
        prefs = {}
    else:
        prefs = clean_prefs(data, request.membership)
    request.membership.menu_prefs = prefs
    request.membership.save(update_fields=["menu_prefs"])
    return JsonResponse({"ok": True, "prefs": prefs})


@org_member_required
@require_POST
def request_access(request):
    if request.membership.is_admin:
        messages.info(request, "You already have admin access.")
    elif AccessRequest.objects.filter(organization=request.organization, user=request.user,
                                      status=AccessRequest.Status.PENDING).exists():
        messages.info(request, "Your request is already waiting for an admin.")
    else:
        AccessRequest.objects.create(organization=request.organization, user=request.user,
                                     message=request.POST.get("message", "").strip()[:300])
        messages.success(request, "Request sent. The organization's admins will see it on their Home page.")
    return redirect(request.POST.get("next") or "dashboard")


@org_admin_required
@require_POST
def decide_access(request, pk):
    access = get_object_or_404(AccessRequest, pk=pk, organization=request.organization, status=AccessRequest.Status.PENDING)
    approve = request.POST.get("decision") == "approve"
    access.status = AccessRequest.Status.APPROVED if approve else AccessRequest.Status.DISMISSED
    access.decided_by, access.decided_at = request.user, timezone.now()
    access.save(update_fields=["status", "decided_by", "decided_at"])
    if approve:
        Membership.objects.filter(organization=request.organization, user=access.user,
                                  role=Membership.Role.MEMBER).update(role=Membership.Role.ADMIN)
        messages.success(request, f"{access.user.username} is now an admin.")
    else:
        messages.info(request, f"Dismissed {access.user.username}'s request.")
    return redirect(reverse("tenants:members") + "#requests")
