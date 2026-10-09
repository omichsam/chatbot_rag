"""Platform Console: operating the whole platform across every organization.

Access is by console role (portal/access.py): owners (superusers) can do
everything; support and finance staff see the parts they need. Every change is
written to the audit log.
"""
import csv
import secrets
from datetime import timedelta

from django.conf import settings
from django.contrib import messages
from django.contrib.auth import get_user_model, login
from django.core.paginator import Paginator
from django.db import transaction
from django.db.models import Q
from django.http import HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils import timezone
from django.utils.http import url_has_allowed_host_and_scheme
from django.views.decorators.http import require_POST

from api.models import APIKey
from chat.explorer import question_rows
from documents import services as document_services
from documents.models import Document
from tenants.middleware import SUPPORT_SESSION_KEY
from tenants.models import Membership, Organization, Plan

from . import ai, forms as portal_forms, health, lifecycle, metrics, totp
from .access import ALL, FINANCE, OWNER, ROLE_LABELS, SESSION_2FA, SUPPORT, portal_role, portal_view
from .models import (PRESETS, AIProvider, AuditLog, PlatformSettings, PlatformStaff, TwoFactorDevice, platform,
                     record)

User = get_user_model()
TEMP_PASSWORD_SESSION_KEY = "portal_temp_password"
BACKUP_CODES_SESSION_KEY = "portal_backup_codes"
PERIODS = (7, 30, 90)


def _days(request, default=30):
    try:
        days = int(request.GET.get("period", default))
    except ValueError:
        return default
    return days if days in PERIODS else default


# --- Dashboard --------------------------------------------------------------------------

@portal_view(*ALL)
def overview(request):
    purged = lifecycle.purge_due()  # deletions whose grace period has ended
    if purged:
        messages.info(request, f"Deleted after their grace period: {', '.join(purged)}.")
    days = _days(request)
    orgs = list(Organization.objects.select_related("plan"))
    stats = metrics.organization_stats(days)
    series, peak = metrics.daily_by_source(days)
    return render(request, "portal/overview.html", {
        "days": days, "periods": PERIODS,
        "ops": metrics.operations(days),
        "series": series, "peak": peak,
        "source_totals": {k: sum(d[k] for d in series) for k in ("web", "api", "test")},
        "attention": metrics.needs_attention(stats, orgs)[:6],
        "org_rows": sorted(((o, stats.get(o.pk, {})) for o in orgs), key=lambda r: -r[1].get("questions", 0)),
        "providers": metrics.provider_insights(days),
        "recent_actions": AuditLog.objects.select_related("actor")[:6],
        "checks": health.status(),
    })


@portal_view(OWNER, FINANCE)
def usage_export(request):
    """Per-organization usage for the period, as CSV (for billing or reporting)."""
    days = _days(request)
    stats = metrics.organization_stats(days)
    response = HttpResponse(content_type="text/csv")
    response["Content-Disposition"] = f'attachment; filename="docassist-usage-{days}d-{timezone.localdate()}.csv"'
    writer = csv.writer(response)
    writer.writerow(["organization", "plan", "active", "members", "documents", "storage_bytes", "questions",
                     "api_questions", "answers", "failed", "monthly_limit"])
    for org in Organization.objects.select_related("plan").order_by("name"):
        s = stats.get(org.pk, {})
        writer.writerow([org.name, org.plan.name if org.plan else "Custom", org.is_active, s.get("members", 0),
                         s.get("docs_ready", 0), s.get("storage", 0), s.get("questions", 0), s.get("api_questions", 0),
                         s.get("answers", 0), s.get("failed", 0), org.monthly_query_limit])
    record(request, "usage.export", None, f"{days} days")
    return response


# --- Organizations -------------------------------------------------------------------------

@portal_view(*ALL)
def organizations(request):
    q = request.GET.get("q", "").strip()
    status = request.GET.get("status", "")
    orgs = Organization.objects.select_related("plan").order_by("name")
    if q:
        orgs = orgs.filter(Q(name__icontains=q) | Q(slug__icontains=q))
    if status == "active":
        orgs = orgs.filter(is_active=True)
    elif status == "suspended":
        orgs = orgs.filter(is_active=False, deletion_scheduled_at__isnull=True)
    elif status == "deleting":
        orgs = orgs.filter(deletion_scheduled_at__isnull=False)
    stats = metrics.organization_stats()
    return render(request, "portal/organizations.html", {
        "rows": [(o, stats.get(o.pk, {})) for o in orgs], "q": q, "status": status,
    })


@portal_view(*ALL)
def organization_detail(request, pk):
    org = get_object_or_404(Organization.objects.select_related("plan"), pk=pk)
    return render(request, "portal/organization_detail.html", {
        "org": org,
        "s": metrics.organization_stats().get(org.pk, {}),
        "members": org.memberships.select_related("user").order_by("user__username"),
        "documents": org.documents.select_related("uploaded_by")[:50],
        "keys": APIKey.objects.filter(organization=org).select_related("created_by"),
        "recent": question_rows(org, 30)[:10] if request.portal_role in (OWNER, SUPPORT) else [],
        "roles": Membership.Role.choices,
        "plans": Plan.objects.all(),
        "queries_this_month": org.queries_this_month(),
        "actions": AuditLog.objects.filter(organization=org).select_related("actor")[:10],
        "grace_days": lifecycle.GRACE_DAYS,
    })


def _confirmed_name(request, org):
    return request.POST.get("confirm_name", "").strip() == org.name


@portal_view(*ALL)
@require_POST
def organization_action(request, pk):
    org = get_object_or_404(Organization, pk=pk)
    action = request.POST.get("action")
    back = redirect("portal:organization", pk=org.pk)

    if action == "suspend":
        reason = request.POST.get("reason", "").strip()[:300]
        if not reason:
            messages.error(request, "Give a reason for suspending; it's shown to the platform team.")
            return back
        org.is_active, org.suspended_reason, org.suspended_at = False, reason, timezone.now()
        org.save(update_fields=["is_active", "suspended_reason", "suspended_at"])
        record(request, "org.suspend", org, org.name, reason)
        messages.success(request, f"{org.name} is suspended.")

    elif action == "activate":
        if org.deletion_scheduled_at:
            messages.error(request, "Cancel the scheduled deletion to reactivate it.")
            return back
        org.is_active, org.suspended_reason, org.suspended_at = True, "", None
        org.save(update_fields=["is_active", "suspended_reason", "suspended_at"])
        record(request, "org.activate", org, org.name)
        messages.success(request, f"{org.name} is active again.")

    elif action == "set_plan":
        plan = Plan.objects.filter(pk=request.POST.get("plan") or 0).first()
        old = org.plan.name if org.plan else "Custom"
        org.plan = plan
        fields = ["plan"]
        if plan is not None:
            org.monthly_query_limit = plan.monthly_query_limit
            fields.append("monthly_query_limit")
        org.save(update_fields=fields)
        record(request, "org.plan", org, org.name, f"Plan: {old} → {plan.name if plan else 'Custom'}")
        messages.success(request, f"{org.name} is now on {plan.name if plan else 'custom limits'}.")

    elif action == "limits":
        try:
            limit = max(0, int(request.POST.get("monthly_query_limit") or 0))
        except ValueError:
            messages.error(request, "The monthly limit must be a whole number.")
            return back
        model = request.POST.get("llm_model", "").strip()[:100]
        detail = f"Monthly limit: {org.monthly_query_limit} → {limit}; Model: {org.llm_model or 'default'} → {model or 'default'}"
        org.monthly_query_limit, org.llm_model = limit, model
        org.save(update_fields=["monthly_query_limit", "llm_model"])
        record(request, "org.limits", org, org.name, detail)
        messages.success(request, "Limits saved.")

    elif action == "add_member":
        username = request.POST.get("username", "").strip()
        role = request.POST.get("role", Membership.Role.MEMBER)
        user = User.objects.filter(username__iexact=username).first()
        if user is None:
            messages.error(request, f"No user called {username}.")
        elif role not in Membership.Role.values:
            messages.error(request, "Pick a valid role.")
        elif Membership.objects.filter(user=user, organization=org).exists():
            messages.error(request, f"{user.username} is already a member.")
        elif org.member_limit_reached():
            messages.error(request, f"The {org.plan.name} plan allows {org.plan.max_members} members.")
        else:
            Membership.objects.create(user=user, organization=org, role=role)
            record(request, "member.add", org, user.username, f"as {role}")
            messages.success(request, f"Added {user.username} as {role}.")

    elif action in ("set_role", "remove_member"):
        membership = get_object_or_404(Membership, pk=request.POST.get("membership"), organization=org)
        owners = org.memberships.filter(role=Membership.Role.OWNER)
        last_owner = membership.role == Membership.Role.OWNER and owners.count() == 1
        if action == "remove_member":
            if last_owner:
                messages.error(request, "That's the only owner. Make someone else owner first.")
                return back
            membership.delete()
            record(request, "member.remove", org, membership.user.username)
            messages.success(request, f"Removed {membership.user.username}.")
        else:
            role = request.POST.get("role")
            if role not in Membership.Role.values:
                messages.error(request, "Pick a valid role.")
            elif last_owner and role != Membership.Role.OWNER:
                messages.error(request, "That's the only owner. Make someone else owner first.")
            else:
                old = membership.role
                membership.role = role
                membership.save(update_fields=["role"])
                record(request, "member.role", org, membership.user.username, f"Role: {old} → {role}")
                messages.success(request, f"{membership.user.username} is now {membership.get_role_display()}.")

    elif action == "revoke_key":
        key = get_object_or_404(APIKey, pk=request.POST.get("key"), organization=org)
        if key.revoked_at is None:
            key.revoked_at = timezone.now()
            key.save(update_fields=["revoked_at"])
            record(request, "key.revoke", org, key.name, key.prefix)
            messages.success(request, f"Revoked {key.name}.")

    elif action == "revoke_all_keys":
        if not _confirmed_name(request, org):
            messages.error(request, "Type the organization's exact name to revoke all its keys.")
            return back
        count = APIKey.objects.filter(organization=org, revoked_at__isnull=True).update(revoked_at=timezone.now())
        record(request, "key.revoke_all", org, org.name, f"{count} keys")
        messages.success(request, f"Revoked {count} key(s).")

    elif action == "reindex_failed":
        failed = list(org.documents.filter(status=Document.Status.FAILED).values_list("pk", flat=True))
        Document.objects.filter(pk__in=failed).update(status=Document.Status.PENDING, error="")
        for pk_ in failed:
            transaction.on_commit(lambda pk_=pk_: document_services.enqueue_ingest(pk_))
        record(request, "docs.reindex_failed", org, f"{len(failed)} documents")
        messages.success(request, f"Re-indexing {len(failed)} failed document(s).")

    elif action == "schedule_delete":
        if not _confirmed_name(request, org):
            messages.error(request, "Type the organization's exact name to schedule its deletion.")
            return back
        lifecycle.schedule_deletion(org, request.POST.get("reason", "").strip()[:300])
        record(request, "org.schedule_delete", org, org.name, f"Deletes on {org.deletion_scheduled_at:%d %b %Y}")
        messages.success(request, f"{org.name} is offline and will be deleted on {org.deletion_scheduled_at:%d %b %Y}. "
                                  "You can cancel until then.")

    elif action == "cancel_delete":
        lifecycle.cancel_deletion(org)
        record(request, "org.cancel_delete", org, org.name)
        messages.success(request, f"Deletion cancelled. {org.name} is active again.")

    elif action == "delete_now":
        if not org.deletion_scheduled_at or not _confirmed_name(request, org):
            messages.error(request, "Schedule the deletion first, then type the exact name to delete it now.")
            return back
        name = org.name
        record(request, "org.delete", None, name, f"{org.documents.count()} documents, {org.memberships.count()} members")
        org.delete()  # signals remove its files and passages
        messages.success(request, f"Deleted {name} and everything in it.")
        return redirect("portal:organizations")

    else:
        messages.error(request, "Unknown action.")
    return back


@portal_view(OWNER)
def organization_export(request, pk):
    org = get_object_or_404(Organization, pk=pk)
    data = lifecycle.export_organization(org)
    record(request, "org.export", org, org.name, f"{len(data):,} bytes")
    response = HttpResponse(data, content_type="application/zip")
    response["Content-Disposition"] = f'attachment; filename="{org.slug}-export-{timezone.localdate()}.zip"'
    return response


# --- Support mode ------------------------------------------------------------------------------

@portal_view(OWNER, SUPPORT, write=(OWNER, SUPPORT))
@require_POST
def support_start(request, pk):
    org = get_object_or_404(Organization, pk=pk)
    request.session[SUPPORT_SESSION_KEY] = org.pk
    record(request, "support.start", org, org.name)
    messages.info(request, f"Viewing {org.name} as an admin would, read-only.")
    return redirect("dashboard")


@require_POST
def support_exit(request):
    org_id = request.session.pop(SUPPORT_SESSION_KEY, None)
    if org_id and request.user.is_authenticated:
        org = Organization.objects.filter(pk=org_id).first()
        record(request, "support.exit", org, org.name if org else str(org_id))
        return redirect("portal:organization", pk=org_id) if org else redirect("portal:overview")
    return redirect("portal:overview")


# --- Plans ---------------------------------------------------------------------------------------

@portal_view(OWNER, FINANCE)
def plans(request):
    if request.method == "POST":
        action = request.POST.get("action")
        plan = Plan.objects.filter(pk=request.POST.get("plan") or 0).first()
        if action == "delete" and plan:
            if plan.organizations.exists():
                messages.error(request, f"{plan.name} is used by {plan.organizations.count()} organization(s). Move them first.")
            else:
                record(request, "plan.delete", None, plan.name)
                plan.delete()
                messages.success(request, "Plan removed.")
            return redirect("portal:plans")
        form = portal_forms.PlanForm(request.POST, instance=plan)
        if form.is_valid():
            saved = form.save()
            record(request, "plan.update" if plan else "plan.add", None, saved.name, form.change_summary())
            if plan and request.POST.get("apply_limits"):
                count = saved.organizations.update(monthly_query_limit=saved.monthly_query_limit)
                messages.success(request, f"{saved.name} saved and applied to {count} organization(s).")
            else:
                messages.success(request, f"{saved.name} saved.")
            return redirect("portal:plans")
        messages.error(request, " ".join(e for errors in form.errors.values() for e in errors))
    rows = [(p, p.organizations.count()) for p in Plan.objects.all()]
    return render(request, "portal/plans.html", {
        "rows": rows, "form": portal_forms.PlanForm(), "custom_count": Organization.objects.filter(plan=None).count(),
    })


# --- Users ---------------------------------------------------------------------------------------

@portal_view(OWNER, SUPPORT)
def users(request):
    q = request.GET.get("q", "").strip()
    qs = User.objects.all().order_by("username").prefetch_related("memberships__organization").select_related(
        "platform_staff", "two_factor")
    if q:
        qs = qs.filter(Q(username__icontains=q) | Q(email__icontains=q))
    flt = request.GET.get("filter", "")
    if flt == "superusers":
        qs = qs.filter(is_superuser=True)
    elif flt == "staff":
        qs = qs.filter(Q(is_superuser=True) | Q(platform_staff__isnull=False))
    elif flt == "inactive":
        qs = qs.filter(is_active=False)
    elif flt == "no_org":
        qs = qs.filter(memberships__isnull=True)
    temp = request.session.pop(TEMP_PASSWORD_SESSION_KEY, None)
    return render(request, "portal/users.html", {
        "users": qs[:300], "q": q, "filter": flt, "temp": temp,
        "staff_roles": PlatformStaff.Role.choices,
    })


@portal_view(OWNER)
@require_POST
def user_action(request, pk):
    target = get_object_or_404(User, pk=pk)
    action = request.POST.get("action")
    is_self = target.pk == request.user.pk

    if action == "reset_password":
        password = secrets.token_urlsafe(12)
        target.set_password(password)
        target.save(update_fields=["password"])
        record(request, "user.reset_password", None, target.username)
        request.session[TEMP_PASSWORD_SESSION_KEY] = {"username": target.username, "password": password}
    elif action in ("deactivate", "activate"):
        if is_self:
            messages.error(request, "You can't deactivate your own account.")
        else:
            target.is_active = action == "activate"
            target.save(update_fields=["is_active"])
            record(request, f"user.{action}", None, target.username)
            messages.success(request, f"{target.username} is now {'active' if target.is_active else 'deactivated'}.")
    elif action in ("grant_super", "revoke_super"):
        if is_self:
            messages.error(request, "You can't change your own superadmin access.")
        elif request.POST.get("confirm_name", "").strip() != target.username:
            messages.error(request, "Type the username exactly to change superadmin access.")
        else:
            grant = action == "grant_super"
            target.is_superuser = target.is_staff = grant
            target.save(update_fields=["is_superuser", "is_staff"])
            record(request, f"user.{action}", None, target.username)
            messages.success(request, f"{target.username} {'is now' if grant else 'is no longer'} a superadmin.")
    elif action == "set_staff_role":
        role = request.POST.get("role", "")
        if target.is_superuser:
            messages.error(request, "Superadmins already have full access.")
        elif role in PlatformStaff.Role.values:
            PlatformStaff.objects.update_or_create(user=target, defaults={"role": role, "created_by": request.user})
            record(request, "user.staff_role", None, target.username, f"Console role: {ROLE_LABELS[role]}")
            messages.success(request, f"{target.username} now has {ROLE_LABELS[role]} access to the console.")
        else:
            PlatformStaff.objects.filter(user=target).delete()
            record(request, "user.staff_role", None, target.username, "Console role: none")
            messages.success(request, f"{target.username} no longer has console access.")
    elif action == "reset_2fa":
        TwoFactorDevice.objects.filter(user=target).delete()
        record(request, "security.2fa_reset", None, target.username)
        messages.success(request, f"Two-factor sign-in removed for {target.username}. They can set it up again.")
    else:
        messages.error(request, "Unknown action.")
    return redirect(request.POST.get("next") or "portal:users")


# --- Security centre -------------------------------------------------------------------------

@portal_view(OWNER, SUPPORT)
def security(request):
    since = timezone.now() - timedelta(days=30)
    events = AuditLog.objects.filter(action__in=["security.login_failed", "security.lockout"], created_at__gte=since)
    staff = (User.objects.filter(Q(is_superuser=True) | Q(platform_staff__isnull=False))
             .select_related("platform_staff", "two_factor").order_by("username"))
    keys = metrics.api_key_inventory()
    return render(request, "portal/security.html", {
        "staff": [(u, portal_role(u), hasattr(u, "two_factor") and u.two_factor.is_confirmed) for u in staff],
        "require_2fa": platform().require_2fa,
        "failed_30d": events.filter(action="security.login_failed").count(),
        "lockouts_30d": events.filter(action="security.lockout").count(),
        "events": events.order_by("-created_at")[:200],
        "keys": keys, "keys_flagged": sum(1 for k in keys if k["flags"]),
    })


@portal_view(OWNER)
@require_POST
def security_action(request):
    action = request.POST.get("action")
    if action in ("require_2fa_on", "require_2fa_off"):
        on = action == "require_2fa_on"
        if on and not TwoFactorDevice.objects.filter(user=request.user, confirmed_at__isnull=False).exists():
            messages.error(request, "Set up your own authenticator first, so you don't lock yourself out.")
            return redirect("portal:two_factor")
        conf = PlatformSettings.objects.get(pk=platform().pk or 1)
        conf.require_2fa, conf.updated_by = on, request.user
        conf.save()
        record(request, "platform.settings", None, "Two-factor requirement",
               f"Require two-factor sign-in: {'off → on' if on else 'on → off'}")
        messages.success(request, f"Two-factor sign-in is now {'required' if on else 'optional'} for the console.")
    elif action == "revoke_key":
        key = get_object_or_404(APIKey, pk=request.POST.get("key"), revoked_at__isnull=True)
        key.revoked_at = timezone.now()
        key.save(update_fields=["revoked_at"])
        record(request, "key.revoke", key.organization, key.name, f"{key.prefix} (from the security centre)")
        messages.success(request, f"Revoked {key.name} ({key.organization.name}).")
    else:
        messages.error(request, "Unknown action.")
    return redirect("portal:security")


# --- Two-factor sign-in ------------------------------------------------------------------------

@portal_view(*ALL, two_factor=False)
def two_factor(request):
    """Set up, check or remove your own authenticator app."""
    device = TwoFactorDevice.objects.filter(user=request.user).first()
    if request.method == "POST":
        action = request.POST.get("action")
        if action == "start":
            device = device or TwoFactorDevice(user=request.user)
            device.set_secret(totp.new_secret())
            device.confirmed_at, device.last_step, device.backup_codes = None, 0, []
            device.save()
        elif action == "confirm" and device and not device.is_confirmed:
            step = totp.matching_step(device.secret, request.POST.get("code", ""))
            if step is None:
                messages.error(request, "That code didn't match. Check the time on your phone and try the newest code.")
            else:
                codes, hashes = totp.new_backup_codes()
                device.confirmed_at, device.last_step, device.backup_codes = timezone.now(), step, hashes
                device.save()
                request.session[SESSION_2FA] = True
                request.session[BACKUP_CODES_SESSION_KEY] = codes
                record(request, "security.2fa_enabled", None, request.user.username)
                messages.success(request, "Two-factor sign-in is on. Save your backup codes now.")
        elif action == "new_codes" and device and device.is_confirmed:
            codes, hashes = totp.new_backup_codes()
            device.backup_codes = hashes
            device.save(update_fields=["backup_codes"])
            request.session[BACKUP_CODES_SESSION_KEY] = codes
            record(request, "security.2fa_codes", None, request.user.username)
        elif action == "disable" and device:
            if platform().require_2fa:
                messages.error(request, "Two-factor sign-in is required for the console, so it can't be removed.")
            elif totp.matching_step(device.secret, request.POST.get("code", "")) is None:
                messages.error(request, "Enter a current code from your authenticator to remove it.")
            else:
                device.delete()
                request.session.pop(SESSION_2FA, None)
                record(request, "security.2fa_disabled", None, request.user.username)
                messages.success(request, "Two-factor sign-in removed.")
        return redirect("portal:two_factor")

    context = {"device": device, "required": platform().require_2fa,
               "backup_codes": request.session.pop(BACKUP_CODES_SESSION_KEY, None)}
    if device and not device.is_confirmed:
        context["secret"] = device.secret
        context["uri"] = totp.provisioning_uri(device.secret, request.user.username)
    if device and device.is_confirmed:
        context["codes_left"] = len(device.backup_codes)
    return render(request, "portal/two_factor.html", context)


MAX_CODE_ATTEMPTS = 5


def two_factor_verify(request):
    """The second sign-in step: a code from the authenticator app, or a backup code."""
    pending = request.session.get("pending_2fa")
    if pending:
        user = User.objects.filter(pk=pending["user"], is_active=True).first()
        next_url = pending.get("next") or reverse("dashboard")
    elif request.user.is_authenticated:
        user = request.user
        next_url = request.GET.get("next") or request.POST.get("next") or reverse("portal:overview")
    else:
        return redirect("login")
    if not url_has_allowed_host_and_scheme(next_url, {request.get_host()}):
        next_url = reverse("dashboard")
    device = TwoFactorDevice.objects.filter(user=user, confirmed_at__isnull=False).first() if user else None
    if device is None:
        request.session.pop("pending_2fa", None)
        return redirect(next_url if request.user.is_authenticated else "login")

    if request.method == "POST":
        code = request.POST.get("code", "").strip()
        step = totp.matching_step(device.secret, code, after_step=device.last_step)
        used_backup = False
        if step is None:
            hashed = totp.hash_backup_code(code)
            if hashed in device.backup_codes:
                device.backup_codes = [h for h in device.backup_codes if h != hashed]
                used_backup = True
        if step is not None or used_backup:
            if step is not None:
                device.last_step = step
            device.save(update_fields=["last_step", "backup_codes"])
            if pending:
                login(request, user, backend="django.contrib.auth.backends.ModelBackend")
                request.session.set_expiry(platform().session_hours * 3600 if pending.get("remember") else 0)
                request.session.pop("pending_2fa", None)
                messages.success(request, f"Welcome back, {user.get_full_name() or user.username}.")
            request.session[SESSION_2FA] = True
            if used_backup:
                messages.warning(request, f"Backup code used. {len(device.backup_codes)} left.")
            return redirect(next_url)
        attempts = request.session.get("two_factor_attempts", 0) + 1
        request.session["two_factor_attempts"] = attempts
        AuditLog.objects.create(action="security.2fa_failed", target=user.username, detail=f"attempt {attempts}")
        if attempts >= MAX_CODE_ATTEMPTS:
            request.session.pop("pending_2fa", None)
            request.session["two_factor_attempts"] = 0
            messages.error(request, "Too many wrong codes. Sign in again.")
            return redirect("login")
        messages.error(request, "That code didn't work. Use the newest code from your app, or a backup code.")
    return render(request, "registration/two_factor_verify.html", {"next": next_url, "username": user.username})


# --- System, settings, providers ------------------------------------------------------------------

@portal_view(OWNER, SUPPORT)
def system(request):
    return render(request, "portal/system.html", {
        "checks": health.status(), "config": health.configuration(),
        "result": request.session.pop("portal_test_result", None),
    })


@portal_view(OWNER, SUPPORT, write=(OWNER, SUPPORT))
@require_POST
def system_test(request):
    kind = request.POST.get("kind", "")
    ok, message = health.run_test(kind)
    record(request, "system.test", None, kind, message)
    request.session["portal_test_result"] = {"kind": kind, "ok": ok, "message": message}
    return redirect("portal:system")


@portal_view(OWNER)
def platform_settings(request):
    conf = platform()
    conf = PlatformSettings.objects.get(pk=conf.pk) if conf.pk else conf  # edit a fresh copy, not the cached one
    if request.method == "POST" and request.POST.get("action") == "reset":
        section = request.POST.get("section", "")
        if section not in {key for key, *_ in portal_forms.SECTIONS}:
            messages.error(request, "Unknown section.")
            return redirect("portal:settings")
        changed = portal_forms.reset_section(conf, section)
        if changed:
            conf.updated_by = request.user
            conf.save()
            record(request, "platform.reset", None, section, "; ".join(changed))
            messages.success(request, f"Restored {len(changed)} setting{'s' if len(changed) != 1 else ''} to the defaults.")
        else:
            messages.info(request, "Those settings already match the defaults.")
        return redirect(f"{request.path}#{section}")

    form = portal_forms.PlatformSettingsForm(request.POST or None, instance=conf)
    if request.method == "POST":
        if form.is_valid():
            if form.has_changed():
                summary = form.change_summary()
                saved = form.save(commit=False)
                saved.updated_by = request.user
                saved.save()
                record(request, "platform.settings", None, "Platform settings", summary)
                messages.success(request, "Platform settings saved. They apply right away.")
            else:
                messages.info(request, "Nothing changed.")
            return redirect("portal:settings")
        messages.error(request, "Some settings need fixing. See the highlighted fields.")
    return render(request, "portal/settings.html", {
        "form": form, "sections": portal_forms.section_fields(form), "conf": platform(),
        "recent": AuditLog.objects.filter(action__startswith="platform.").select_related("actor")[:5],
    })


@portal_view(OWNER)
def ai_providers(request):
    providers = list(AIProvider.objects.all())
    days = _days(request)
    insights = metrics.provider_insights(days)
    return render(request, "portal/ai_providers.html", {
        "providers": providers,
        "chain": [p for p in providers if p.is_enabled],
        "quoted": platform().quoted_fallback,
        "key_links": [(AIProvider.Kind(k).label, p["key_url"]) for k, p in PRESETS.items() if p["key_url"]],
        "insights": insights, "insight_by_id": {r["provider"].pk: r for r in insights["rows"]},
        "days": days, "periods": PERIODS,
    })


@portal_view(OWNER)
def ai_provider_edit(request, pk=None):
    provider = get_object_or_404(AIProvider, pk=pk) if pk else None
    form = ai.AIProviderForm(request.POST or None, instance=provider)
    if request.method == "POST" and form.is_valid():
        saved = form.save()
        # The key itself is never written to the audit log.
        record(request, "ai_provider.update" if provider else "ai_provider.add", None, saved.name,
               f"{saved.get_kind_display()} · {saved.model or 'default model'} · {'enabled' if saved.is_enabled else 'disabled'}")
        if request.POST.get("then") == "test":
            ok, message = ai.test_provider(saved)
            (messages.success if ok else messages.error)(request, f"{saved.name} saved. Test {'passed' if ok else 'failed'}: {message}")
        else:
            messages.success(request, f"{saved.name} saved.")
        return redirect("portal:ai_providers")
    presets = {k: {**p, "label": AIProvider.Kind(k).label} for k, p in PRESETS.items()}
    return render(request, "portal/ai_provider_form.html", {
        "form": form, "provider": provider, "presets": presets,
        "env_gemini_key": bool(settings.GOOGLE_API_KEY), "default_model": platform().default_model,
    })


@portal_view(OWNER)
@require_POST
def ai_provider_action(request, pk):
    provider = get_object_or_404(AIProvider, pk=pk)
    action = request.POST.get("action")
    if action == "test":
        ok, message = ai.test_provider(provider)
        (messages.success if ok else messages.error)(request, f"{provider.name}: {message}")
        record(request, "ai_provider.test", None, provider.name, message)
        return redirect("portal:ai_providers")
    if action in ("enable", "disable"):
        provider.is_enabled = action == "enable"
        provider.save(update_fields=["is_enabled", "updated_at"])
        messages.success(request, f"{provider.name} {'enabled' if provider.is_enabled else 'disabled'}.")
    elif action in ("up", "down"):
        ai.move(provider, -1 if action == "up" else 1)
    elif action == "delete":
        provider.delete()
        ai.renumber()
        messages.success(request, f"Removed {provider.name}.")
    else:
        messages.error(request, "Unknown action.")
        return redirect("portal:ai_providers")
    record(request, f"ai_provider.{action}", None, provider.name)
    return redirect("portal:ai_providers")


# --- Audit log -------------------------------------------------------------------------------------

AUDIT_PAGE_SIZE = 25
AUDIT_GROUPS = [
    ("org.", "Organizations"), ("member.", "Members"), ("user.", "Users"), ("key.", "API keys"),
    ("platform.", "Platform settings"), ("ai_provider.", "AI providers"), ("plan.", "Plans"),
    ("support.", "Support mode"), ("security.", "Security"), ("system.", "System tests"), ("docs.", "Documents"),
    ("usage.", "Exports"),
]


def _audit_queryset(request):
    logs = AuditLog.objects.select_related("actor", "organization")
    f = {k: request.GET.get(k, "").strip() for k in ("action", "actor", "org", "q", "since", "until")}
    if f["action"]:
        logs = logs.filter(action__startswith=f["action"])
    if f["actor"]:
        logs = logs.filter(actor__username__iexact=f["actor"])
    if f["org"]:
        logs = logs.filter(organization_name__icontains=f["org"])
    if f["q"]:
        logs = logs.filter(Q(target__icontains=f["q"]) | Q(detail__icontains=f["q"]))
    if f["since"]:
        logs = logs.filter(created_at__date__gte=f["since"])
    if f["until"]:
        logs = logs.filter(created_at__date__lte=f["until"])
    return logs, f


def _changes(detail):
    """'Label: old → new; Other: a → b' as [(label, old, new)], or [] when the detail isn't a change list."""
    rows = []
    for part in detail.split("; "):
        label, sep, rest = part.partition(": ")
        old, arrow, new = rest.partition(" → ")
        if not (sep and arrow):
            return []
        rows.append((label, old, new))
    return rows


@portal_view(OWNER, SUPPORT)
def audit(request):
    logs, filters = _audit_queryset(request)
    if request.GET.get("format") == "csv":
        response = HttpResponse(content_type="text/csv")
        response["Content-Disposition"] = f'attachment; filename="docassist-audit-{timezone.localdate()}.csv"'
        writer = csv.writer(response)
        writer.writerow(["time", "who", "action", "organization", "target", "detail"])
        for a in logs[:10000]:
            writer.writerow([a.created_at.isoformat(), a.actor.username if a.actor else "", a.action,
                             a.organization_name, a.target, a.detail])
        return response
    page = Paginator(logs, AUDIT_PAGE_SIZE).get_page(request.GET.get("page"))
    for entry in page:
        entry.changes = _changes(entry.detail) if "→" in entry.detail else []
    keep = {k: v for k, v in filters.items() if v}
    return render(request, "portal/audit.html", {
        "page": page, "filters": filters, "groups": AUDIT_GROUPS,
        "page_numbers": page.paginator.get_elided_page_range(page.number, on_each_side=1, on_ends=1),
        "ellipsis": Paginator.ELLIPSIS, "query": "&".join(f"{k}={v}" for k, v in keep.items()),
        "actors": User.objects.filter(pk__in=AuditLog.objects.values("actor")).order_by("username")
                  .values_list("username", flat=True),
    })


# --- Console search and guide ----------------------------------------------------------------------

CONSOLE_PAGES = [
    ("Dashboard", "dashboard", "portal:overview", ALL), ("Organizations", "building", "portal:organizations", ALL),
    ("Plans", "chart", "portal:plans", (OWNER, FINANCE)), ("Users", "users", "portal:users", (OWNER, SUPPORT)),
    ("AI providers", "sparkles", "portal:ai_providers", (OWNER,)),
    ("Security centre", "shield", "portal:security", (OWNER, SUPPORT)),
    ("Audit log", "history", "portal:audit", (OWNER, SUPPORT)),
    ("Platform settings", "settings", "portal:settings", (OWNER,)),
    ("System health", "gauge", "portal:system", (OWNER, SUPPORT)),
    ("Two-factor sign-in", "lock", "portal:two_factor", ALL), ("Portal guide", "book", "portal:guide", ALL),
]


@portal_view(*ALL)
def search(request):
    """Console Ctrl+K: pages, organizations, users and API keys across the whole platform."""
    q = request.GET.get("q", "").strip()[:100]
    role = request.portal_role
    groups = []
    pages = [{"label": label, "icon": icon, "url": reverse(name)} for label, icon, name, roles in CONSOLE_PAGES
             if role in roles and (not q or q.lower() in label.lower())]
    if pages:
        groups.append({"title": "Go to", "items": pages[:6]})
    if len(q) >= 2:
        orgs = Organization.objects.filter(Q(name__icontains=q) | Q(slug__icontains=q))[:5]
        if orgs:
            groups.append({"title": "Organizations", "items": [
                {"label": o.name, "icon": "building", "meta": "active" if o.is_active else "offline",
                 "url": reverse("portal:organization", args=[o.pk])} for o in orgs]})
        if role in (OWNER, SUPPORT):
            people = User.objects.filter(Q(username__icontains=q) | Q(email__icontains=q))[:5]
            if people:
                groups.append({"title": "Users", "items": [
                    {"label": u.username, "icon": "users", "meta": u.email,
                     "url": reverse("portal:users") + "?q=" + u.username} for u in people]})
            keys = APIKey.objects.filter(Q(name__icontains=q) | Q(prefix__icontains=q)).select_related("organization")[:5]
            if keys:
                groups.append({"title": "API keys", "items": [
                    {"label": k.name, "icon": "key", "meta": f"{k.organization.name} · {k.prefix}…",
                     "url": reverse("portal:organization", args=[k.organization_id])} for k in keys]})
    return JsonResponse({"groups": groups})


@portal_view(*ALL)
def guide(request):
    return render(request, "portal/guide.html")
