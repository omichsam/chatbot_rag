from django.contrib import messages
from django.http import JsonResponse
from django.shortcuts import redirect

from .models import Membership, Organization

SESSION_KEY = "current_organization_id"
SUPPORT_SESSION_KEY = "support_organization_id"
# The only changes allowed while in support mode: leaving it, and signing out.
SUPPORT_ALLOWED_POSTS = ("/platform/support/exit/", "/accounts/logout/")
SAFE_METHODS = ("GET", "HEAD", "OPTIONS")


class CurrentOrganizationMiddleware:
    """Sets request.organization and request.membership for logged-in users.

    The active organization is remembered in the session; if it is missing or
    the user lost access, the user's oldest membership is used instead.
    API views override these from the API key.

    Support mode: a superadmin or support person who chose "View as this
    organization" in the Platform Console sees it as an admin would, read-only.
    """

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        request.organization = None
        request.membership = None
        request.support_mode = False

        user = getattr(request, "user", None)
        if user is not None and user.is_authenticated:
            if request.session.get(SUPPORT_SESSION_KEY) and self._enter_support_mode(request, user):
                if request.method not in SAFE_METHODS and not request.path.startswith(SUPPORT_ALLOWED_POSTS):
                    return self._read_only(request)
                return self.get_response(request)

            memberships = Membership.objects.select_related("organization").filter(
                user=user, organization__is_active=True
            )
            selected_id = request.session.get(SESSION_KEY)
            membership = None
            if selected_id:
                membership = memberships.filter(organization_id=selected_id).first()
            if membership is None:
                membership = memberships.order_by("created_at").first()
            if membership is not None:
                request.membership = membership
                request.organization = membership.organization
                if selected_id != membership.organization_id:
                    request.session[SESSION_KEY] = membership.organization_id

        return self.get_response(request)

    @staticmethod
    def _enter_support_mode(request, user):
        from portal.access import OWNER, SUPPORT, portal_role

        org = Organization.objects.filter(pk=request.session.get(SUPPORT_SESSION_KEY)).first()
        if org is None or portal_role(user) not in (OWNER, SUPPORT):
            request.session.pop(SUPPORT_SESSION_KEY, None)
            return False
        request.organization = org
        # An unsaved membership: the pages render the admin view, nothing is stored.
        request.membership = Membership(user=user, organization=org, role=Membership.Role.ADMIN)
        request.support_mode = True
        return True

    @staticmethod
    def _read_only(request):
        message = "Support mode is read-only. Exit support mode to make changes."
        wants_json = "application/json" in (request.content_type or "") or "application/json" in request.headers.get("Accept", "")
        if wants_json:
            return JsonResponse({"success": False, "error": message}, status=403)
        messages.error(request, message)
        return redirect(request.headers.get("Referer") or "dashboard")
