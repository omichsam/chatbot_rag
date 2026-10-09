from functools import wraps

from django.contrib import messages
from django.contrib.auth.decorators import login_required
from django.core.exceptions import PermissionDenied
from django.shortcuts import redirect

from .models import Membership


def org_required(min_role=Membership.Role.MEMBER):
    """Require a logged-in user with at least `min_role` in the current organization."""

    def decorator(view):
        @wraps(view)
        def wrapped(request, *args, **kwargs):
            if request.organization is None:
                # Platform superadmins usually belong to no organization; their home is the portal.
                if request.user.is_superuser:
                    return redirect("portal:overview")
                messages.info(request, "Create an organization to get started.")
                return redirect("tenants:create")
            if not request.membership.has_role(min_role):
                raise PermissionDenied
            return view(request, *args, **kwargs)

        return login_required(wrapped)

    return decorator


org_member_required = org_required(Membership.Role.MEMBER)
org_admin_required = org_required(Membership.Role.ADMIN)
