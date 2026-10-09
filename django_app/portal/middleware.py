"""Maintenance mode: while a superadmin has it on, everyone else sees a maintenance page."""
from django.http import JsonResponse
from django.shortcuts import render

from .models import platform

# Superadmins must still be able to sign in and switch maintenance off.
ALWAYS_OPEN = ("/accounts/login/", "/accounts/logout/", "/static/", "/platform/", "/admin/")
RETRY_AFTER_SECONDS = 300


class MaintenanceModeMiddleware:
    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request):
        conf = platform()
        user = getattr(request, "user", None)
        if (conf.maintenance_mode and not (user is not None and user.is_superuser)
                and not request.path.startswith(ALWAYS_OPEN)):
            if request.path.startswith("/api/"):
                response = JsonResponse({"success": False, "error": conf.maintenance_message}, status=503)
            else:
                response = render(request, "maintenance.html", {"message": conf.maintenance_message}, status=503)
            response["Retry-After"] = str(RETRY_AFTER_SECONDS)
            return response
        return self.get_response(request)
