"""Web pages for managing an organization's API keys."""
from django import forms
from django.contrib import messages
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from tenants.decorators import org_admin_required, org_member_required

from .models import APIKey


class CreateKeyForm(forms.Form):
    name = forms.CharField(max_length=100, widget=forms.TextInput(attrs={"placeholder": "e.g. Website widget"}))


@org_admin_required
def key_list(request):
    form = CreateKeyForm(request.POST or None)
    new_key = None
    if request.method == "POST" and form.is_valid():
        _api_key, new_key = APIKey.create_key(request.organization, form.cleaned_data["name"], request.user)
        form = CreateKeyForm()
    return render(request, "api/keys.html", {
        "form": form,
        "new_key": new_key,
        "keys": APIKey.objects.filter(organization=request.organization).select_related("created_by"),
        "api_base": request.build_absolute_uri("/api/v1/"),
    })


@org_member_required
def developer(request):
    """Interactive API playground. Requests run from the browser with the key the user pastes."""
    return render(request, "api/developer.html", {
        "api_base": request.build_absolute_uri("/api/v1/"),
        "active_keys": APIKey.objects.filter(organization=request.organization, revoked_at__isnull=True).count(),
    })


@org_admin_required
@require_POST
def revoke(request, pk):
    api_key = get_object_or_404(APIKey, pk=pk, organization=request.organization)
    if api_key.revoked_at is None:
        api_key.revoked_at = timezone.now()
        api_key.save(update_fields=["revoked_at"])
        messages.success(request, f"Revoked {api_key.name}. It stops working immediately.")
    return redirect("keys:list")
