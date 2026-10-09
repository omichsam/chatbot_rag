"""AI providers in the portal: the add/edit form, live tests and answer order."""
import time

from django import forms
from django.conf import settings
from django.db.models import Max
from django.utils import timezone

from rag import llm

from .models import PRESETS, AIProvider

TEST_PROMPT = "Reply with the single word OK."


def test_provider(provider):
    """Send a tiny request to the provider and remember the result. Returns (ok, message)."""
    model = provider.model_for(None)
    started = time.monotonic()
    if not model:
        ok, message = False, "No model is set."
    else:
        write, _ = llm.writer(provider, model, max_output_tokens=20, temperature=0)
        try:
            reply = write(TEST_PROMPT)
        except llm.LLMError as exc:
            ok, message = False, f"Failed after {time.monotonic() - started:.1f} s: {str(exc)[:240]}"
        else:
            ok, message = True, f"{model} replied “{reply.strip()[:40]}” in {time.monotonic() - started:.1f} s"
    provider.last_test_ok, provider.last_test_message, provider.last_tested_at = ok, message[:300], timezone.now()
    provider.save(update_fields=["last_test_ok", "last_test_message", "last_tested_at"])
    return ok, message


def renumber():
    """Priorities 10, 20, 30... in the current order, so moves are simple swaps."""
    for i, provider in enumerate(AIProvider.objects.all(), start=1):
        if provider.priority != i * 10:
            AIProvider.objects.filter(pk=provider.pk).update(priority=i * 10)


def move(provider, step):
    """Move a provider one place earlier (step=-1) or later (step=1) in the answer order."""
    order = list(AIProvider.objects.all())
    i = next(n for n, p in enumerate(order) if p.pk == provider.pk)
    j = i + step
    if 0 <= j < len(order):
        order[i], order[j] = order[j], order[i]
        for n, p in enumerate(order, start=1):
            AIProvider.objects.filter(pk=p.pk).update(priority=n * 10)


class AIProviderForm(forms.ModelForm):
    api_key = forms.CharField(
        label="API key", required=False, strip=True,
        widget=forms.PasswordInput(render_value=False, attrs={"autocomplete": "new-password", "spellcheck": "false"}))
    clear_key = forms.BooleanField(label="Remove the saved key", required=False)

    class Meta:
        model = AIProvider
        fields = ["kind", "name", "model", "base_url", "timeout", "is_enabled"]
        widgets = {"kind": forms.RadioSelect}

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.fields["name"].required = False
        self.fields["kind"].choices = AIProvider.Kind.choices  # no empty "---------" option
        self.fields["timeout"].widget.attrs.update(min=5, max=600)
        saved = self.instance.pk and self.instance.api_key_encrypted
        self.fields["api_key"].help_text = (
            f"Saved ({self.instance.key_display}). Leave blank to keep it." if saved
            else "Stored encrypted. It is never shown again after saving.")
        if not saved:
            del self.fields["clear_key"]

    def clean(self):
        data = super().clean()
        kind = data.get("kind")
        if not kind:
            return data
        preset = PRESETS[kind]
        if not data.get("name"):
            data["name"] = AIProvider.Kind(kind).label
        if kind == AIProvider.Kind.CUSTOM and not data.get("base_url"):
            self.add_error("base_url", "Enter the API address, e.g. https://example.com/v1")
        if kind != AIProvider.Kind.GEMINI and not data.get("model"):
            example = f", for example {preset['model']}" if preset["model"] else ""
            self.add_error("model", f"Enter the model name{example}.")
        has_key = bool(data.get("api_key")) or (bool(self.instance.api_key_encrypted) and not data.get("clear_key"))
        if kind == AIProvider.Kind.GEMINI and settings.GOOGLE_API_KEY:
            has_key = True  # falls back to GOOGLE_API_KEY from .env
        if preset["needs_key"] and not has_key:
            self.add_error("api_key", "This provider needs an API key.")
        return data

    def save(self, commit=True):
        provider = super().save(commit=False)
        provider.name = self.cleaned_data["name"]
        if self.cleaned_data.get("clear_key"):
            provider.set_api_key("")
        if self.cleaned_data.get("api_key"):
            provider.set_api_key(self.cleaned_data["api_key"])
        if not provider.pk:
            last = AIProvider.objects.aggregate(m=Max("priority"))["m"] or 0
            provider.priority = last + 10
        if commit:
            provider.save()
        return provider
