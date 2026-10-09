"""The platform settings form, grouped into the tabs the portal shows."""
from django import forms

from .models import PlatformSettings

# (key, title, icon, description, fields) per tab.
SECTIONS = [
    ("access", "Access & security", "shield", "Who can join, and how sign-in is protected.",
     ["allow_signup", "session_hours", "login_max_attempts", "login_lockout_minutes"]),
    ("organizations", "New organizations", "building",
     "Starting values for organizations created from now on. Existing organizations keep their own settings.",
     ["default_model", "default_monthly_limit", "default_top_k", "default_history_turns", "default_gap_threshold"]),
    ("answers", "Documents & answers", "file", "Upload size and what happens when AI providers fail.",
     ["max_upload_mb", "quoted_fallback", "answer_cache_hours"]),
    ("announcement", "Announcement", "bell", "A message at the top of every page, for planned changes or downtime.",
     ["announcement", "announcement_tone"]),
    ("maintenance", "Maintenance", "clock", "Take DocAssist offline for everyone except superadmins.",
     ["maintenance_mode", "maintenance_message"]),
]
BOOLEAN_FIELDS = {"allow_signup", "quoted_fallback", "maintenance_mode"}


class PlatformSettingsForm(forms.ModelForm):
    class Meta:
        model = PlatformSettings
        fields = [name for _, _, _, _, names in SECTIONS for name in names]
        widgets = {
            "announcement_tone": forms.RadioSelect,
            "announcement": forms.TextInput(attrs={"placeholder": "e.g. DocAssist will be offline on Saturday 10:00–11:00."}),
            "session_hours": forms.NumberInput(attrs={"min": 1, "max": 2160}),
            "login_max_attempts": forms.NumberInput(attrs={"min": 3, "max": 20}),
            "login_lockout_minutes": forms.NumberInput(attrs={"min": 1, "max": 1440}),
            "default_monthly_limit": forms.NumberInput(attrs={"min": 0}),
            "default_top_k": forms.NumberInput(attrs={"min": 1, "max": 12}),
            "default_history_turns": forms.NumberInput(attrs={"min": 0, "max": 10}),
            "default_gap_threshold": forms.NumberInput(attrs={"min": 0.05, "max": 0.9, "step": 0.05}),
            "max_upload_mb": forms.NumberInput(attrs={"min": 1, "max": 500}),
            "answer_cache_hours": forms.NumberInput(attrs={"min": 0, "max": 720}),
        }

    def clean_default_model(self):
        value = self.cleaned_data["default_model"].strip()
        if not value:
            raise forms.ValidationError("Enter a model name, e.g. gemini-flash-latest.")
        return value

    def clean_default_gap_threshold(self):
        return round(self.cleaned_data["default_gap_threshold"], 2)

    def clean(self):
        data = super().clean()
        if data.get("maintenance_mode") and not (data.get("maintenance_message") or "").strip():
            self.add_error("maintenance_message", "Tell people why DocAssist is offline.")
        return data

    def change_summary(self):
        """"Label: old → new" for each changed field, for the audit log."""
        parts = []
        for name in self.changed_data:
            field = self.fields[name]
            old, new = self.initial.get(name), self.cleaned_data.get(name)
            if isinstance(field, forms.BooleanField):
                old, new = ("on" if old else "off"), ("on" if new else "off")
            parts.append(f"{field.label}: {old if old not in (None, '') else '–'} → {new if new not in (None, '') else '–'}")
        return "; ".join(parts)


def section_fields(form):
    """[(key, title, icon, description, [(bound field, is_switch, default)])] for the template."""
    model_fields = {f.name: f for f in PlatformSettings._meta.get_fields() if hasattr(f, "get_default")}
    return [
        (key, title, icon, description,
         [(form[name], name in BOOLEAN_FIELDS, model_fields[name].get_default()) for name in names])
        for key, title, icon, description, names in SECTIONS
    ]


def reset_section(conf, key):
    """Put one tab's settings back to their defaults (from .env). Returns the changed labels."""
    names = next(names for k, _, _, _, names in SECTIONS if k == key)
    changed = []
    for name in names:
        field = PlatformSettings._meta.get_field(name)
        default = field.get_default()
        if getattr(conf, name) != default:
            changed.append(f"{field.verbose_name}: {getattr(conf, name)} → {default}")
            setattr(conf, name, default)
    return changed


class PlanForm(forms.ModelForm):
    class Meta:
        from tenants.models import Plan

        model = Plan
        fields = ["name", "description", "monthly_query_limit", "max_documents", "max_members", "allow_external_ai",
                  "sort_order"]

    def change_summary(self):
        if not self.instance.pk or not self.changed_data:
            return ""
        return "; ".join(f"{self.fields[n].label}: {self.initial.get(n)} → {self.cleaned_data.get(n)}" for n in self.changed_data)
