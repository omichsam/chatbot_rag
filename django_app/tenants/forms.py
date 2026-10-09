from django import forms
from django.contrib.auth import get_user_model
from django.contrib.auth.forms import AuthenticationForm, UserCreationForm
from django.core.cache import cache
from django.contrib.auth.password_validation import validate_password

from .models import Membership, Organization

User = get_user_model()


class LockoutAuthenticationForm(AuthenticationForm):
    """Sign-in that pauses a username + network address after too many wrong passwords.

    Counting per address as well as per username means a stranger can't lock a real
    person out from elsewhere. Successful sign-in clears the count.
    """

    def _failure_key(self):
        username = (self.data.get("username") or "").strip().lower()
        address = self.request.META.get("REMOTE_ADDR", "") if self.request else ""
        return f"login-failures:{username}:{address}"

    def clean(self):
        from portal.models import AuditLog, platform

        conf = platform()
        key = self._failure_key()
        failures = cache.get(key, 0)
        if failures >= conf.login_max_attempts:
            raise forms.ValidationError(
                f"Too many failed sign-ins. Please wait {conf.login_lockout_minutes} minutes and try again.",
                code="locked")
        try:
            data = super().clean()
        except forms.ValidationError:
            failures += 1
            cache.set(key, failures, conf.login_lockout_minutes * 60)
            if failures == conf.login_max_attempts:
                AuditLog.objects.create(
                    action="security.lockout", target=(self.data.get("username") or "")[:200],
                    detail=f"{failures} failed sign-ins from {self.request.META.get('REMOTE_ADDR', '?') if self.request else '?'}; "
                           f"paused for {conf.login_lockout_minutes} minutes")
            raise
        cache.delete(key)
        return data


class SignupForm(UserCreationForm):
    email = forms.EmailField(required=False)
    organization_name = forms.CharField(
        max_length=150, help_text="Your company or team. You can invite others later."
    )

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email")


class OrganizationCreateForm(forms.ModelForm):
    class Meta:
        model = Organization
        fields = ["name"]


class OrganizationSettingsForm(forms.ModelForm):
    MAX_SUGGESTION_CHARS = 150

    class Meta:
        model = Organization
        fields = [
            "name", "assistant_name",
            "welcome_message", "suggested_questions", "show_sources",
            "answer_length", "creativity", "knowledge_scope", "response_language",
            "use_external_ai", "llm_model", "system_prompt",
            "top_k", "history_turns", "gap_threshold",
            "monthly_query_limit", "cache_answers",
        ]
        widgets = {
            "system_prompt": forms.Textarea(attrs={"rows": 4, "placeholder": "You are {assistant}, a friendly document assistant for {org}."}),
            "suggested_questions": forms.Textarea(attrs={"rows": 4, "placeholder": "How do I apply?\nWhat are the eligibility requirements?"}),
            "welcome_message": forms.TextInput(attrs={"placeholder": "Ask anything about our policies and programmes."}),
            "assistant_name": forms.TextInput(attrs={"placeholder": "DocAssist"}),
            "response_language": forms.TextInput(attrs={"placeholder": "Same as the question"}),
            "llm_model": forms.TextInput(attrs={"placeholder": "Platform default"}),
            "answer_length": forms.RadioSelect, "creativity": forms.RadioSelect, "knowledge_scope": forms.RadioSelect,
            "top_k": forms.NumberInput(attrs={"min": 1, "max": 12}),
            "history_turns": forms.NumberInput(attrs={"min": 0, "max": 10}),
            "gap_threshold": forms.NumberInput(attrs={"min": 0.05, "max": 0.9, "step": 0.05}),
            "monthly_query_limit": forms.NumberInput(attrs={"min": 0}),
        }
        help_texts = {
            "system_prompt": "Who the assistant is and how it should sound. {org} and {assistant} are filled in. "
                             "The answer settings on this page are always added after it.",
        }

    def clean_suggested_questions(self):
        lines = [line.strip() for line in self.cleaned_data["suggested_questions"].splitlines() if line.strip()]
        if len(lines) > Organization.MAX_SUGGESTIONS:
            raise forms.ValidationError(f"Add at most {Organization.MAX_SUGGESTIONS} suggestions.")
        too_long = [line for line in lines if len(line) > self.MAX_SUGGESTION_CHARS]
        if too_long:
            raise forms.ValidationError(f"Keep each suggestion under {self.MAX_SUGGESTION_CHARS} characters.")
        return "\n".join(lines)

    def clean_gap_threshold(self):
        return round(self.cleaned_data["gap_threshold"], 2)


class AddMemberForm(forms.Form):
    username = forms.CharField(max_length=150)
    email = forms.EmailField(required=False, help_text="Only used when creating a new account.")
    password = forms.CharField(
        required=False,
        widget=forms.PasswordInput,
        help_text="Leave blank to add an existing user. Fill in to create a new account.",
    )
    role = forms.ChoiceField(choices=Membership.Role.choices, initial=Membership.Role.MEMBER)

    def __init__(self, *args, organization, can_grant_owner, **kwargs):
        super().__init__(*args, **kwargs)
        self.organization = organization
        if not can_grant_owner:
            self.fields["role"].choices = [
                c for c in Membership.Role.choices if c[0] != Membership.Role.OWNER
            ]

    def clean(self):
        cleaned = super().clean()
        if self.organization.member_limit_reached():
            raise forms.ValidationError(
                f"Your plan allows {self.organization.plan.max_members} members. Remove someone or ask the platform team to change the plan.")
        username = cleaned.get("username", "").strip()
        password = cleaned.get("password")
        existing = User.objects.filter(username__iexact=username).first()

        if existing:
            if password:
                raise forms.ValidationError(
                    "That username already exists. Leave the password blank to add them."
                )
            if Membership.objects.filter(user=existing, organization=self.organization).exists():
                raise forms.ValidationError("That user is already a member.")
        else:
            if not password:
                raise forms.ValidationError("No such user. Enter a password to create the account.")
            validate_password(password, User(username=username))

        cleaned["username"] = username
        cleaned["existing_user"] = existing
        return cleaned

    def save(self):
        user = self.cleaned_data["existing_user"]
        if user is None:
            user = User.objects.create_user(
                username=self.cleaned_data["username"],
                email=self.cleaned_data["email"],
                password=self.cleaned_data["password"],
            )
        return Membership.objects.create(
            user=user, organization=self.organization, role=self.cleaned_data["role"]
        )
