from django.conf import settings
from django.core.cache import cache
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import OperationalError, ProgrammingError, models, transaction


class AuditLog(models.Model):
    """One row per action a superadmin takes in the platform portal."""

    actor = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    action = models.CharField(max_length=40)
    # Kept as text too, so the entry still reads correctly after the organization is deleted.
    organization = models.ForeignKey("tenants.Organization", null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    organization_name = models.CharField(max_length=150, blank=True)
    target = models.CharField(max_length=200, blank=True)
    detail = models.TextField(blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at", "-id"]

    def __str__(self):
        return f"{self.actor} {self.action} {self.target}"


def record(request, action, organization=None, target="", detail=""):
    return AuditLog.objects.create(
        actor=request.user,
        action=action,
        organization=organization,
        organization_name=organization.name if organization else "",
        target=str(target)[:200],
        detail=detail,
    )


# --- AI providers ---------------------------------------------------------------
# The models that write answers, managed by superadmins in the portal. Answers try
# the enabled providers in priority order and fall back to quoting the documents
# (rag/pipeline.py). API keys are stored encrypted with a key derived from
# DJANGO_SECRET_KEY; changing that secret means re-entering the keys.

# Per kind: how to call it, where it lives, a model to start with and where to get a key.
# The model names are starting points; each provider's own list is the reference.
PRESETS = {
    "gemini": {"style": "gemini", "base_url": "", "model": "gemini-flash-latest",
               "key_url": "https://aistudio.google.com/apikey", "needs_key": True, "external": True},
    "openai": {"style": "openai", "base_url": "https://api.openai.com/v1", "model": "gpt-4o-mini",
               "key_url": "https://platform.openai.com/api-keys", "needs_key": True, "external": True},
    "anthropic": {"style": "anthropic", "base_url": "https://api.anthropic.com/v1", "model": "claude-haiku-4-5-20251001",
                  "key_url": "https://console.anthropic.com/settings/keys", "needs_key": True, "external": True},
    "groq": {"style": "openai", "base_url": "https://api.groq.com/openai/v1", "model": "llama-3.3-70b-versatile",
             "key_url": "https://console.groq.com/keys", "needs_key": True, "external": True},
    "deepseek": {"style": "openai", "base_url": "https://api.deepseek.com/v1", "model": "deepseek-chat",
                 "key_url": "https://platform.deepseek.com/api_keys", "needs_key": True, "external": True},
    "openrouter": {"style": "openai", "base_url": "https://openrouter.ai/api/v1", "model": "meta-llama/llama-3.3-70b-instruct",
                   "key_url": "https://openrouter.ai/keys", "needs_key": True, "external": True},
    "ollama": {"style": "openai", "base_url": "http://localhost:11434/v1", "model": "llama3.2:3b",
               "key_url": "https://ollama.com/download", "needs_key": False, "external": False},
    "custom": {"style": "openai", "base_url": "", "model": "",
               "key_url": "", "needs_key": False, "external": True},
}


def _fernet():
    import base64
    import hashlib

    from cryptography.fernet import Fernet

    digest = hashlib.sha256(("docassist-ai-provider-keys:" + settings.SECRET_KEY).encode()).digest()
    return Fernet(base64.urlsafe_b64encode(digest))


class AIProvider(models.Model):
    class Kind(models.TextChoices):
        GEMINI = "gemini", "Google Gemini"
        OPENAI = "openai", "OpenAI (ChatGPT)"
        ANTHROPIC = "anthropic", "Anthropic Claude"
        GROQ = "groq", "Groq"
        DEEPSEEK = "deepseek", "DeepSeek"
        OPENROUTER = "openrouter", "OpenRouter"
        OLLAMA = "ollama", "Local model (Ollama)"
        CUSTOM = "custom", "Other (OpenAI-compatible)"

    name = models.CharField(max_length=60, help_text="Shown to admins, e.g. \"Groq Llama 70B\".")
    kind = models.CharField("Provider", max_length=12, choices=Kind.choices)
    model = models.CharField(
        max_length=120, blank=True,
        help_text="The model to use, exactly as the provider names it. For Gemini, blank follows the platform default.")
    base_url = models.URLField(
        "API address", max_length=300, blank=True,
        help_text="Filled in for known providers. Change it only for a proxy or your own server.")
    api_key_encrypted = models.TextField(blank=True, editable=False)
    key_hint = models.CharField(max_length=8, blank=True, editable=False)
    is_enabled = models.BooleanField("Enabled", default=True)
    priority = models.PositiveIntegerField(default=100, help_text="Lower answers first.")
    timeout = models.PositiveSmallIntegerField(
        "Timeout (seconds)", default=60, help_text="How long to wait before moving on to the next provider.")
    last_test_ok = models.BooleanField(null=True, blank=True)
    last_test_message = models.CharField(max_length=300, blank=True)
    last_tested_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["priority", "id"]

    def __str__(self):
        return self.name

    @property
    def preset(self):
        return PRESETS.get(self.kind, PRESETS["custom"])

    @property
    def style(self):
        return self.preset["style"]

    @property
    def is_external(self):
        """Sends text off this server (everything except a local model)."""
        return self.preset["external"]

    @property
    def effective_base_url(self):
        return (self.base_url or self.preset["base_url"]).rstrip("/")

    @property
    def api_key(self):
        """The decrypted key, or "" when none is stored or it can't be decrypted."""
        if not self.api_key_encrypted:
            return ""
        from cryptography.fernet import InvalidToken

        try:
            return _fernet().decrypt(self.api_key_encrypted.encode()).decode()
        except InvalidToken:
            return ""

    def set_api_key(self, raw):
        raw = raw.strip()
        self.api_key_encrypted = _fernet().encrypt(raw.encode()).decode() if raw else ""
        self.key_hint = raw[-4:] if raw else ""

    @property
    def key_unreadable(self):
        return bool(self.api_key_encrypted) and not self.api_key

    @property
    def key_display(self):
        if self.key_unreadable:
            return "Re-enter the key (server secret changed)"
        if self.api_key_encrypted:
            return f"•••• {self.key_hint}"
        if self.kind == self.Kind.GEMINI and settings.GOOGLE_API_KEY:
            return "From .env (GOOGLE_API_KEY)"
        return "Not needed" if not self.preset["needs_key"] else "Missing"

    def model_for(self, organization=None):
        """The model name to call. Gemini follows an organization's own model choice."""
        if self.kind == self.Kind.GEMINI:
            org_choice = organization.llm_model if organization is not None else ""
            return org_choice or self.model or platform().default_model
        return self.model


# --- Platform settings ---------------------------------------------------------------
# One row of platform-wide settings that superadmins change in the portal. Each starts
# from its .env value (config/settings.py), so a fresh install behaves as before;
# after that the portal is the place to change them, without a restart.

# Defaults read from .env (named functions, so migrations can refer to them).
def env_allow_signup():
    return settings.ALLOW_SIGNUP


def env_default_model():
    return settings.RAG_DEFAULT_MODEL


def env_top_k():
    return settings.RAG_TOP_K


def env_history_turns():
    return settings.RAG_HISTORY_TURNS


def env_gap_threshold():
    return settings.RAG_GAP_SCORE


def env_max_upload_mb():
    return settings.RAG_MAX_UPLOAD_MB


def env_quoted_fallback():
    return settings.RAG_QUOTED_FALLBACK


def env_session_hours():
    return settings.SESSION_COOKIE_AGE // 3600


def env_answer_cache_hours():
    return settings.RAG_ANSWER_CACHE_SECONDS // 3600


class PlatformSettings(models.Model):
    class Tone(models.TextChoices):
        INFO = "info", "Information"
        WARNING = "warning", "Warning"

    # Access and security
    allow_signup = models.BooleanField(
        "Allow self sign-up", default=env_allow_signup,
        help_text="Anyone can create an account and their own organization. Off: only superadmins and organization admins add people.")
    session_hours = models.PositiveSmallIntegerField(
        "Stay signed in for (hours)", default=env_session_hours,
        validators=[MinValueValidator(1), MaxValueValidator(2160)],
        help_text="When “Remember me” is ticked (1–2160). Without it, the session ends when the browser closes.")
    login_max_attempts = models.PositiveSmallIntegerField(
        "Failed sign-ins before a pause", default=5, validators=[MinValueValidator(3), MaxValueValidator(20)],
        help_text="Per username and network address, to slow down password guessing (3–20).")
    login_lockout_minutes = models.PositiveSmallIntegerField(
        "Pause length (minutes)", default=15, validators=[MinValueValidator(1), MaxValueValidator(1440)])

    # Defaults for new organizations
    default_model = models.CharField(
        "Default Gemini model", max_length=100, default=env_default_model,
        help_text="Used when an organization hasn't chosen its own. “gemini-flash-latest” follows Google's current Flash model.")
    default_monthly_limit = models.PositiveIntegerField(
        "Monthly question limit", default=0, help_text="For new organizations. 0 means unlimited.")
    default_top_k = models.PositiveSmallIntegerField(
        "Passages per answer", default=env_top_k, validators=[MinValueValidator(1), MaxValueValidator(12)])
    default_history_turns = models.PositiveSmallIntegerField(
        "Conversation memory (turns)", default=env_history_turns, validators=[MaxValueValidator(10)])
    default_gap_threshold = models.FloatField(
        "Knowledge-gap threshold", default=env_gap_threshold,
        validators=[MinValueValidator(0.05), MaxValueValidator(0.9)])

    # Documents and answers
    max_upload_mb = models.PositiveSmallIntegerField(
        "Largest upload (MB)", default=env_max_upload_mb, validators=[MinValueValidator(1), MaxValueValidator(500)])
    quoted_fallback = models.BooleanField(
        "Quote the documents when every AI provider fails", default=env_quoted_fallback,
        help_text="Off: people see “AI service unavailable” instead, and the API returns 503.")
    answer_cache_hours = models.PositiveSmallIntegerField(
        "Reuse answers to repeated questions for (hours)", default=env_answer_cache_hours,
        validators=[MaxValueValidator(720)], help_text="0 turns reuse off for everyone (0–720).")

    # Announcement and maintenance
    announcement = models.CharField(
        max_length=300, blank=True, help_text="Shown at the top of every page for signed-in people. Blank shows nothing.")
    announcement_tone = models.CharField(max_length=10, choices=Tone.choices, default=Tone.INFO)
    maintenance_mode = models.BooleanField(
        default=False,
        help_text="Everyone except superadmins sees a maintenance page, and the API returns 503. Superadmins can still sign in.")
    maintenance_message = models.CharField(
        max_length=300, default="DocAssist is being updated and will be back shortly.")
    require_2fa = models.BooleanField(
        "Require two-factor sign-in for the Platform Console", default=False,
        help_text="Superadmins and platform staff must set up an authenticator app before using the console.")

    updated_at = models.DateTimeField(auto_now=True)
    updated_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")

    class Meta:
        verbose_name_plural = "platform settings"

    def __str__(self):
        return "Platform settings"

    def save(self, *args, **kwargs):
        self.pk = 1  # there is only ever one row
        super().save(*args, **kwargs)
        cache.delete(PLATFORM_CACHE_KEY)


PLATFORM_CACHE_KEY = "platform-settings"
PLATFORM_CACHE_SECONDS = 30  # other server processes see a change within this time


def platform():
    """The platform settings, cached briefly. Unsaved defaults before the first migration."""
    found = cache.get(PLATFORM_CACHE_KEY)
    if found is None:
        try:
            # A savepoint, so a missing table (during the first migrate) doesn't
            # abort the surrounding transaction on PostgreSQL.
            with transaction.atomic():
                found = PlatformSettings.objects.get_or_create(pk=1)[0]
        except (OperationalError, ProgrammingError):  # table not created yet
            return PlatformSettings()
        cache.set(PLATFORM_CACHE_KEY, found, PLATFORM_CACHE_SECONDS)
    return found


# --- Platform staff and two-factor sign-in --------------------------------------------------

class PlatformStaff(models.Model):
    """A non-superuser who can use parts of the Platform Console. Superusers are always owners."""

    class Role(models.TextChoices):
        SUPPORT = "support", "Support"
        FINANCE = "finance", "Finance"

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="platform_staff")
    role = models.CharField(max_length=10, choices=Role.choices)
    created_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL,
                                   related_name="+")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user} ({self.get_role_display()})"


class TwoFactorDevice(models.Model):
    """An authenticator app (TOTP) linked to a user, with one-time backup codes."""

    user = models.OneToOneField(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="two_factor")
    secret_encrypted = models.TextField()
    confirmed_at = models.DateTimeField(null=True, blank=True)
    last_step = models.BigIntegerField(default=0, help_text="Last accepted time step, so a code can't be reused.")
    backup_codes = models.JSONField(default=list, blank=True, help_text="SHA-256 hashes of unused backup codes.")
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"2FA for {self.user}"

    @property
    def is_confirmed(self):
        return self.confirmed_at is not None

    @property
    def secret(self):
        return _fernet().decrypt(self.secret_encrypted.encode()).decode()

    def set_secret(self, secret):
        self.secret_encrypted = _fernet().encrypt(secret.encode()).decode()
