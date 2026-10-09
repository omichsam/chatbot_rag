from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
from django.db import models
from django.utils import timezone
from django.utils.text import slugify


# New organizations start from the platform defaults superadmins set in the portal.
def _platform():
    from portal.models import platform  # portal's models refer back to tenants

    return platform()


def default_top_k():
    return _platform().default_top_k


def default_history_turns():
    return _platform().default_history_turns


def default_gap_threshold():
    return _platform().default_gap_threshold


def default_monthly_limit():
    return _platform().default_monthly_limit


class Plan(models.Model):
    """A package of limits superadmins apply to organizations in one click (0 means unlimited)."""

    name = models.CharField(max_length=60, unique=True)
    description = models.CharField(max_length=200, blank=True)
    monthly_query_limit = models.PositiveIntegerField("Questions per month", default=0)
    max_documents = models.PositiveIntegerField("Documents", default=0)
    max_members = models.PositiveIntegerField("Members", default=0)
    allow_external_ai = models.BooleanField(
        "Cloud AI providers allowed", default=True,
        help_text="Off: organizations on this plan only use local AI providers, then quoting.")
    sort_order = models.PositiveSmallIntegerField(default=0)

    class Meta:
        ordering = ["sort_order", "name"]

    def __str__(self):
        return self.name

    @staticmethod
    def limit_text(value):
        return "Unlimited" if not value else f"{value:,}"


class Organization(models.Model):
    """A tenant. Every document, chat and API key belongs to exactly one."""

    class AnswerLength(models.TextChoices):
        SHORT = "short", "Short"
        BALANCED = "balanced", "Balanced"
        DETAILED = "detailed", "Detailed"

    class Creativity(models.TextChoices):
        PRECISE = "precise", "Precise"
        BALANCED = "balanced", "Balanced"
        FLEXIBLE = "flexible", "Flexible"

    class KnowledgeScope(models.TextChoices):
        STRICT = "strict", "Documents only"
        ASSISTED = "assisted", "Documents first"

    MAX_SUGGESTIONS = 6

    name = models.CharField(max_length=150)
    slug = models.SlugField(max_length=160, unique=True)
    llm_model = models.CharField(
        "Gemini model", max_length=100, blank=True,
        help_text="Leave blank to use the platform default.",
    )
    system_prompt = models.TextField(
        blank=True,
        help_text="Instructions for the assistant. Leave blank for the default. "
        "{org} is replaced with the organization name.",
    )
    monthly_query_limit = models.PositiveIntegerField(
        default=default_monthly_limit, help_text="Questions allowed per calendar month. 0 means unlimited."
    )

    # Chat experience
    assistant_name = models.CharField(
        max_length=60, blank=True, help_text="What the assistant calls itself. Leave blank for \"DocAssist\".")
    welcome_message = models.CharField(
        max_length=300, blank=True, help_text="Shown on a new chat. Leave blank for the default greeting.")
    suggested_questions = models.TextField(
        blank=True, help_text=f"One per line, up to {MAX_SUGGESTIONS}. Shown as buttons on a new chat. "
        "Leave blank for automatic suggestions.")

    # How answers are written
    answer_length = models.CharField(max_length=10, choices=AnswerLength.choices, default=AnswerLength.BALANCED)
    creativity = models.CharField(max_length=10, choices=Creativity.choices, default=Creativity.PRECISE)
    knowledge_scope = models.CharField(max_length=10, choices=KnowledgeScope.choices, default=KnowledgeScope.STRICT)
    response_language = models.CharField(
        max_length=40, blank=True, help_text="For example English or Kiswahili. Leave blank to reply in the language of the question.")
    use_external_ai = models.BooleanField(
        "Use external AI providers", default=True,
        help_text="Off: nothing is sent outside your server. Answers come from a local model if one is set up, "
        "otherwise they quote the documents.")
    show_sources = models.BooleanField(
        default=True, help_text="Show the file and page each answer came from in the web chat. The API always returns them.")

    # Retrieval and quality
    top_k = models.PositiveSmallIntegerField(
        "Passages per answer", default=default_top_k, validators=[MinValueValidator(1), MaxValueValidator(12)],
        help_text="How many document passages the assistant reads before answering (1–12). More can help broad questions but is slower.")
    history_turns = models.PositiveSmallIntegerField(
        "Conversation memory", default=default_history_turns, validators=[MaxValueValidator(10)],
        help_text="Earlier question-and-answer pairs remembered for follow-ups (0–10). 0 treats every question on its own.")
    gap_threshold = models.FloatField(
        "Knowledge-gap threshold", default=default_gap_threshold,
        validators=[MinValueValidator(0.05), MaxValueValidator(0.9)],
        help_text="Answers whose best passage matches below this score (0.05–0.9) are flagged as knowledge gaps.")
    cache_answers = models.BooleanField(
        default=True, help_text="Reuse a recent answer when the same question is asked again. Faster and saves AI quota.")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # Platform lifecycle, managed by superadmins
    plan = models.ForeignKey(Plan, null=True, blank=True, on_delete=models.SET_NULL, related_name="organizations",
                             help_text="Blank: custom limits set on the organization itself.")
    suspended_reason = models.CharField(max_length=300, blank=True)
    suspended_at = models.DateTimeField(null=True, blank=True)
    deletion_scheduled_at = models.DateTimeField(
        null=True, blank=True, help_text="When set, the organization is offline and is deleted for good at this time.")

    class Meta:
        ordering = ["name"]

    def __str__(self):
        return self.name

    def save(self, *args, **kwargs):
        if not self.slug:
            self.slug = self._unique_slug()
        super().save(*args, **kwargs)

    def _unique_slug(self):
        base = slugify(self.name)[:150] or "org"
        slug, suffix = base, 2
        while Organization.objects.filter(slug=slug).exists():
            slug, suffix = f"{base}-{suffix}", suffix + 1
        return slug

    @property
    def effective_model(self):
        return self.llm_model or _platform().default_model

    @property
    def display_assistant_name(self):
        return self.assistant_name or "DocAssist"

    @property
    def suggestions(self):
        lines = (line.strip() for line in self.suggested_questions.splitlines())
        return [line for line in lines if line][: self.MAX_SUGGESTIONS]

    @property
    def max_output_tokens(self):
        base = settings.RAG_MAX_OUTPUT_TOKENS
        return {self.AnswerLength.SHORT: base // 2, self.AnswerLength.DETAILED: base * 2}.get(self.answer_length, base)

    @property
    def temperature(self):
        return {self.Creativity.BALANCED: 0.5, self.Creativity.FLEXIBLE: 0.8}.get(self.creativity, settings.RAG_TEMPERATURE)

    def answer_fingerprint(self):
        """Everything that changes how an answer is written, for the answer cache key."""
        return "|".join(map(str, (
            self.effective_model, self.system_prompt, self.assistant_name, self.answer_length, self.creativity,
            self.knowledge_scope, self.response_language, self.show_sources, self.top_k, self.use_external_ai,
        )))

    def queries_this_month(self):
        from chat.models import Message

        month_start = timezone.localtime().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        return Message.objects.filter(
            organization=self, role=Message.Role.USER, created_at__gte=month_start
        ).count()

    @property
    def external_ai_allowed(self):
        """The organization's own switch, narrowed by its plan."""
        return self.use_external_ai and (self.plan is None or self.plan.allow_external_ai)

    def document_limit_reached(self, adding=1):
        limit = self.plan.max_documents if self.plan else 0
        return bool(limit) and self.documents.count() + adding > limit

    def member_limit_reached(self, adding=1):
        limit = self.plan.max_members if self.plan else 0
        return bool(limit) and self.memberships.count() + adding > limit

    def quota_exceeded(self):
        return bool(self.monthly_query_limit) and self.queries_this_month() >= self.monthly_query_limit


class Membership(models.Model):
    class Role(models.TextChoices):
        OWNER = "owner", "Owner"
        ADMIN = "admin", "Admin"
        MEMBER = "member", "Member"

    RANK = {Role.MEMBER: 0, Role.ADMIN: 1, Role.OWNER: 2}

    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="memberships")
    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="memberships")
    role = models.CharField(max_length=10, choices=Role.choices, default=Role.MEMBER)
    # This person's own sidebar arrangement: {"order": [...], "hidden": [...], "pinned": [...]}
    menu_prefs = models.JSONField(default=dict, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["organization__name", "user__username"]
        constraints = [
            models.UniqueConstraint(fields=["user", "organization"], name="unique_membership"),
        ]

    def __str__(self):
        return f"{self.user} @ {self.organization} ({self.role})"

    def has_role(self, minimum):
        return self.RANK[self.role] >= self.RANK[minimum]

    @property
    def is_admin(self):
        return self.has_role(self.Role.ADMIN)


class AccessRequest(models.Model):
    """A member asking the organization's admins for admin access."""

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        APPROVED = "approved", "Approved"
        DISMISSED = "dismissed", "Dismissed"

    organization = models.ForeignKey(Organization, on_delete=models.CASCADE, related_name="access_requests")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="access_requests")
    message = models.CharField(max_length=300, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    decided_by = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL, related_name="+")
    decided_at = models.DateTimeField(null=True, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"{self.user} → admin of {self.organization} ({self.status})"
