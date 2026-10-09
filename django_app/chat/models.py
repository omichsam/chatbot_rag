import uuid

from django.conf import settings
from django.db import models


class ChatSession(models.Model):
    class Channel(models.TextChoices):
        WEB = "web", "Web"
        API = "api", "API"

    public_id = models.UUIDField(default=uuid.uuid4, unique=True, editable=False)
    organization = models.ForeignKey("tenants.Organization", on_delete=models.CASCADE, related_name="chat_sessions")
    user = models.ForeignKey(settings.AUTH_USER_MODEL, null=True, blank=True, on_delete=models.SET_NULL)
    api_key = models.ForeignKey("api.APIKey", null=True, blank=True, on_delete=models.SET_NULL)
    channel = models.CharField(max_length=3, choices=Channel.choices, default=Channel.WEB)
    # API sessions started from the in-app Developer playground, so test
    # traffic can be told apart from real integrations.
    is_test = models.BooleanField(default=False)
    title = models.CharField(max_length=200, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)
    updated_at = models.DateTimeField(auto_now=True)

    class Meta:
        ordering = ["-updated_at"]

    def __str__(self):
        return self.title or str(self.public_id)


class Message(models.Model):
    class Role(models.TextChoices):
        USER = "user", "User"
        ASSISTANT = "assistant", "Assistant"

    session = models.ForeignKey(ChatSession, on_delete=models.CASCADE, related_name="messages")
    # Denormalized from the session so analytics and quotas are one indexed query.
    organization = models.ForeignKey("tenants.Organization", on_delete=models.CASCADE, related_name="messages")
    role = models.CharField(max_length=10, choices=Role.choices)
    content = models.TextField()
    sources = models.JSONField(default=list, blank=True)
    latency_ms = models.PositiveIntegerField(null=True, blank=True)
    is_error = models.BooleanField(default=False)
    # Why an answer failed (e.g. the AI provider's error), for operators.
    error_detail = models.CharField(max_length=500, blank=True)
    # Assistant messages: which writer produced the answer (see rag/pipeline.py).
    # A backup writer with error_detail set means Gemini failed and the backup stepped in.
    answered_by = models.CharField(max_length=10, blank=True)
    created_at = models.DateTimeField(auto_now_add=True)

    # User messages only: how the question is categorized (see chat/categorize.py).
    intent = models.CharField(max_length=20, blank=True)
    topic = models.ForeignKey("Topic", null=True, blank=True, on_delete=models.SET_NULL, related_name="questions")

    # Assistant messages only ------------------------------------------------
    # Best similarity score among the retrieved passages (0–1). Low or missing
    # means the documents probably don't cover the question: a knowledge gap.
    retrieval_score = models.FloatField(null=True, blank=True)
    gap_resolved = models.BooleanField(default=False)
    # 👍 / 👎 from the person who asked.
    rating = models.SmallIntegerField(null=True, blank=True, choices=[(1, "Helpful"), (-1, "Not helpful")])
    feedback_comment = models.TextField(blank=True)
    rated_at = models.DateTimeField(null=True, blank=True)
    feedback_reviewed = models.BooleanField(default=False)

    class Meta:
        ordering = ["created_at", "id"]
        indexes = [models.Index(fields=["organization", "role", "created_at"])]

    def __str__(self):
        return f"{self.role}: {self.content[:60]}"

    def question(self):
        """For an answer, the user message it replied to."""
        return (
            Message.objects.filter(session_id=self.session_id, role=self.Role.USER, id__lt=self.id)
            .order_by("-id").first()
        )


class Topic(models.Model):
    """An admin-defined subject that questions are sorted into, e.g. "Funding"."""

    organization = models.ForeignKey("tenants.Organization", on_delete=models.CASCADE, related_name="topics")
    name = models.CharField(max_length=60)
    description = models.CharField(
        max_length=300, blank=True,
        help_text="Words and phrases people use for this subject. Helps questions land in the right topic.",
    )
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["name"]
        constraints = [models.UniqueConstraint(fields=["organization", "name"], name="unique_topic_name")]

    def __str__(self):
        return self.name

    @property
    def match_text(self):
        return f"{self.name}. {self.description}".strip()


class SavedAnswer(models.Model):
    user = models.ForeignKey(settings.AUTH_USER_MODEL, on_delete=models.CASCADE, related_name="saved_answers")
    message = models.ForeignKey(Message, on_delete=models.CASCADE, related_name="saves")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [models.UniqueConstraint(fields=["user", "message"], name="unique_saved_answer")]
