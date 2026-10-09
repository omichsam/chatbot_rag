from django.contrib import admin

from .models import ChatSession, Message


class MessageInline(admin.TabularInline):
    model = Message
    extra = 0
    fields = ["role", "content", "sources", "is_error", "latency_ms", "created_at"]
    readonly_fields = fields


@admin.register(ChatSession)
class ChatSessionAdmin(admin.ModelAdmin):
    list_display = ["__str__", "organization", "user", "channel", "updated_at"]
    list_filter = ["channel", "organization"]
    search_fields = ["title", "user__username"]
    inlines = [MessageInline]


@admin.register(Message)
class MessageAdmin(admin.ModelAdmin):
    list_display = ["__str__", "organization", "role", "is_error", "latency_ms", "created_at"]
    list_filter = ["role", "is_error", "organization"]
    search_fields = ["content"]
