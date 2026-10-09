from django.contrib import admin

from .models import APIKey


@admin.register(APIKey)
class APIKeyAdmin(admin.ModelAdmin):
    list_display = ["name", "prefix", "organization", "is_active", "last_used_at", "created_at"]
    list_filter = ["organization"]
    search_fields = ["name", "prefix"]
    readonly_fields = ["prefix", "key_hash", "last_used_at", "created_at"]

    @admin.display(boolean=True)
    def is_active(self, obj):
        return obj.is_active
