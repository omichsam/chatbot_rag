from django.contrib import admin

from .models import Document


@admin.register(Document)
class DocumentAdmin(admin.ModelAdmin):
    list_display = ["original_name", "organization", "status", "page_count", "chunk_count", "size", "created_at"]
    list_filter = ["status", "organization"]
    search_fields = ["original_name", "checksum"]
    readonly_fields = ["checksum", "size", "page_count", "chunk_count", "processed_at", "error"]
