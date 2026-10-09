from django.urls import path

from . import views

app_name = "api"

urlpatterns = [
    path("v1/chat", views.chat, name="chat"),
    path("v1/health", views.health, name="health"),
    path("v1/validate-key", views.validate_key, name="validate_key"),
    path("v1/documents", views.documents, name="documents"),
    # Aliases with the same paths as shit_api.py, for existing clients.
    path("chat", views.chat, name="legacy_chat"),
    path("health", views.health, name="legacy_health"),
    path("validate-key", views.validate_key, name="legacy_validate_key"),
]
