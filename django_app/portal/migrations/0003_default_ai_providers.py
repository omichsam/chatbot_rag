"""Start the provider list with what the app used before providers were configurable:
Google Gemini with the key from .env (and a local Ollama model if RAG_LOCAL_MODEL was set)."""
import os

from django.db import migrations


def add_defaults(apps, schema_editor):
    AIProvider = apps.get_model("portal", "AIProvider")
    if AIProvider.objects.exists():
        return
    # Blank model and key: the platform default model and GOOGLE_API_KEY from .env.
    AIProvider.objects.create(name="Google Gemini", kind="gemini", priority=10)
    local_model = os.getenv("RAG_LOCAL_MODEL", "")
    if local_model:
        AIProvider.objects.create(
            name="Local model", kind="ollama", model=local_model, priority=20,
            base_url=os.getenv("RAG_LOCAL_URL", "http://localhost:11434").rstrip("/") + "/v1")


class Migration(migrations.Migration):
    dependencies = [("portal", "0002_ai_providers")]
    operations = [migrations.RunPython(add_defaults, migrations.RunPython.noop)]
