from django.apps import AppConfig


class ChatConfig(AppConfig):
    name = "chat"

    def ready(self):
        from rag.warmup import start_in_background

        start_in_background()
