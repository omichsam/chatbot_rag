from django.urls import path

from . import explorer, insights, navigation, views

app_name = "chat"

urlpatterns = [
    path("", views.index, name="index"),
    path("s/<uuid:session_id>/", views.index, name="session"),
    path("s/<uuid:session_id>/delete/", views.delete_session, name="delete_session"),
    path("send/", views.send, name="send"),
    path("stream/", views.send_stream, name="stream"),
    path("analytics/", views.analytics, name="analytics"),
    path("messages/<int:message_id>/rate/", insights.rate, name="rate"),
    path("messages/<int:message_id>/save/", insights.toggle_save, name="save"),
    path("saved/", insights.saved_answers, name="saved"),
    path("feedback/", insights.feedback, name="feedback"),
    path("feedback/<int:message_id>/reviewed/", insights.mark_reviewed, name="mark_reviewed"),
    path("gaps/", insights.knowledge_gaps, name="gaps"),
    path("gaps/resolve/", insights.resolve_gap, name="resolve_gap"),
    path("help/", insights.help_page, name="help"),
    path("history/", navigation.history, name="history"),
    path("search/", navigation.search, name="search"),
    path("questions/", explorer.questions, name="questions"),
    path("questions/topics/add/", explorer.topic_add, name="topic_add"),
    path("questions/topics/<int:pk>/delete/", explorer.topic_delete, name="topic_delete"),
    path("questions/topics/resort/", explorer.topic_resort, name="topic_resort"),
]
