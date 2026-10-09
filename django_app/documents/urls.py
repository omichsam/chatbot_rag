from django.urls import path

from . import views

app_name = "documents"

urlpatterns = [
    path("", views.document_list, name="list"),
    path("upload/", views.upload, name="upload"),
    path("<int:pk>/delete/", views.delete, name="delete"),
    path("<int:pk>/reprocess/", views.reprocess, name="reprocess"),
    path("<int:pk>/pause/", views.toggle_pause, name="pause"),
]
