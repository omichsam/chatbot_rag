from django.urls import path

from . import web

app_name = "keys"

urlpatterns = [
    path("", web.key_list, name="list"),
    path("<int:pk>/revoke/", web.revoke, name="revoke"),
]
