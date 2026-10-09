from django.urls import path

from . import views

app_name = "tenants"

urlpatterns = [
    path("new/", views.create_organization, name="create"),
    path("switch/", views.switch_organization, name="switch"),
    path("settings/", views.organization_settings, name="settings"),
    path("members/", views.members, name="members"),
    path("members/<int:membership_id>/", views.update_member, name="update_member"),
    path("menu/", views.save_menu_prefs, name="menu_prefs"),
    path("access/request/", views.request_access, name="request_access"),
    path("access/<int:pk>/decide/", views.decide_access, name="decide_access"),
]
