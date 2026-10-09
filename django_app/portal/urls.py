from django.urls import path

from . import views

app_name = "portal"

urlpatterns = [
    path("", views.overview, name="overview"),
    path("organizations/", views.organizations, name="organizations"),
    path("organizations/<int:pk>/", views.organization_detail, name="organization"),
    path("organizations/<int:pk>/action/", views.organization_action, name="organization_action"),
    path("organizations/<int:pk>/export/", views.organization_export, name="organization_export"),
    path("organizations/<int:pk>/support/", views.support_start, name="support_start"),
    path("support/exit/", views.support_exit, name="support_exit"),
    path("plans/", views.plans, name="plans"),
    path("usage.csv", views.usage_export, name="usage_export"),
    path("security/", views.security, name="security"),
    path("security/action/", views.security_action, name="security_action"),
    path("security/two-factor/", views.two_factor, name="two_factor"),
    path("search/", views.search, name="search"),
    path("users/", views.users, name="users"),
    path("users/<int:pk>/action/", views.user_action, name="user_action"),
    path("system/", views.system, name="system"),
    path("system/test/", views.system_test, name="system_test"),
    path("settings/", views.platform_settings, name="settings"),
    path("ai/", views.ai_providers, name="ai_providers"),
    path("ai/add/", views.ai_provider_edit, name="ai_provider_add"),
    path("ai/<int:pk>/", views.ai_provider_edit, name="ai_provider"),
    path("ai/<int:pk>/action/", views.ai_provider_action, name="ai_provider_action"),
    path("audit/", views.audit, name="audit"),
    path("guide/", views.guide, name="guide"),
]
