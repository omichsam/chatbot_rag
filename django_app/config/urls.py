from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView

from api import web as api_web
from chat.dashboard import dashboard
from tenants import views as tenant_views
from portal import views as portal_views

urlpatterns = [
    path("", RedirectView.as_view(pattern_name="dashboard"), name="home"),
    path("dashboard/", dashboard, name="dashboard"),
    path("admin/", admin.site.urls),
    path("accounts/login/", tenant_views.LoginView.as_view(), name="login"),
    path("accounts/logout/", tenant_views.LogoutView.as_view(), name="logout"),
    path("accounts/signup/", tenant_views.signup, name="signup"),
    path("accounts/2fa/", portal_views.two_factor_verify, name="two_factor_verify"),
    path("accounts/", include("django.contrib.auth.urls")),
    path("org/", include("tenants.urls")),
    path("documents/", include("documents.urls")),
    path("chat/", include("chat.urls")),
    path("keys/", include("api.web_urls")),
    path("developer/", api_web.developer, name="developer"),
    path("platform/", include("portal.urls")),
    path("api/", include("api.urls")),
]
