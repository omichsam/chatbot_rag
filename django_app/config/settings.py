"""Settings for the multi-tenant RAG platform.

Everything environment-specific is read from environment variables (or a .env
file in this folder). The legacy project's .env one level up is also loaded so
GOOGLE_API_KEY keeps working without copying it.
"""
import os
from pathlib import Path

from django.core.exceptions import ImproperlyConfigured
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent.parent

# The release version (semantic versioning; see docs/CHANGELOG.md). One source: the VERSION file.
APP_VERSION = (BASE_DIR / "VERSION").read_text(encoding="utf-8").strip() if (BASE_DIR / "VERSION").exists() else "0.0.0"

load_dotenv(BASE_DIR / ".env")
load_dotenv(BASE_DIR.parent / ".env")


def env_bool(name, default=False):
    return os.getenv(name, str(default)).strip().lower() in ("1", "true", "yes", "on")


def env_list(name, default=""):
    return [item.strip() for item in os.getenv(name, default).split(",") if item.strip()]


# --- Core -------------------------------------------------------------------

DEV_SECRET_KEY = "dev-insecure-change-me"
SECRET_KEY = os.getenv("DJANGO_SECRET_KEY", DEV_SECRET_KEY)
DEBUG = env_bool("DJANGO_DEBUG", True)
ALLOWED_HOSTS = env_list("DJANGO_ALLOWED_HOSTS", "localhost,127.0.0.1")
CSRF_TRUSTED_ORIGINS = env_list("DJANGO_CSRF_TRUSTED_ORIGINS")

if not DEBUG and SECRET_KEY == DEV_SECRET_KEY:
    raise ImproperlyConfigured("Set DJANGO_SECRET_KEY when DJANGO_DEBUG is off.")

# Anyone can create an account + their own organization from the login page.
ALLOW_SIGNUP = env_bool("ALLOW_SIGNUP", True)

INSTALLED_APPS = [
    "django.contrib.admin",
    "django.contrib.auth",
    "django.contrib.contenttypes",
    "django.contrib.sessions",
    "django.contrib.messages",
    "django.contrib.staticfiles",
    "django.contrib.humanize",
    "tenants",
    "documents",
    "chat",
    "api",
    "portal",
]

MIDDLEWARE = [
    "django.middleware.security.SecurityMiddleware",
    # Serves collected static files (the admin's CSS and JS) when DEBUG is off, e.g. in Docker.
    "whitenoise.middleware.WhiteNoiseMiddleware",
    "django.contrib.sessions.middleware.SessionMiddleware",
    "django.middleware.common.CommonMiddleware",
    "django.middleware.csrf.CsrfViewMiddleware",
    "django.contrib.auth.middleware.AuthenticationMiddleware",
    "django.contrib.messages.middleware.MessageMiddleware",
    "portal.middleware.MaintenanceModeMiddleware",
    "tenants.middleware.CurrentOrganizationMiddleware",
    "django.middleware.clickjacking.XFrameOptionsMiddleware",
]

ROOT_URLCONF = "config.urls"

TEMPLATES = [
    {
        "BACKEND": "django.template.backends.django.DjangoTemplates",
        "DIRS": [BASE_DIR / "templates"],
        "APP_DIRS": True,
        "OPTIONS": {
            "context_processors": [
                "django.template.context_processors.request",
                "django.contrib.auth.context_processors.auth",
                "django.contrib.messages.context_processors.messages",
                "tenants.context_processors.organization",
            ],
        },
    },
]

WSGI_APPLICATION = "config.wsgi.application"

# --- Database ---------------------------------------------------------------
# SQLite by default; set POSTGRES_DB (and friends) to use PostgreSQL.

if os.getenv("POSTGRES_DB"):
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.postgresql",
            "NAME": os.environ["POSTGRES_DB"],
            "USER": os.getenv("POSTGRES_USER", "postgres"),
            "PASSWORD": os.getenv("POSTGRES_PASSWORD", ""),
            "HOST": os.getenv("POSTGRES_HOST", "localhost"),
            "PORT": os.getenv("POSTGRES_PORT", "5432"),
        }
    }
else:
    DATABASES = {
        "default": {
            "ENGINE": "django.db.backends.sqlite3",
            "NAME": BASE_DIR / "db.sqlite3",
            "OPTIONS": {"timeout": 20},
        }
    }

# --- Auth -------------------------------------------------------------------

PASSWORD_HASHERS = [
    "django.contrib.auth.hashers.PBKDF2PasswordHasher",
    "django.contrib.auth.hashers.PBKDF2SHA1PasswordHasher",
    "django.contrib.auth.hashers.ScryptPasswordHasher",
    # Accepts unsalted SHA-256 hashes imported from the Streamlit app's
    # users.json and upgrades them to PBKDF2 on the user's next login.
    "tenants.hashers.LegacySHA256PasswordHasher",
]

AUTH_PASSWORD_VALIDATORS = [
    {"NAME": "django.contrib.auth.password_validation.UserAttributeSimilarityValidator"},
    {"NAME": "django.contrib.auth.password_validation.MinimumLengthValidator"},
    {"NAME": "django.contrib.auth.password_validation.CommonPasswordValidator"},
    {"NAME": "django.contrib.auth.password_validation.NumericPasswordValidator"},
]

LOGIN_URL = "login"
LOGIN_REDIRECT_URL = "dashboard"
LOGOUT_REDIRECT_URL = "login"

# --- i18n / static / media --------------------------------------------------

LANGUAGE_CODE = "en-us"
TIME_ZONE = os.getenv("DJANGO_TIME_ZONE", "UTC")
USE_I18N = True
USE_TZ = True

STATIC_URL = "static/"
STATIC_ROOT = BASE_DIR / "staticfiles"

# Uploaded documents. Deliberately NOT served over HTTP: they are private to
# their organization.
MEDIA_ROOT = Path(os.getenv("MEDIA_ROOT", BASE_DIR / "media"))

DEFAULT_AUTO_FIELD = "django.db.models.BigAutoField"

if not DEBUG:
    SESSION_COOKIE_SECURE = env_bool("SESSION_COOKIE_SECURE", True)
    CSRF_COOKIE_SECURE = env_bool("CSRF_COOKIE_SECURE", True)

# --- RAG --------------------------------------------------------------------

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")

# "sentence-transformers" (real, local model) or "fake" (hash-based, for tests)
RAG_EMBEDDING_BACKEND = os.getenv("RAG_EMBEDDING_BACKEND", "sentence-transformers")
RAG_EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
# "gemini" or "echo" (returns the question back, for tests)
RAG_LLM_BACKEND = os.getenv("RAG_LLM_BACKEND", "gemini")
# An alias, so a model retirement on Google's side doesn't break answers.
# Pin a specific model per organization (Settings page) or here if you need stability.
RAG_DEFAULT_MODEL = os.getenv("RAG_DEFAULT_MODEL", "gemini-flash-latest")

# Where passage vectors live: "pgvector" (inside PostgreSQL; the Docker setup) or
# "chroma" (a folder at RAG_VECTOR_DIR; works with any database).
RAG_VECTOR_BACKEND = os.getenv("RAG_VECTOR_BACKEND", "chroma")
RAG_VECTOR_DIR = Path(os.getenv("RAG_VECTOR_DIR", BASE_DIR / "vector_store"))
RAG_CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", 1000))
RAG_CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", 200))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", 5))
# How many previous question/answer pairs are sent to the model as context.
RAG_HISTORY_TURNS = int(os.getenv("RAG_HISTORY_TURNS", 3))
# Speed: answer length cap, focus, repeat-question cache and startup warm-up.
RAG_MAX_OUTPUT_TOKENS = int(os.getenv("RAG_MAX_OUTPUT_TOKENS", 600))
RAG_TEMPERATURE = float(os.getenv("RAG_TEMPERATURE", 0.2))

# The AI providers that write answers (Gemini, OpenAI, Claude, Groq, ...) are
# managed by superadmins in the portal, AI providers page. When every provider
# fails, answers quote sentences straight from the documents unless this is off.
RAG_QUOTED_FALLBACK = env_bool("RAG_QUOTED_FALLBACK", True)

# Shown in the Platform Console header, so operators always know which system they're on.
PLATFORM_ENVIRONMENT = os.getenv("PLATFORM_ENVIRONMENT", "Development" if DEBUG else "Production")
RAG_ANSWER_CACHE_SECONDS = int(os.getenv("RAG_ANSWER_CACHE_SECONDS", 24 * 3600))
RAG_WARMUP = env_bool("RAG_WARMUP", True)
RAG_MAX_UPLOAD_MB = int(os.getenv("RAG_MAX_UPLOAD_MB", 25))
RAG_MAX_QUESTION_CHARS = 4000
# Answers whose best passage scores below this (cosine similarity, 0–1) are
# listed under Knowledge gaps: the documents likely don't cover the question.
RAG_GAP_SCORE = float(os.getenv("RAG_GAP_SCORE", 0.35))

# "thread" processes uploads in a background thread pool; "sync" does it inside
# the request (used by tests and management commands).
RAG_INGEST_MODE = os.getenv("RAG_INGEST_MODE", "thread")
RAG_INGEST_WORKERS = int(os.getenv("RAG_INGEST_WORKERS", 2))

LOGGING = {
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {"simple": {"format": "%(asctime)s %(levelname)s %(name)s: %(message)s"}},
    "handlers": {"console": {"class": "logging.StreamHandler", "formatter": "simple"}},
    "root": {"handlers": ["console"], "level": os.getenv("LOG_LEVEL", "INFO")},
}
