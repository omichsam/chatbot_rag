"""Three starting plans. Existing organizations keep custom limits (no plan) until a superadmin picks one."""
from django.db import migrations

PLANS = [
    {"name": "Free", "description": "To try DocAssist", "monthly_query_limit": 500, "max_documents": 10,
     "max_members": 5, "allow_external_ai": True, "sort_order": 1},
    {"name": "Standard", "description": "For a team or department", "monthly_query_limit": 5000,
     "max_documents": 200, "max_members": 50, "allow_external_ai": True, "sort_order": 2},
    {"name": "Enterprise", "description": "No limits", "monthly_query_limit": 0, "max_documents": 0,
     "max_members": 0, "allow_external_ai": True, "sort_order": 3},
]


def add_plans(apps, schema_editor):
    Plan = apps.get_model("tenants", "Plan")
    for values in PLANS:
        Plan.objects.get_or_create(name=values["name"], defaults=values)


class Migration(migrations.Migration):
    dependencies = [("tenants", "0007_platform_console")]
    operations = [migrations.RunPython(add_plans, migrations.RunPython.noop)]
