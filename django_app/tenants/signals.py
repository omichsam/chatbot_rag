import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from rag import vectorstore

from .models import Organization

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=Organization)
def drop_vector_collection(sender, instance, **kwargs):
    try:
        vectorstore.delete_organization(instance.pk)
    except Exception:
        logger.exception("Could not drop vector collection for organization %s", instance.pk)
