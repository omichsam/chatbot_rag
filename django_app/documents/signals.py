import logging

from django.db.models.signals import post_delete
from django.dispatch import receiver

from rag import vectorstore

from .models import Document

logger = logging.getLogger(__name__)


@receiver(post_delete, sender=Document)
def remove_document_data(sender, instance, **kwargs):
    try:
        vectorstore.delete_document(instance.organization_id, instance.pk)
    except Exception:
        logger.exception("Could not remove vectors for document %s", instance.pk)
    if instance.file:
        instance.file.delete(save=False)
