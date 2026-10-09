from django.contrib import messages
from django.db import transaction
from django.db.models import Sum
from django.shortcuts import get_object_or_404, redirect, render
from django.utils import timezone
from django.views.decorators.http import require_POST

from portal.models import platform
from tenants.decorators import org_admin_required, org_member_required

from . import services
from .forms import UploadForm
from .models import Document


@org_member_required
def document_list(request):
    documents = Document.objects.filter(organization=request.organization).select_related("uploaded_by")
    return render(request, "documents/list.html", {
        "documents": documents,
        "form": UploadForm() if request.membership.is_admin else None,
        "any_busy": any(d.is_busy for d in documents),
        "total_size": documents.aggregate(total=Sum("size"))["total"] or 0,
        "paused_count": sum(d.is_paused for d in documents),
        "max_mb": platform().max_upload_mb,
    })


@org_admin_required
@require_POST
def upload(request):
    form = UploadForm(request.POST, request.FILES)
    if not form.is_valid():
        for error in form.errors.get("files", []):
            messages.error(request, error)
        return redirect("documents:list")

    added = 0
    for uploaded in form.cleaned_data["files"]:
        if request.organization.document_limit_reached():
            messages.error(request, f"Your plan allows {request.organization.plan.max_documents} documents. "
                                    f"{uploaded.name} and any later files were not uploaded.")
            break
        checksum = services.compute_checksum(uploaded)
        duplicate = (
            Document.objects.filter(organization=request.organization, checksum=checksum)
            .exclude(status=Document.Status.FAILED)
            .first()
        )
        if duplicate and duplicate.is_paused:
            messages.warning(request, f"{uploaded.name} is already uploaded as {duplicate.original_name}, which is paused. "
                                      "Resume it instead of uploading again.")
            continue
        if duplicate:
            messages.warning(request, f"{uploaded.name} is already uploaded as {duplicate.original_name}. Skipped.")
            continue
        document = Document.objects.create(
            organization=request.organization,
            uploaded_by=request.user,
            file=uploaded,
            original_name=uploaded.name,
            size=uploaded.size,
            checksum=checksum,
        )
        transaction.on_commit(lambda pk=document.pk: services.enqueue_ingest(pk))
        added += 1

    if added:
        messages.success(request, f"Uploaded {added} document(s). Indexing runs in the background.")
    return redirect("documents:list")


@org_admin_required
@require_POST
def delete(request, pk):
    document = get_object_or_404(Document, pk=pk, organization=request.organization)
    name = document.original_name
    services.delete_document(document)
    messages.success(request, f"Deleted {name}.")
    return redirect("documents:list")


@org_admin_required
@require_POST
def toggle_pause(request, pk):
    """Pause: keep the file and vectors but leave the document out of answers. Resume undoes it instantly."""
    document = get_object_or_404(Document, pk=pk, organization=request.organization)
    pausing = not document.is_paused
    Document.objects.filter(pk=document.pk).update(is_paused=pausing, paused_at=timezone.now() if pausing else None)
    if pausing:
        messages.success(request, f"Paused {document.original_name}. Answers no longer use it.")
    else:
        messages.success(request, f"Resumed {document.original_name}. Answers use it again.")
    return redirect("documents:list")


@org_admin_required
@require_POST
def reprocess(request, pk):
    document = get_object_or_404(Document, pk=pk, organization=request.organization)
    Document.objects.filter(pk=document.pk).update(status=Document.Status.PENDING, error="")
    transaction.on_commit(lambda: services.enqueue_ingest(document.pk))
    messages.info(request, f"Re-indexing {document.original_name}.")
    return redirect("documents:list")
