from pathlib import Path

from django import forms

from portal.models import platform
from rag.loaders import SUPPORTED_EXTENSIONS


class MultipleFileInput(forms.ClearableFileInput):
    allow_multiple_selected = True


class MultipleFileField(forms.FileField):
    def __init__(self, *args, **kwargs):
        kwargs.setdefault("widget", MultipleFileInput(attrs={"accept": ",".join(SUPPORTED_EXTENSIONS)}))
        super().__init__(*args, **kwargs)

    def clean(self, data, initial=None):
        files = data if isinstance(data, (list, tuple)) else [data]
        return [super(MultipleFileField, self).clean(f, initial) for f in files]


class UploadForm(forms.Form):
    files = MultipleFileField(label="Documents", help_text="PDF, TXT or MD. You can select several.")

    def clean_files(self):
        files = self.cleaned_data["files"]
        max_mb = platform().max_upload_mb
        max_bytes = max_mb * 1024 * 1024
        for f in files:
            if Path(f.name).suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise forms.ValidationError(f"{f.name}: only PDF, TXT and MD files are supported.")
            if f.size > max_bytes:
                raise forms.ValidationError(f"{f.name} is larger than {max_mb} MB.")
        return files
