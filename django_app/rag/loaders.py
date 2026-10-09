"""Turn an uploaded file into (page_number, text) pairs."""
from pathlib import Path

from pypdf import PdfReader

SUPPORTED_EXTENSIONS = (".pdf", ".txt", ".md")
# latin-1 decodes any byte sequence, so it must stay last.
TEXT_ENCODINGS = ("utf-8", "cp1252", "latin-1")


class UnsupportedFileType(ValueError):
    pass


def extract_pages(path, filename):
    extension = Path(filename).suffix.lower()
    if extension == ".pdf":
        reader = PdfReader(path)
        return [(number, page.extract_text() or "") for number, page in enumerate(reader.pages, start=1)]
    if extension in (".txt", ".md"):
        raw = Path(path).read_bytes()
        for encoding in TEXT_ENCODINGS:
            try:
                return [(1, raw.decode(encoding))]
            except UnicodeDecodeError:
                continue
    raise UnsupportedFileType(f"Unsupported file type: {extension or 'no extension'}")
