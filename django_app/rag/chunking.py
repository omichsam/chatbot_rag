from dataclasses import dataclass

from django.conf import settings
from langchain_text_splitters import RecursiveCharacterTextSplitter


@dataclass
class Chunk:
    text: str
    page: int


def split_pages(pages):
    """Split (page, text) pairs into overlapping chunks that remember their page."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.RAG_CHUNK_SIZE,
        chunk_overlap=settings.RAG_CHUNK_OVERLAP,
    )
    chunks = []
    for page, text in pages:
        text = text.strip()
        if not text:
            continue
        chunks.extend(Chunk(piece, page) for piece in splitter.split_text(text) if piece.strip())
    return chunks
