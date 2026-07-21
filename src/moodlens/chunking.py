from __future__ import annotations

from collections.abc import Iterable

from .models import Chunk, Document


def chunk_document(document: Document, *, words: int, overlap: int) -> list[Chunk]:
    if words < 1 or overlap < 0 or overlap >= words:
        raise ValueError("chunk size must be positive and overlap smaller than chunk size")
    tokens = document.text.split()
    if not tokens:
        return []
    step = words - overlap
    chunks: list[Chunk] = []
    for position, start in enumerate(range(0, len(tokens), step)):
        text = " ".join(tokens[start : start + words])
        chunks.append(
            Chunk(
                chunk_id=f"{document.document_id}::c{position:03d}",
                document_id=document.document_id,
                text=text,
                position=position,
                metadata=document.metadata(),
            )
        )
        if start + words >= len(tokens):
            break
    return chunks


def chunk_documents(documents: Iterable[Document], *, words: int, overlap: int) -> list[Chunk]:
    return [
        chunk
        for document in documents
        for chunk in chunk_document(document, words=words, overlap=overlap)
    ]

