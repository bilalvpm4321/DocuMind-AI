"""A small in-memory RAG index. It is reset whenever the backend restarts."""
from dataclasses import dataclass
from typing import Any
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import EMBEDDING_MODEL

_embedder: SentenceTransformer | None = None


def get_embedder() -> SentenceTransformer:
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(EMBEDDING_MODEL)
    return _embedder


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    text = " ".join(text.split())
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start = end - overlap
    return chunks


@dataclass
class DocumentIndex:
    index: Any
    chunks: list[dict]


# Prototype storage. Use a persistent vector DB and user-scoped storage for production.
DOCUMENTS: dict[str, DocumentIndex] = {}


def index_pages(document_id: str, pages: list[dict]) -> int:
    chunks: list[dict] = []
    for page_data in pages:
        for part in split_text(page_data["text"]):
            chunks.append({
                "text": part,
                "page": page_data.get("page"),
                "kind": page_data.get("kind", "text"),
            })
    if not chunks:
        raise ValueError("No usable text chunks were created.")

    model = get_embedder()
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")
    index = faiss.IndexFlatIP(vectors.shape[1])
    index.add(vectors)
    DOCUMENTS[document_id] = DocumentIndex(index=index, chunks=chunks)
    return len(chunks)


def retrieve(document_id: str, question: str, top_k: int = 5) -> list[dict]:
    stored = DOCUMENTS.get(document_id)
    if stored is None:
        raise KeyError("Document index not found. Upload the document again if the backend restarted.")
    query_vector = get_embedder().encode([question], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")
    scores, indices = stored.index.search(query_vector, min(top_k, len(stored.chunks)))
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            item = dict(stored.chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results
