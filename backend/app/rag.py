"""RAG index implementation with FAISS and Sentence Transformers.

Supports multilingual semantic embeddings (e.g. paraphrase-multilingual-MiniLM-L12-v2),
strict index compatibility validation, and safe re-indexing without data loss.
"""
import logging
from dataclasses import dataclass
from typing import Any
import faiss
import numpy as np

logger = logging.getLogger(__name__)

_embedder: Any = None
_current_model_name: str | None = None


class IncompatibleIndexError(Exception):
    """Raised when an existing FAISS index is incompatible with the active embedding model."""
    pass


class EmbeddingModelError(Exception):
    """Raised when the configured embedding model cannot be loaded."""
    pass


def get_embedder(model_name: str | None = None) -> Any:
    """Retrieve or lazily initialize the SentenceTransformer embedder.
    
    If model_name is provided and differs from the active one, the model is reloaded.
    """
    global _embedder, _current_model_name
    from .config import EMBEDDING_MODEL

    target_model = model_name or EMBEDDING_MODEL

    if _embedder is not None and _current_model_name == target_model:
        return _embedder

    logger.info(f"Loading embedding model: '{target_model}'...")
    try:
        from sentence_transformers import SentenceTransformer
        loaded_model = SentenceTransformer(target_model)
        _embedder = loaded_model
        _current_model_name = target_model
        return _embedder
    except Exception as exc:
        logger.error(f"Failed to load embedding model '{target_model}': {exc}", exc_info=True)
        raise EmbeddingModelError(
            f"Failed to load embedding model '{target_model}'. Check dependencies and connectivity: {exc}"
        ) from exc


def get_current_embedding_model() -> str:
    """Return the currently configured or active embedding model name."""
    global _current_model_name
    if _current_model_name:
        return _current_model_name
    from .config import EMBEDDING_MODEL
    return EMBEDDING_MODEL


def get_embedding_dimension(embedder: Any = None) -> int:
    """Return vector dimension for the embedder."""
    if embedder is None:
        embedder = get_embedder()
    if hasattr(embedder, "get_embedding_dimension"):
        return int(embedder.get_embedding_dimension())
    if hasattr(embedder, "get_sentence_embedding_dimension"):
        return int(embedder.get_sentence_embedding_dimension())
    dummy = embedder.encode(["test"], normalize_embeddings=True)
    return int(dummy.shape[1])


def reset_embedder() -> None:
    """Reset cached embedder (used in testing and configuration changes)."""
    global _embedder, _current_model_name
    _embedder = None
    _current_model_name = None


def set_embedder(embedder_instance: Any, model_name: str = "custom") -> None:
    """Set custom embedder instance (primarily used for unit test mocking)."""
    global _embedder, _current_model_name
    _embedder = embedder_instance
    _current_model_name = model_name


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> list[str]:
    """Split text into overlapping chunks while collapsing excessive whitespace."""
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
    embedding_model: str
    dimension: int
    pages: list[dict] | None = None


# Storage for in-memory session indexes
DOCUMENTS: dict[str, DocumentIndex] = {}


def index_pages(document_id: str, pages: list[dict], model_name: str | None = None) -> int:
    """Chunk and embed document pages, then store in a FAISS Flat Inner Product index.
    
    Retains page text and metadata for citations and safe model migration.
    """
    if not pages:
        raise ValueError("No pages provided for indexing.")

    chunks: list[dict] = []
    for page_data in pages:
        page_text = page_data.get("text", "")
        for part in split_text(page_text):
            chunks.append({
                "text": part,
                "page": page_data.get("page"),
                "kind": page_data.get("kind", "text"),
            })
    if not chunks:
        raise ValueError("No usable text chunks were created.")

    model = get_embedder(model_name)
    active_model_name = get_current_embedding_model()
    vectors = model.encode([c["text"] for c in chunks], normalize_embeddings=True)
    vectors = np.asarray(vectors, dtype="float32")
    dim = vectors.shape[1]

    index = faiss.IndexFlatIP(dim)
    index.add(vectors)
    DOCUMENTS[document_id] = DocumentIndex(
        index=index,
        chunks=chunks,
        embedding_model=active_model_name,
        dimension=dim,
        pages=pages,
    )
    return len(chunks)


def rebuild_index(document_id: str, model_name: str | None = None) -> int:
    """Rebuild an existing document's FAISS index using the active or target embedding model."""
    stored = DOCUMENTS.get(document_id)
    if stored is None:
        raise KeyError(f"Document index for '{document_id}' not found.")
    if not stored.pages:
        raise ValueError(f"Cannot rebuild index for '{document_id}': original page content was not retained.")
    return index_pages(document_id, stored.pages, model_name=model_name)


def rebuild_all_indexes(model_name: str | None = None) -> dict[str, int]:
    """Rebuild all loaded document indexes with the target embedding model."""
    results = {}
    for doc_id in list(DOCUMENTS.keys()):
        try:
            results[doc_id] = rebuild_index(doc_id, model_name=model_name)
        except Exception as exc:
            logger.error(f"Failed to rebuild index for '{doc_id}': {exc}")
    return results


def retrieve(document_id: str, question: str, top_k: int = 5) -> list[dict]:
    """Retrieve top-K chunks for a question using cosine similarity (via normalized vectors & Inner Product).
    
    Validates model and vector dimension compatibility. Automatically rebuilds the index if
    the active embedding model has changed and original page data is available.
    """
    stored = DOCUMENTS.get(document_id)
    if stored is None:
        raise KeyError("Document index not found. Upload the document again if the backend restarted.")

    embedder = get_embedder()
    active_model_name = get_current_embedding_model()
    active_dim = get_embedding_dimension(embedder)

    # Incompatible index check: Model mismatch or vector dimension mismatch
    if stored.embedding_model != active_model_name or stored.dimension != active_dim:
        logger.warning(
            f"Index mismatch for doc '{document_id}': index was built with "
            f"'{stored.embedding_model}' (dim {stored.dimension}), but active model is "
            f"'{active_model_name}' (dim {active_dim})."
        )
        if stored.pages:
            logger.info(f"Rebuilding index for '{document_id}' with '{active_model_name}'...")
            rebuild_index(document_id, model_name=active_model_name)
            stored = DOCUMENTS[document_id]
        else:
            raise IncompatibleIndexError(
                f"Document index for '{document_id}' was built with '{stored.embedding_model}' "
                f"(dim {stored.dimension}), which is incompatible with the active model '{active_model_name}' "
                f"(dim {active_dim}). Original pages are not retained; please re-upload the document."
            )

    query_vector = embedder.encode([question], normalize_embeddings=True)
    query_vector = np.asarray(query_vector, dtype="float32")

    if query_vector.shape[1] != stored.index.d:
        raise ValueError(
            f"Query vector dimension ({query_vector.shape[1]}) does not match FAISS index dimension ({stored.index.d})."
        )

    search_k = min(top_k, len(stored.chunks))
    scores, indices = stored.index.search(query_vector, search_k)
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx >= 0:
            item = dict(stored.chunks[int(idx)])
            item["score"] = float(score)
            results.append(item)
    return results
