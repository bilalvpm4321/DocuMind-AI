"""Multilingual RAG retrieval and FAISS index lifecycle test suite.

Validates:
1. English query -> English document retrieval.
2. Malayalam query -> Malayalam document retrieval.
3. Malayalam query -> English document (cross-lingual retrieval).
4. English query -> Malayalam document (cross-lingual retrieval).
5. Top-5 chunk retrieval and correct evidence ranking.
6. Page metadata and source citation preservation.
7. FAISS index lifecycle, model change detection, and safe rebuilding.
8. Robust error handling (empty docs, unknown IDs, model loading errors).
"""
import pytest
import numpy as np
from app.rag import (
    DOCUMENTS,
    DocumentIndex,
    get_embedder,
    get_current_embedding_model,
    get_embedding_dimension,
    index_pages,
    retrieve,
    rebuild_index,
    rebuild_all_indexes,
    reset_embedder,
    set_embedder,
    IncompatibleIndexError,
    EmbeddingModelError,
)
from app.config import EMBEDDING_MODEL


@pytest.fixture(scope="module")
def multilingual_model():
    """Ensure the configured multilingual model is loaded once for this test module."""
    model = get_embedder(EMBEDDING_MODEL)
    return model


@pytest.fixture
def english_corpus():
    return [
        {
            "page": 1,
            "kind": "page",
            "text": (
                "Bilal Ahamed is pursuing a Master of Technology (M.Tech) in Computer Science "
                "and Engineering with specialization in Artificial Intelligence and Data Science "
                "at Cochin University of Science and Technology (CUSAT), Kochi, Kerala."
            ),
        },
        {
            "page": 2,
            "kind": "page",
            "text": (
                "DocuMind AI architecture utilizes FastAPI for high-performance backend routing, "
                "React with TypeScript for the user interface, and FAISS for vector indexing."
            ),
        },
        {
            "page": 3,
            "kind": "page",
            "text": (
                "The system deployment requires Docker containerization, PostgreSQL for persistent storage, "
                "and AWS S3 for object storage with encrypted bucket policies."
            ),
        },
    ]


@pytest.fixture
def malayalam_corpus():
    return [
        {
            "page": 1,
            "kind": "page",
            "text": (
                "ബിലാൽ അഹമ്മദ് കൊച്ചിൻ ശാസ്ത്ര സാങ്കേതിക സർവകലാശാലയിൽ (CUSAT) "
                "ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസ് ആൻഡ് ഡാറ്റാ സയൻസിൽ എം.ടെക് പഠനം നടത്തുന്നു."
            ),
        },
        {
            "page": 2,
            "kind": "page",
            "text": (
                "ഡോകുമൈൻഡ് എഐ (DocuMind AI) ഡോക്യുമെന്റ് ചോദ്യോത്തര ആപ്ലിക്കേഷനാണ്. "
                "ഇത് മലയാളം, ഇംഗ്ലീഷ് ഭാഷകളിലെ ചോദ്യങ്ങൾക്ക് ഉത്തരം നൽകാൻ പ്രാപ്തമാണ്."
            ),
        },
        {
            "page": 3,
            "kind": "page",
            "text": (
                "പദ്ധതിയുടെ ആകെ ബജറ്റ് 50,000 ഡോളർ ആണ്. 2026 ഒക്ടോബർ മാസത്തിൽ പദ്ധതി പൂർത്തിയാക്കും."
            ),
        },
    ]


def test_embedding_model_configuration_and_dimension(multilingual_model):
    """Verify that the embedding model is paraphrase-multilingual-MiniLM-L12-v2 and has dim 384."""
    assert "paraphrase-multilingual-MiniLM-L12-v2" in EMBEDDING_MODEL
    dim = get_embedding_dimension(multilingual_model)
    assert dim == 384

    # Verify vector normalization (cosine metric equivalence with inner product)
    sample_vec = multilingual_model.encode(["ടെസ്റ്റ് വാചകം"], normalize_embeddings=True)
    norm = np.linalg.norm(sample_vec[0])
    assert pytest.approx(norm, rel=1e-3) == 1.0


def test_english_query_to_english_document(english_corpus, multilingual_model):
    """Scenario 1: English query -> English document."""
    doc_id = "test-en-to-en"
    chunks_count = index_pages(doc_id, english_corpus)
    assert chunks_count == len(english_corpus)

    results = retrieve(doc_id, "Which university is Bilal studying at?", top_k=3)
    assert len(results) > 0

    # Top result must contain university info from Page 1
    top_chunk = results[0]
    assert "Cochin University" in top_chunk["text"]
    assert top_chunk["page"] == 1
    assert top_chunk["kind"] == "page"
    assert top_chunk["score"] > 0.3


def test_malayalam_query_to_malayalam_document(malayalam_corpus, multilingual_model):
    """Scenario 2: Malayalam query -> Malayalam document."""
    doc_id = "test-ml-to-ml"
    index_pages(doc_id, malayalam_corpus)

    # Query: ബിലാൽ എവിടെയാണ് പഠിക്കുന്നത്? (Where is Bilal studying?)
    results = retrieve(doc_id, "ബിലാൽ എവിടെയാണ് പഠിക്കുന്നത്?", top_k=3)
    assert len(results) > 0

    top_chunk = results[0]
    assert "കൊച്ചിൻ ശാസ്ത്ര സാങ്കേതിക സർവകലാശാലയിൽ" in top_chunk["text"]
    assert top_chunk["page"] == 1
    assert top_chunk["score"] > 0.3


def test_malayalam_query_to_english_document_cross_lingual(english_corpus, multilingual_model):
    """Scenario 3: Malayalam query -> English document (Cross-lingual retrieval)."""
    doc_id = "test-ml-to-en"
    index_pages(doc_id, english_corpus)

    # Query in Malayalam asking about university
    results = retrieve(doc_id, "ബിലാൽ ഏത് സർവകലാശാലയിലാണ് പഠിക്കുന്നത്?", top_k=3)
    assert len(results) > 0

    # The top retrieved chunk MUST be the educational background chunk on page 1
    top_chunk = results[0]
    assert "Cochin University of Science and Technology" in top_chunk["text"]
    assert top_chunk["page"] == 1
    assert top_chunk["score"] > 0.25

    # Secondary cross-lingual test: Query about frontend framework / user interface
    results_fe = retrieve(doc_id, "ഉപയോക്തൃ ഇന്റർഫേസിന് ഉപയോഗിക്കുന്ന ഫ്രണ്ട്എൻഡ് ഏതാണ്?", top_k=3)
    assert len(results_fe) > 0
    assert any("React" in chunk["text"] for chunk in results_fe)

    # Cloud storage cross-lingual query: AWS S3 evidence must be in retrieved chunks
    results_storage = retrieve(doc_id, "ക്ലൗഡ് സംഭരണത്തിനായി എന്ത് സേവനമാണ് ഉപയോഗിക്കുന്നത്?", top_k=3)
    assert len(results_storage) > 0
    assert any("AWS S3" in chunk["text"] for chunk in results_storage)


def test_english_query_to_malayalam_document_cross_lingual(malayalam_corpus, multilingual_model):
    """Scenario 4: English query -> Malayalam document (Cross-lingual retrieval)."""
    doc_id = "test-en-to-ml"
    index_pages(doc_id, malayalam_corpus)

    # English query about project budget
    results = retrieve(doc_id, "What is the total project budget?", top_k=3)
    assert len(results) > 0

    top_chunk = results[0]
    assert "50,000" in top_chunk["text"]
    assert top_chunk["page"] == 3


def test_top_five_ranking_and_metadata_preservation(english_corpus, multilingual_model):
    """Verify that top-5 ranking returns correct evidence with preserved metadata."""
    doc_id = "test-metadata-preservation"
    index_pages(doc_id, english_corpus)

    results = retrieve(doc_id, "Tell me about Docker containerization and AWS storage", top_k=5)
    assert len(results) <= 5
    assert len(results) == len(english_corpus)

    # Page 3 should be top ranked
    top_chunk = results[0]
    assert "Docker" in top_chunk["text"]
    assert "AWS S3" in top_chunk["text"]
    assert top_chunk["page"] == 3
    assert top_chunk["kind"] == "page"
    assert "score" in top_chunk
    assert isinstance(top_chunk["score"], float)

    # Ensure all returned chunks have page and score
    for item in results:
        assert "page" in item
        assert "kind" in item
        assert "text" in item
        assert "score" in item


def test_index_rebuild_on_model_change(english_corpus, multilingual_model):
    """Verify that when an index was built with a different model, it automatically rebuilds safely."""
    doc_id = "test-rebuild-compat"
    index_pages(doc_id, english_corpus)

    # Simulate an index built with an older model name
    stored = DOCUMENTS[doc_id]
    original_model = stored.embedding_model
    assert "paraphrase-multilingual-MiniLM-L12-v2" in original_model

    # Artificially alter the stored embedding_model to mimic an old index
    DOCUMENTS[doc_id] = DocumentIndex(
        index=stored.index,
        chunks=stored.chunks,
        embedding_model="sentence-transformers/all-MiniLM-L6-v2",
        dimension=stored.dimension,
        pages=stored.pages,
    )

    # Upon retrieval, rag.py must detect the mismatch and rebuild using the active model
    results = retrieve(doc_id, "CUSAT University", top_k=2)
    assert len(results) > 0

    # Verify the document index was rebuilt and now reflects the active multilingual model
    updated_doc = DOCUMENTS[doc_id]
    assert updated_doc.embedding_model == get_current_embedding_model()
    assert "Cochin University" in results[0]["text"]


def test_incompatible_index_raises_when_pages_missing(english_corpus):
    """Verify that if model differs and raw pages are missing, IncompatibleIndexError is raised."""
    doc_id = "test-incompatible-no-pages"
    index_pages(doc_id, english_corpus)

    stored = DOCUMENTS[doc_id]
    # Simulate missing original pages
    DOCUMENTS[doc_id] = DocumentIndex(
        index=stored.index,
        chunks=stored.chunks,
        embedding_model="old-legacy-model",
        dimension=stored.dimension,
        pages=None,
    )

    with pytest.raises(IncompatibleIndexError) as exc_info:
        retrieve(doc_id, "Test question")

    assert "incompatible with the active model" in str(exc_info.value)


def test_rebuild_all_indexes(english_corpus, malayalam_corpus, multilingual_model):
    """Verify that rebuild_all_indexes reindexes all documents successfully."""
    doc1 = "rebuild-doc-1"
    doc2 = "rebuild-doc-2"
    index_pages(doc1, english_corpus)
    index_pages(doc2, malayalam_corpus)

    results = rebuild_all_indexes()
    assert doc1 in results
    assert doc2 in results
    assert results[doc1] == len(english_corpus)
    assert results[doc2] == len(malayalam_corpus)


def test_error_handling_empty_pages():
    """Verify ValueError is raised on empty page input."""
    with pytest.raises(ValueError, match="No pages provided for indexing"):
        index_pages("empty-doc", [])

    with pytest.raises(ValueError, match="No usable text chunks"):
        index_pages("empty-text-doc", [{"page": 1, "text": "   ", "kind": "page"}])


def test_error_handling_nonexistent_document():
    """Verify KeyError is raised when retrieving from an unknown document."""
    with pytest.raises(KeyError, match="Document index not found"):
        retrieve("nonexistent-document-id", "question")


def test_error_handling_model_load_failure():
    """Verify EmbeddingModelError is raised when an invalid model name is requested."""
    with pytest.raises(EmbeddingModelError) as exc:
        get_embedder("nonexistent-model-org/nonexistent-model-xyz-12345")
    assert "Failed to load embedding model" in str(exc.value)
