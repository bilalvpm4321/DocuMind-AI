"""Malayalam quality evaluation test suite for RAG and citation grounding."""
import pytest
import numpy as np
from unittest.mock import patch
from app.rag import index_pages, retrieve
from app.prompts import build_prompts
from app.schemas import AnswerLanguage


class MockEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        arr = np.ones((len(texts), 384), dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms


@pytest.fixture(autouse=True)
def mock_embedding_model():
    with patch("app.rag.get_embedder", return_value=MockEmbedder()):
        yield


@pytest.fixture
def indexed_malayalam_sample():
    doc_id = "eval-sample-ml-001"
    pages = [
        {
            "page": 1,
            "text": "DocuMind AI ഒരു രേഖ വിശകലന ആപ്ലിക്കേഷനാണ്. ഇത് നിർമ്മിച്ചത് ബിലാൽ അഹമ്മദ് ആണ്. ആകെ 5 മോഡലുകൾ പിന്തുണയ്ക്കുന്നു.",
            "kind": "page"
        },
        {
            "page": 2,
            "text": "ആപ്ലിക്കേഷന്റെ വാർഷിക വരുമാനം $50,000 ആണ്. റിലീസ് തീയതി 2026 ഒക്ടോബർ 12 ആണ്.",
            "kind": "page"
        }
    ]
    index_pages(doc_id, pages)
    return doc_id


def test_malayalam_retrieval_and_citation_preservation(indexed_malayalam_sample):
    doc_id = indexed_malayalam_sample
    chunks = retrieve(doc_id, "ആരാണ് ഇത് നിർമ്മിച്ചത്?", top_k=2)

    assert len(chunks) > 0
    # Must retrieve chunks containing text
    found_author_chunk = any("ബിലാൽ അഹമ്മദ്" in c["text"] for c in chunks)
    assert found_author_chunk is True
    assert chunks[0]["page"] in (1, 2)


def test_malayalam_prompt_contains_evidence_and_script(indexed_malayalam_sample):
    doc_id = indexed_malayalam_sample
    chunks = retrieve(doc_id, "വരുമാനം എത്രയാണ്?", top_k=2)

    sys_prompt, user_prompt, _ = build_prompts(
        question="വരുമാനം എത്രയാണ്?",
        context_chunks=chunks,
        language=AnswerLanguage.ML,
    )

    # Prompt enforces Malayalam script and citations
    assert "മലയാളം ലിപി" in sys_prompt
    assert "$50,000" in user_prompt
    assert "[Source: page 2" in user_prompt or "[Source: page 1" in user_prompt


def test_unanswerable_question_insufficient_evidence_rule():
    sys_prompt, _, _ = build_prompts(
        question="ചൊവ്വയിലേക്കുള്ള റോക്കറ്റ് വിക്ഷേപണ സമയം എത്രയാണ്?",
        context_chunks=[],
        language=AnswerLanguage.ML,
    )
    # Checks rule that models must explicitly report lack of evidence
    assert "മതിയായ വിശദാംശങ്ങൾ ലഭ്യമല്ല" in sys_prompt
