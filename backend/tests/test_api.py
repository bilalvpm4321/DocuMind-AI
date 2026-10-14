"""API integration tests for FastAPI endpoints."""
import pytest
import numpy as np
from unittest.mock import patch, AsyncMock
from fastapi.testclient import TestClient
from app.main import app


class MockEmbedder:
    def encode(self, texts, normalize_embeddings=True):
        # Deterministic 384-dim normalized vector
        arr = np.ones((len(texts), 384), dtype="float32")
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / norms


@pytest.fixture(autouse=True)
def mock_embedding_model():
    with patch("app.rag.get_embedder", return_value=MockEmbedder()):
        yield


client = TestClient(app)


def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert "provider" in data
    assert "default_answer_language" in data
    assert "supported_languages" in data
    assert "ml" in data["supported_languages"]


def _create_sample_doc():
    content = b"This is a test document about artificial intelligence. Page 1 content."
    response = client.post(
        "/api/documents/upload",
        files={"file": ("test.txt", content, "text/plain")}
    )
    assert response.status_code == 200
    return response.json()["document_id"]


def test_upload_document_text():
    doc_id = _create_sample_doc()
    assert doc_id is not None


def test_chat_english_language():
    doc_id = _create_sample_doc()
    with patch("app.main.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "provider": "gemini",
            "answer_language": "en",
            "answer": "This document is about AI [Page 1].",
            "answers": {"gemini": "This document is about AI [Page 1]."}
        }

        response = client.post(
            "/api/chat",
            json={
                "document_id": doc_id,
                "question": "What is this doc about?",
                "provider": "gemini",
                "answer_language": "en"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer_language"] == "en"
        assert "sources" in data
        assert len(data["sources"]) > 0
        assert data["sources"][0]["text"] is not None


def test_chat_malayalam_language():
    doc_id = _create_sample_doc()
    with patch("app.main.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "provider": "gemini",
            "answer_language": "ml",
            "answer": "ഈ രേഖ ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസിനെക്കുറിച്ചാണ് [Page 1].",
            "answers": {"gemini": "ഈ രേഖ ആർട്ടിഫിഷ്യൽ ഇന്റലിജൻസിനെക്കുറിച്ചാണ് [Page 1]."}
        }

        response = client.post(
            "/api/chat",
            json={
                "document_id": doc_id,
                "question": "ഇത് എന്തിനെക്കുറിച്ചാണ്?",
                "provider": "gemini",
                "answer_language": "ml"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer_language"] == "ml"
        assert "ആർട്ടിഫിഷ്യൽ" in data["answer"]


def test_chat_legacy_request_without_answer_language():
    doc_id = _create_sample_doc()
    with patch("app.main.generate_answer", new_callable=AsyncMock) as mock_gen:
        mock_gen.return_value = {
            "provider": "ollama",
            "answer_language": "en",
            "answer": "Legacy request default answer",
            "answers": {"ollama": "Legacy request default answer"}
        }

        response = client.post(
            "/api/chat",
            json={
                "document_id": doc_id,
                "question": "Summarize please",
                "provider": "ollama"
            }
        )

        assert response.status_code == 200
        data = response.json()
        assert data["answer_language"] == "en"


def test_chat_invalid_language_422():
    response = client.post(
        "/api/chat",
        json={
            "document_id": "nonexistent-doc",
            "question": "Question",
            "answer_language": "french"
        }
    )
    assert response.status_code == 422


def test_chat_missing_document_404():
    response = client.post(
        "/api/chat",
        json={
            "document_id": "nonexistent-doc",
            "question": "Question",
            "answer_language": "en"
        }
    )
    assert response.status_code == 404
