"""Unit tests for Pydantic schemas and AnswerLanguage enum."""
import pytest
from pydantic import ValidationError
from app.schemas import AnswerLanguage, ChatRequest


def test_answer_language_enum_values():
    assert AnswerLanguage.EN.value == "en"
    assert AnswerLanguage.ML.value == "ml"


def test_answer_language_from_str_valid():
    assert AnswerLanguage.from_str("en") == AnswerLanguage.EN
    assert AnswerLanguage.from_str("EN") == AnswerLanguage.EN
    assert AnswerLanguage.from_str("english") == AnswerLanguage.EN
    assert AnswerLanguage.from_str("ml") == AnswerLanguage.ML
    assert AnswerLanguage.from_str("Malayalam") == AnswerLanguage.ML
    assert AnswerLanguage.from_str(None) == AnswerLanguage.EN


def test_answer_language_from_str_invalid():
    with pytest.raises(ValueError, match="Unsupported answer language"):
        AnswerLanguage.from_str("spanish")


def test_chat_request_default_language():
    req = ChatRequest(document_id="doc-123", question="What is RAG?")
    assert req.answer_language == AnswerLanguage.EN
    assert req.provider == "gemini"


def test_chat_request_explicit_malayalam():
    req = ChatRequest(
        document_id="doc-123",
        question="What is RAG?",
        answer_language="ml",
    )
    assert req.answer_language == AnswerLanguage.ML


def test_chat_request_invalid_language():
    with pytest.raises(ValidationError):
        ChatRequest(
            document_id="doc-123",
            question="What is RAG?",
            answer_language="german",
        )
