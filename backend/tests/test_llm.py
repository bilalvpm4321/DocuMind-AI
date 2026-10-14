"""Unit tests for llm.py provider routing and language propagation."""
import pytest
from unittest.mock import patch, AsyncMock
from app.llm import generate_answer
from app.schemas import AnswerLanguage


@pytest.mark.asyncio
async def test_generate_answer_gemini_routing():
    with patch("app.llm.call_gemini", new_callable=AsyncMock) as mock_gemini:
        mock_gemini.return_value = "Gemini answer in Malayalam"
        result = await generate_answer(
            question="Tell me about the doc",
            context_chunks=[{"page": 1, "text": "Some text", "kind": "page"}],
            provider="gemini",
            answer_language=AnswerLanguage.ML,
        )
        assert result["provider"] == "gemini"
        assert result["answer_language"] == "ml"
        assert result["answer"] == "Gemini answer in Malayalam"
        assert mock_gemini.called
        # Check system prompt passed to call_gemini has Malayalam instruction
        sys_prompt = mock_gemini.call_args[0][0]
        assert "മലയാളം" in sys_prompt


@pytest.mark.asyncio
async def test_generate_answer_ollama_routing():
    with patch("app.llm.call_ollama", new_callable=AsyncMock) as mock_ollama:
        mock_ollama.return_value = "Ollama answer in English"
        result = await generate_answer(
            question="Tell me about the doc",
            context_chunks=[{"page": 1, "text": "Some text", "kind": "page"}],
            provider="ollama",
            answer_language=AnswerLanguage.EN,
        )
        assert result["provider"] == "ollama"
        assert result["answer_language"] == "en"
        assert result["answer"] == "Ollama answer in English"
        assert mock_ollama.called
        sys_prompt = mock_ollama.call_args[0][0]
        assert "precise document assistant" in sys_prompt


@pytest.mark.asyncio
async def test_generate_answer_both_providers_language_propagation():
    with patch("app.llm.call_gemini", new_callable=AsyncMock) as mock_gemini, \
         patch("app.llm.call_ollama", new_callable=AsyncMock) as mock_ollama:
        mock_gemini.return_value = "Gemini മലയാളം"
        mock_ollama.return_value = "Ollama മലയാളം"

        result = await generate_answer(
            question="Tell me about the doc",
            context_chunks=[{"page": 1, "text": "Some text", "kind": "page"}],
            provider="both",
            answer_language=AnswerLanguage.ML,
        )

        assert result["provider"] == "both"
        assert result["answer_language"] == "ml"
        assert "Gemini മലയാളം" in result["answer"]
        assert "Ollama മലയാളം" in result["answer"]
        # Both received the same Malayalam system prompt
        assert "മലയാളം" in mock_gemini.call_args[0][0]
        assert "മലയാളം" in mock_ollama.call_args[0][0]
