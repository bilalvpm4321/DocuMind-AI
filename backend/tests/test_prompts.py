"""Unit tests for answer_prompts builder."""
import pytest
from app.prompts import build_prompts
from app.schemas import AnswerLanguage


def test_english_prompt_generation():
    chunks = [
        {"page": 1, "text": "DocuMind is an intelligent document chatbot.", "kind": "page"},
        {"page": 2, "text": "It supports FAISS vector search.", "kind": "page"},
    ]
    system_prompt, user_prompt, safe_history = build_prompts(
        question="What is DocuMind?",
        context_chunks=chunks,
        history=[{"role": "user", "content": "hello"}],
        language=AnswerLanguage.EN,
    )

    assert "precise document assistant" in system_prompt
    assert "Cite page numbers like [Page 3]" in system_prompt
    assert "DOCUMENT EXCERPTS:" in user_prompt
    assert "[Source: page 1; type: page]" in user_prompt
    assert "What is DocuMind?" in user_prompt
    assert len(safe_history) == 1


def test_malayalam_prompt_generation():
    chunks = [
        {"page": 1, "text": "Bilal Ahamed is a Full Stack Developer.", "kind": "page"},
    ]
    system_prompt, user_prompt, safe_history = build_prompts(
        question="Who is Bilal?",
        context_chunks=chunks,
        language=AnswerLanguage.ML,
    )

    # Must contain Malayalam script instructions
    assert "മലയാളം" in system_prompt
    assert "CRITICAL ANSWER LANGUAGE INSTRUCTION" in system_prompt
    assert "Do NOT use Manglish" in system_prompt
    assert "[Page 3] or [പേജ് 3]" in system_prompt
    assert "QUESTION (Answer in Malayalam using the excerpts above):" in user_prompt
    assert "Who is Bilal?" in user_prompt


def test_insufficient_evidence_instruction():
    system_prompt, _, _ = build_prompts(
        question="What is the CEO salary?",
        context_chunks=[],
        language=AnswerLanguage.ML,
    )
    assert "മതിയായ വിശദാംശങ്ങൾ ലഭ്യമല്ല" in system_prompt


def test_string_language_coercion():
    _, _, _ = build_prompts("hi", [], language="ml")
    _, _, _ = build_prompts("hi", [], language="english")
    with pytest.raises(ValueError):
        build_prompts("hi", [], language="unsupported_lang")
