"""Pydantic schemas and enums for DocuMind API."""
from enum import Enum
from typing import Any
from pydantic import BaseModel, Field, field_validator


class AnswerLanguage(str, Enum):
    EN = "en"
    ML = "ml"

    @classmethod
    def from_str(cls, val: str | None) -> "AnswerLanguage":
        if not val:
            return cls.EN
        cleaned = str(val).strip().lower()
        if cleaned in ("en", "english"):
            return cls.EN
        if cleaned in ("ml", "malayalam"):
            return cls.ML
        raise ValueError(f"Unsupported answer language: '{val}'. Supported options: 'en' (English), 'ml' (Malayalam).")


class ChatRequest(BaseModel):
    document_id: str
    question: str = Field(min_length=1, max_length=4000)
    history: list[dict[str, Any]] = Field(default_factory=list)
    provider: str = Field(default="gemini")
    answer_language: AnswerLanguage = Field(default=AnswerLanguage.EN)

    @field_validator("answer_language", mode="before")
    @classmethod
    def parse_answer_language(cls, v: Any) -> AnswerLanguage:
        if isinstance(v, AnswerLanguage):
            return v
        return AnswerLanguage.from_str(v)

    # Backward-compatible hook if legacy client sent 'language' instead of 'answer_language'
    @field_validator("provider", mode="before")
    @classmethod
    def normalize_provider(cls, v: Any) -> str:
        return str(v or "gemini").strip().lower()


class SourceChunk(BaseModel):
    page: int | None = None
    kind: str | None = "text"
    text: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    provider: str
    answer_language: AnswerLanguage
    answers: dict[str, str] = Field(default_factory=dict)
    sources: list[dict[str, Any]] = Field(default_factory=list)
