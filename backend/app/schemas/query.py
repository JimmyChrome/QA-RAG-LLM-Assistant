"""Pydantic schemas for the public RAG query API."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ConversationMessageRequest(BaseModel):
    """A prior user or assistant message included with a query."""

    model_config = ConfigDict(extra="forbid")

    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=12000)

    @field_validator("content")
    @classmethod
    def validate_content(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("content must not be empty")
        return cleaned


class GenerationOptionsRequest(BaseModel):
    """Optional model-generation settings exposed by the API."""

    model_config = ConfigDict(extra="forbid")

    temperature: float | None = Field(default=None, ge=0)
    top_p: float | None = Field(default=None, gt=0, le=1)
    max_tokens: int | None = Field(default=None, ge=1, le=8192)
    seed: int | None = None
    stop: list[str] | None = None

    @field_validator("stop")
    @classmethod
    def validate_stop(cls, value: list[str] | None) -> list[str] | None:
        if value is None:
            return None

        cleaned = [item.strip() for item in value if item.strip()]
        if not cleaned:
            raise ValueError("stop must contain at least one non-empty value")
        return cleaned


class RAGQueryRequestSchema(BaseModel):
    """HTTP request body for a grounded document query."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    limit: int | None = Field(default=None, ge=1, le=50)
    document_ids: set[str] | None = None
    allowed_version_ids: set[str] | None = None
    minimum_relevance_score: float | None = Field(
        default=None,
        ge=0,
        le=1,
    )
    deduplicate: bool = True
    conversation_history: list[ConversationMessageRequest] = Field(
        default_factory=list,
        max_length=20,
    )
    generation_options: GenerationOptionsRequest | None = None

    @field_validator("question")
    @classmethod
    def validate_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned

    @field_validator("document_ids", "allowed_version_ids")
    @classmethod
    def validate_identifier_sets(
        cls,
        value: set[str] | None,
    ) -> set[str] | None:
        if value is None:
            return None

        cleaned = {item.strip() for item in value if item.strip()}
        if not cleaned:
            raise ValueError("identifier filters must not be empty")
        return cleaned


class AnswerCitationResponse(BaseModel):
    """Frontend-ready source citation."""

    model_config = ConfigDict(extra="forbid")

    source_number: int
    citation_text: str
    chunk_id: str
    document_id: str
    version_id: str
    title: str | None
    page_number: int | None
    source_path: str | None
    excerpt: str
    relevance_score: float
    metadata: dict[str, Any]


class RAGQueryResponseSchema(BaseModel):
    """Successful grounded-answer response."""

    model_config = ConfigDict(extra="forbid")

    question: str
    answer: str
    citations: list[AnswerCitationResponse]
    model: str
    finish_reason: str | None
    prompt_tokens: int | None
    completion_tokens: int | None
    source_count: int
    has_context: bool
    has_valid_citations: bool
    used_fallback: bool


class APIErrorResponse(BaseModel):
    """Consistent API error response."""

    model_config = ConfigDict(extra="forbid")

    detail: str
    code: str
