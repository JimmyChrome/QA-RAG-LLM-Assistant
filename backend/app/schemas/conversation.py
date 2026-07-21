"""Pydantic schemas for conversation endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

from app.schemas.query import (
    GenerationOptionsRequest,
    RAGQueryResponseSchema,
)


class ConversationCreateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, max_length=200)

    @field_validator("title")
    @classmethod
    def clean_title(cls, value: str | None) -> str | None:
        if value is None:
            return None
        cleaned = value.strip()
        return cleaned or None


class ConversationQueryRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=4000)
    limit: int | None = Field(default=None, ge=1, le=50)
    document_ids: set[str] | None = None
    allowed_version_ids: set[str] | None = None
    minimum_relevance_score: float | None = Field(default=None, ge=0, le=1)
    deduplicate: bool = True
    generation_options: GenerationOptionsRequest | None = None

    @field_validator("question")
    @classmethod
    def clean_question(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must not be empty")
        return cleaned


class ConversationMessageResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    role: str
    content: str
    citations: list[dict[str, Any]]
    sequence_number: int
    created_at: datetime


class ConversationResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    title: str
    created_at: datetime
    updated_at: datetime
    messages: list[ConversationMessageResponse] = Field(default_factory=list)


class ConversationListResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    items: list[ConversationResponse]
    count: int


class ConversationQueryResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")

    conversation: ConversationResponse
    query: RAGQueryResponseSchema
