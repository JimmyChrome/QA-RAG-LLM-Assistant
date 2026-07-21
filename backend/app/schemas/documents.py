"""Pydantic schemas for document management."""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, ConfigDict, Field

from app.db.models import DocumentStatus, VersionStatus


class DocumentCreate(BaseModel):
    """Input used to create a logical document."""

    title: str = Field(min_length=1, max_length=255)
    category: str | None = Field(default=None, max_length=120)
    description: str | None = None


class DocumentUpdate(BaseModel):
    """Editable logical document metadata."""

    title: str | None = Field(default=None, min_length=1, max_length=255)
    category: str | None = Field(default=None, max_length=120)
    description: str | None = None


class DocumentVersionCreate(BaseModel):
    """Input used to register one uploaded file version."""

    original_filename: str = Field(min_length=1, max_length=255)
    stored_filename: str = Field(min_length=1, max_length=255)
    file_path: str = Field(min_length=1)
    file_extension: str = Field(min_length=1, max_length=20)
    mime_type: str | None = Field(default=None, max_length=120)
    file_size_bytes: int = Field(ge=0)
    checksum_sha256: str = Field(min_length=64, max_length=64)
    uploaded_by: str | None = Field(default=None, max_length=255)


class DocumentVersionRead(BaseModel):
    """Serialized document-version metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    document_id: str
    version_number: int
    original_filename: str
    stored_filename: str
    file_path: str
    file_extension: str
    mime_type: str | None
    file_size_bytes: int
    checksum_sha256: str
    status: VersionStatus
    is_active: bool
    page_count: int | None
    character_count: int | None
    chunk_count: int
    error_message: str | None
    uploaded_by: str | None
    created_at: datetime
    indexed_at: datetime | None


class DocumentRead(BaseModel):
    """Serialized logical document metadata."""

    model_config = ConfigDict(from_attributes=True)

    id: str
    title: str
    category: str | None
    description: str | None
    status: DocumentStatus
    active_version_id: str | None
    created_at: datetime
    updated_at: datetime


class DocumentDetail(DocumentRead):
    """Logical document together with all versions."""

    versions: list[DocumentVersionRead] = Field(default_factory=list)


class DocumentList(BaseModel):
    """Paginated document result."""

    items: list[DocumentRead]
    total: int
    limit: int
    offset: int
