"""Application service for document-management workflows."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.db.models import Document, DocumentStatus, DocumentVersion
from app.repositories.documents import DocumentRepository
from app.schemas.documents import (
    DocumentCreate,
    DocumentList,
    DocumentRead,
    DocumentUpdate,
    DocumentVersionCreate,
)


class DocumentService:
    """Coordinates validation and repository operations for documents."""

    def __init__(self, db: Session) -> None:
        self.repository = DocumentRepository(db)

    def create_document(self, payload: DocumentCreate) -> Document:
        return self.repository.create_document(payload)

    def get_document(self, document_id: str) -> Document:
        return self.repository.get_document(document_id, include_versions=True)

    def list_documents(
        self,
        *,
        search: str | None = None,
        category: str | None = None,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> DocumentList:
        safe_limit = min(max(limit, 1), 100)
        safe_offset = max(offset, 0)

        items, total = self.repository.list_documents(
            search=search,
            category=category,
            status=status,
            limit=safe_limit,
            offset=safe_offset,
        )

        return DocumentList(
            items=[DocumentRead.model_validate(item) for item in items],
            total=total,
            limit=safe_limit,
            offset=safe_offset,
        )

    def update_document(
        self,
        document_id: str,
        payload: DocumentUpdate,
    ) -> Document:
        return self.repository.update_document(document_id, payload)

    def register_version(
        self,
        document_id: str,
        payload: DocumentVersionCreate,
        *,
        make_active: bool = False,
    ) -> DocumentVersion:
        return self.repository.create_version(
            document_id,
            payload,
            make_active=make_active,
        )

    def activate_version(
        self,
        document_id: str,
        version_id: str,
    ) -> DocumentVersion:
        return self.repository.activate_version(document_id, version_id)

    def archive_document(self, document_id: str) -> Document:
        return self.repository.archive_document(document_id)

    def restore_document(self, document_id: str) -> Document:
        return self.repository.restore_document(document_id)
