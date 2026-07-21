"""Repository for document and version persistence."""

from __future__ import annotations

from sqlalchemy import func, select, update
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    Document,
    DocumentStatus,
    DocumentVersion,
    VersionStatus,
)
from app.schemas.documents import (
    DocumentCreate,
    DocumentUpdate,
    DocumentVersionCreate,
)


class DocumentNotFoundError(LookupError):
    """Raised when a logical document does not exist."""


class DocumentVersionNotFoundError(LookupError):
    """Raised when a document version does not exist."""


class DuplicateDocumentVersionError(ValueError):
    """Raised when the same file checksum already exists for a document."""


class DocumentRepository:
    """Encapsulates document-related SQLAlchemy operations."""

    def __init__(self, db: Session) -> None:
        self.db = db

    def create_document(self, payload: DocumentCreate) -> Document:
        document = Document(
            title=payload.title.strip(),
            category=payload.category.strip() if payload.category else None,
            description=payload.description.strip() if payload.description else None,
        )
        self.db.add(document)
        self.db.commit()
        self.db.refresh(document)
        return document

    def get_document(
        self,
        document_id: str,
        *,
        include_versions: bool = False,
    ) -> Document:
        statement = select(Document).where(Document.id == document_id)

        if include_versions:
            statement = statement.options(selectinload(Document.versions))

        document = self.db.scalar(statement)
        if document is None:
            raise DocumentNotFoundError(f"Document '{document_id}' was not found.")

        if include_versions:
            document.versions.sort(
                key=lambda item: item.version_number,
                reverse=True,
            )

        return document

    def list_documents(
        self,
        *,
        search: str | None = None,
        category: str | None = None,
        status: DocumentStatus | None = None,
        limit: int = 50,
        offset: int = 0,
    ) -> tuple[list[Document], int]:
        filters = []

        if search:
            search_term = f"%{search.strip().lower()}%"
            filters.append(
                func.lower(Document.title).like(search_term)
                | func.lower(func.coalesce(Document.description, "")).like(search_term)
            )

        if category:
            filters.append(func.lower(Document.category) == category.strip().lower())

        if status:
            filters.append(Document.status == status)

        query = (
            select(Document)
            .where(*filters)
            .order_by(Document.updated_at.desc())
            .limit(limit)
            .offset(offset)
        )
        count_query = select(func.count(Document.id)).where(*filters)

        documents = list(self.db.scalars(query).all())
        total = int(self.db.scalar(count_query) or 0)
        return documents, total

    def update_document(
        self,
        document_id: str,
        payload: DocumentUpdate,
    ) -> Document:
        document = self.get_document(document_id)

        updates = payload.model_dump(exclude_unset=True)
        if "title" in updates and updates["title"] is not None:
            updates["title"] = updates["title"].strip()
        if "category" in updates and updates["category"] is not None:
            updates["category"] = updates["category"].strip()
        if "description" in updates and updates["description"] is not None:
            updates["description"] = updates["description"].strip()

        for field_name, value in updates.items():
            setattr(document, field_name, value)

        self.db.commit()
        self.db.refresh(document)
        return document

    def create_version(
        self,
        document_id: str,
        payload: DocumentVersionCreate,
        *,
        make_active: bool = False,
    ) -> DocumentVersion:
        document = self.get_document(document_id)

        duplicate = self.db.scalar(
            select(DocumentVersion).where(
                DocumentVersion.document_id == document_id,
                DocumentVersion.checksum_sha256 == payload.checksum_sha256,
            )
        )
        if duplicate is not None:
            raise DuplicateDocumentVersionError(
                "This exact file already exists as a version of the document."
            )

        latest_number = self.db.scalar(
            select(func.max(DocumentVersion.version_number)).where(
                DocumentVersion.document_id == document_id
            )
        )
        next_number = int(latest_number or 0) + 1

        version = DocumentVersion(
            document_id=document_id,
            version_number=next_number,
            original_filename=payload.original_filename,
            stored_filename=payload.stored_filename,
            file_path=payload.file_path,
            file_extension=payload.file_extension.lower(),
            mime_type=payload.mime_type,
            file_size_bytes=payload.file_size_bytes,
            checksum_sha256=payload.checksum_sha256.lower(),
            uploaded_by=payload.uploaded_by,
            status=VersionStatus.PENDING,
            is_active=False,
        )
        self.db.add(version)
        self.db.flush()

        if make_active:
            self._activate_version(document, version)

        self.db.commit()
        self.db.refresh(version)
        return version

    def get_version(
        self,
        document_id: str,
        version_id: str,
    ) -> DocumentVersion:
        version = self.db.scalar(
            select(DocumentVersion).where(
                DocumentVersion.id == version_id,
                DocumentVersion.document_id == document_id,
            )
        )
        if version is None:
            raise DocumentVersionNotFoundError(
                f"Version '{version_id}' was not found for document '{document_id}'."
            )
        return version

    def activate_version(
        self,
        document_id: str,
        version_id: str,
    ) -> DocumentVersion:
        document = self.get_document(document_id)
        version = self.get_version(document_id, version_id)

        if version.status not in {VersionStatus.INDEXED, VersionStatus.PENDING}:
            raise ValueError(
                "Only pending or indexed document versions can be activated."
            )

        self._activate_version(document, version)
        self.db.commit()
        self.db.refresh(version)
        return version

    def archive_document(self, document_id: str) -> Document:
        document = self.get_document(document_id)
        document.status = DocumentStatus.ARCHIVED

        self.db.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_id == document_id)
            .values(is_active=False)
        )
        document.active_version_id = None

        self.db.commit()
        self.db.refresh(document)
        return document

    def restore_document(self, document_id: str) -> Document:
        document = self.get_document(document_id)
        document.status = DocumentStatus.ACTIVE
        self.db.commit()
        self.db.refresh(document)
        return document

    def _activate_version(
        self,
        document: Document,
        version: DocumentVersion,
    ) -> None:
        self.db.execute(
            update(DocumentVersion)
            .where(DocumentVersion.document_id == document.id)
            .values(is_active=False)
        )
        version.is_active = True
        document.active_version_id = version.id
        document.status = DocumentStatus.ACTIVE
