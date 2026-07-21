"""Repository for document and version persistence."""

from __future__ import annotations

from sqlalchemy import delete, func, select, update
from sqlalchemy.orm import Session, selectinload

from app.db.models import (
    Document,
    DocumentChunk,
    DocumentStatus,
    DocumentVersion,
    IndexingJob,
    IndexingJobStatus,
    VersionStatus,
    utc_now,
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
            raise ValueError(
                "A newly uploaded version must be indexed before it can be activated."
            )

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

        if version.status != VersionStatus.INDEXED:
            raise ValueError(
                "Only indexed document versions can be activated."
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

    def begin_indexing(
        self,
        document_id: str,
        version_id: str,
    ) -> tuple[DocumentVersion, IndexingJob]:
        """Mark a version as processing and create an indexing job."""
        version = self.get_version(document_id, version_id)

        if version.status == VersionStatus.PROCESSING:
            raise ValueError("This document version is already being indexed.")

        if version.status == VersionStatus.ARCHIVED:
            raise ValueError("An archived document version cannot be indexed.")

        version.status = VersionStatus.PROCESSING
        version.error_message = None

        job = IndexingJob(
            document_version_id=version.id,
            status=IndexingJobStatus.RUNNING,
            stage="loading",
            progress_percent=5,
            started_at=utc_now(),
        )

        self.db.add(job)
        self.db.commit()
        self.db.refresh(version)
        self.db.refresh(job)

        return version, job

    def update_indexing_job(
        self,
        job_id: str,
        *,
        stage: str,
        progress_percent: int,
    ) -> IndexingJob:
        """Update the current indexing stage and progress."""
        job = self.db.get(IndexingJob, job_id)

        if job is None:
            raise LookupError(f"Indexing job '{job_id}' was not found.")

        job.stage = stage
        job.progress_percent = max(0, min(progress_percent, 100))

        self.db.commit()
        self.db.refresh(job)
        return job

    def complete_indexing(
        self,
        *,
        document_id: str,
        version_id: str,
        job_id: str,
        page_count: int,
        character_count: int,
        chunks: list,
        vector_ids: list[str],
    ) -> DocumentVersion:
        """Persist chunk metadata and mark indexing as completed."""
        if len(chunks) != len(vector_ids):
            raise ValueError(
                "The number of chunks does not match the number of vector IDs."
            )

        version = self.get_version(document_id, version_id)
        job = self.db.get(IndexingJob, job_id)

        if job is None:
            raise LookupError(f"Indexing job '{job_id}' was not found.")

        self.db.execute(
            delete(DocumentChunk).where(
                DocumentChunk.document_version_id == version_id
            )
        )

        for chunk, vector_id in zip(chunks, vector_ids, strict=True):
            self.db.add(
                DocumentChunk(
                    document_version_id=version_id,
                    chunk_index=chunk.chunk_index,
                    vector_id=vector_id,
                    content_preview=chunk.text[:500],
                    character_start=chunk.start_char,
                    character_end=chunk.end_char,
                    page_number=chunk.page_number,
                    token_count=None,
                )
            )

        now = utc_now()

        version.status = VersionStatus.INDEXED
        version.page_count = page_count
        version.character_count = character_count
        version.chunk_count = len(chunks)
        version.error_message = None
        version.indexed_at = now

        job.status = IndexingJobStatus.COMPLETED
        job.stage = "completed"
        job.progress_percent = 100
        job.completed_at = now
        job.error_message = None

        self.db.commit()
        self.db.refresh(version)
        return version

    def fail_indexing(
        self,
        *,
        document_id: str,
        version_id: str,
        job_id: str | None,
        error_message: str,
    ) -> DocumentVersion:
        """Persist a failed indexing state."""
        self.db.rollback()

        version = self.get_version(document_id, version_id)
        version.status = VersionStatus.FAILED
        version.error_message = error_message[:4000]

        if job_id is not None:
            job = self.db.get(IndexingJob, job_id)

            if job is not None:
                job.status = IndexingJobStatus.FAILED
                job.stage = "failed"
                job.error_message = error_message[:4000]
                job.completed_at = utc_now()

        self.db.commit()
        self.db.refresh(version)
        return version

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
