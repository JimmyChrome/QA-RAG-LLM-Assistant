"""Tests for the document-management core."""

from __future__ import annotations

import hashlib

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.db.database import Base
from app.db.models import DocumentStatus, VersionStatus
from app.repositories.documents import DuplicateDocumentVersionError
from app.schemas.documents import DocumentCreate, DocumentVersionCreate
from app.services.documents import DocumentService


@pytest.fixture
def db_session() -> Session:
    """Provide a fresh in-memory SQLite database for every test."""
    engine = create_engine(
        "sqlite+pysqlite:///:memory:",
        connect_args={"check_same_thread": False},
    )
    Base.metadata.create_all(engine)
    testing_session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )

    session = testing_session()
    try:
        yield session
    finally:
        session.close()
        Base.metadata.drop_all(engine)
        engine.dispose()


def build_version(filename: str, contents: bytes) -> DocumentVersionCreate:
    """Create valid version metadata for repository tests."""
    checksum = hashlib.sha256(contents).hexdigest()
    extension = "." + filename.rsplit(".", maxsplit=1)[-1].lower()

    return DocumentVersionCreate(
        original_filename=filename,
        stored_filename=f"stored-{filename}",
        file_path=f"data/uploads/stored-{filename}",
        file_extension=extension,
        mime_type="application/pdf",
        file_size_bytes=len(contents),
        checksum_sha256=checksum,
        uploaded_by="test-user",
    )


def test_document_lifecycle(db_session: Session) -> None:
    service = DocumentService(db_session)

    document = service.create_document(
        DocumentCreate(
            title="QA Manual",
            category="Manual",
            description="Quality assurance procedures.",
        )
    )

    assert document.title == "QA Manual"
    assert document.status == DocumentStatus.ACTIVE
    assert document.active_version_id is None

    version_one = service.register_version(
        document.id,
        build_version("qa-manual-v1.pdf", b"version one"),
    )
    version_two = service.register_version(
        document.id,
        build_version("qa-manual-v2.pdf", b"version two"),
    )

    assert version_one.version_number == 1
    assert version_two.version_number == 2
    assert version_one.status == VersionStatus.PENDING

    active_version = service.activate_version(document.id, version_two.id)
    refreshed = service.get_document(document.id)

    assert active_version.is_active is True
    assert refreshed.active_version_id == version_two.id
    assert next(
        version for version in refreshed.versions if version.id == version_one.id
    ).is_active is False

    archived = service.archive_document(document.id)
    assert archived.status == DocumentStatus.ARCHIVED
    assert archived.active_version_id is None

    restored = service.restore_document(document.id)
    assert restored.status == DocumentStatus.ACTIVE


def test_list_and_search_documents(db_session: Session) -> None:
    service = DocumentService(db_session)

    service.create_document(
        DocumentCreate(
            title="Internal Assessment Guide",
            category="Guide",
        )
    )
    service.create_document(
        DocumentCreate(
            title="Accreditation Handbook",
            category="Handbook",
        )
    )

    result = service.list_documents(search="assessment")

    assert result.total == 1
    assert result.items[0].title == "Internal Assessment Guide"


def test_duplicate_file_version_is_rejected(db_session: Session) -> None:
    service = DocumentService(db_session)
    document = service.create_document(DocumentCreate(title="Duplicate Test"))
    payload = build_version("same.pdf", b"identical contents")

    service.register_version(document.id, payload)

    with pytest.raises(DuplicateDocumentVersionError):
        service.register_version(document.id, payload)
