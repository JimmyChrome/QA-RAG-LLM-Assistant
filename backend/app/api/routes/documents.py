"""Document-management and upload API routes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status
from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.database import get_db
from app.db.models import DocumentStatus
from app.repositories.documents import (
    DocumentNotFoundError,
    DocumentVersionNotFoundError,
    DuplicateDocumentVersionError,
)
from app.schemas.documents import (
    DocumentCreate,
    DocumentDetail,
    DocumentList,
    DocumentRead,
    DocumentUpdate,
    DocumentVersionCreate,
    DocumentVersionRead,
)
from app.services.documents import DocumentService
from app.services.file_storage import (
    EmptyUploadError,
    FileStorageService,
    UnsupportedFileTypeError,
    UploadTooLargeError,
)


logger = get_logger(__name__)
router = APIRouter(
    prefix=f"{settings.api_prefix}/documents",
    tags=["Documents"],
)


def _service(db: Session) -> DocumentService:
    return DocumentService(db)


@router.post(
    "",
    response_model=DocumentRead,
    status_code=status.HTTP_201_CREATED,
)
def create_document(
    payload: DocumentCreate,
    db: Session = Depends(get_db),
) -> DocumentRead:
    """Create a logical document before uploading versions."""
    document = _service(db).create_document(payload)
    return DocumentRead.model_validate(document)


@router.get("", response_model=DocumentList)
def list_documents(
    search: str | None = Query(default=None, max_length=255),
    category: str | None = Query(default=None, max_length=120),
    document_status: DocumentStatus | None = Query(
        default=None,
        alias="status",
    ),
    limit: int = Query(default=50, ge=1, le=100),
    offset: int = Query(default=0, ge=0),
    db: Session = Depends(get_db),
) -> DocumentList:
    """Browse and filter logical documents."""
    return _service(db).list_documents(
        search=search,
        category=category,
        status=document_status,
        limit=limit,
        offset=offset,
    )


@router.get("/{document_id}", response_model=DocumentDetail)
def get_document(
    document_id: str,
    db: Session = Depends(get_db),
) -> DocumentDetail:
    """Return one document with all registered versions."""
    try:
        document = _service(db).get_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DocumentDetail.model_validate(document)


@router.patch("/{document_id}", response_model=DocumentRead)
def update_document(
    document_id: str,
    payload: DocumentUpdate,
    db: Session = Depends(get_db),
) -> DocumentRead:
    """Update document metadata."""
    try:
        document = _service(db).update_document(document_id, payload)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DocumentRead.model_validate(document)


@router.post(
    "/{document_id}/versions",
    response_model=DocumentVersionRead,
    status_code=status.HTTP_201_CREATED,
)
async def upload_document_version(
    document_id: str,
    file: UploadFile = File(...),
    uploaded_by: str | None = Form(default=None),
    make_active: bool = Form(default=False),
    db: Session = Depends(get_db),
) -> DocumentVersionRead:
    """Upload and register a new version of an existing document."""
    service = _service(db)

    try:
        service.get_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    storage = FileStorageService()

    try:
        stored = await storage.save_upload(file, document_id=document_id)
        payload = DocumentVersionCreate(
            original_filename=stored.original_filename,
            stored_filename=stored.stored_filename,
            file_path=stored.file_path,
            file_extension=stored.file_extension,
            mime_type=stored.mime_type,
            file_size_bytes=stored.file_size_bytes,
            checksum_sha256=stored.checksum_sha256,
            uploaded_by=uploaded_by,
        )
        version = service.register_version(
            document_id,
            payload,
            make_active=make_active,
        )

    except (
        UnsupportedFileTypeError,
        UploadTooLargeError,
        EmptyUploadError,
    ) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    except DuplicateDocumentVersionError as exc:
        storage.delete_file(stored.file_path)
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    except Exception:
        if "stored" in locals():
            storage.delete_file(stored.file_path)
        logger.exception("Failed to upload a version for document %s.", document_id)
        raise

    logger.info(
        "Uploaded document version %s for document %s.",
        version.id,
        document_id,
    )
    return DocumentVersionRead.model_validate(version)


@router.post(
    "/{document_id}/versions/{version_id}/activate",
    response_model=DocumentVersionRead,
)
def activate_document_version(
    document_id: str,
    version_id: str,
    db: Session = Depends(get_db),
) -> DocumentVersionRead:
    """Set one version as the active version."""
    try:
        version = _service(db).activate_version(document_id, version_id)
    except (DocumentNotFoundError, DocumentVersionNotFoundError) as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return DocumentVersionRead.model_validate(version)


@router.post("/{document_id}/archive", response_model=DocumentRead)
def archive_document(
    document_id: str,
    db: Session = Depends(get_db),
) -> DocumentRead:
    """Archive a logical document and deactivate all versions."""
    try:
        document = _service(db).archive_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DocumentRead.model_validate(document)


@router.post("/{document_id}/restore", response_model=DocumentRead)
def restore_document(
    document_id: str,
    db: Session = Depends(get_db),
) -> DocumentRead:
    """Restore an archived logical document."""
    try:
        document = _service(db).restore_document(document_id)
    except DocumentNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return DocumentRead.model_validate(document)
