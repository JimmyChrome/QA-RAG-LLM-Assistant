"""Safe local file storage for uploaded documents."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass
from pathlib import Path

from fastapi import UploadFile

from app.core.config import settings


class UnsupportedFileTypeError(ValueError):
    """Raised when an uploaded file extension is not allowed."""


class UploadTooLargeError(ValueError):
    """Raised when an uploaded file exceeds the configured size limit."""


class EmptyUploadError(ValueError):
    """Raised when an uploaded file contains no bytes."""


@dataclass(frozen=True, slots=True)
class StoredUpload:
    """Metadata for a safely persisted upload."""

    original_filename: str
    stored_filename: str
    file_path: str
    file_extension: str
    mime_type: str | None
    file_size_bytes: int
    checksum_sha256: str


def _safe_filename_component(value: str) -> str:
    """Return a filesystem-safe filename component."""
    normalized = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip())
    normalized = normalized.strip("._-")
    return normalized or "document"


class FileStorageService:
    """Validate and persist uploaded documents to local storage."""

    def __init__(self, upload_directory: Path | None = None) -> None:
        self.upload_directory = upload_directory or settings.upload_dir

    async def save_upload(
        self,
        upload: UploadFile,
        *,
        document_id: str,
    ) -> StoredUpload:
        """Validate, hash, and save one uploaded file."""
        original_filename = Path(upload.filename or "").name
        if not original_filename:
            raise EmptyUploadError("The uploaded file must have a filename.")

        extension = Path(original_filename).suffix.lower()
        if extension not in settings.supported_extensions:
            allowed = ", ".join(sorted(settings.supported_extensions))
            raise UnsupportedFileTypeError(
                f"Unsupported file type '{extension or '(none)'}'. "
                f"Allowed types: {allowed}."
            )

        maximum_bytes = settings.max_upload_size_mb * 1024 * 1024
        digest = hashlib.sha256()
        total_bytes = 0

        document_directory = self.upload_directory / document_id
        document_directory.mkdir(parents=True, exist_ok=True)

        safe_stem = _safe_filename_component(Path(original_filename).stem)
        stored_filename = f"{uuid.uuid4().hex}-{safe_stem}{extension}"
        destination = document_directory / stored_filename

        try:
            with destination.open("wb") as target:
                while chunk := await upload.read(1024 * 1024):
                    total_bytes += len(chunk)
                    if total_bytes > maximum_bytes:
                        raise UploadTooLargeError(
                            f"Upload exceeds the {settings.max_upload_size_mb} MB limit."
                        )

                    digest.update(chunk)
                    target.write(chunk)

            if total_bytes == 0:
                raise EmptyUploadError("The uploaded file is empty.")

        except Exception:
            destination.unlink(missing_ok=True)
            raise
        finally:
            await upload.close()

        return StoredUpload(
            original_filename=original_filename,
            stored_filename=stored_filename,
            file_path=str(destination.resolve()),
            file_extension=extension,
            mime_type=upload.content_type,
            file_size_bytes=total_bytes,
            checksum_sha256=digest.hexdigest(),
        )

    def delete_file(self, file_path: str | Path) -> None:
        """Delete a stored file if it exists."""
        Path(file_path).unlink(missing_ok=True)
