"""End-to-end tests for the document upload API."""

from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker

from app.core.config import settings
from app.db.database import Base, get_db
from app.main import app


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    """Provide a test client with isolated database and upload storage."""
    database_path = tmp_path / "test.db"
    engine = create_engine(
        f"sqlite:///{database_path.as_posix()}",
        connect_args={"check_same_thread": False},
    )
    testing_session = sessionmaker(
        bind=engine,
        autoflush=False,
        autocommit=False,
        expire_on_commit=False,
        class_=Session,
    )
    Base.metadata.create_all(engine)

    original_upload_dir = settings.upload_dir
    settings.upload_dir = tmp_path / "uploads"

    def override_get_db():
        database = testing_session()
        try:
            yield database
        finally:
            database.close()

    app.dependency_overrides[get_db] = override_get_db

    with TestClient(app) as test_client:
        yield test_client

    app.dependency_overrides.clear()
    settings.upload_dir = original_upload_dir
    Base.metadata.drop_all(engine)
    engine.dispose()


def test_create_upload_browse_and_archive(client: TestClient) -> None:
    create_response = client.post(
        f"{settings.api_prefix}/documents",
        json={
            "title": "Quality Assurance Manual",
            "category": "Manual",
            "description": "Internal QA procedures.",
        },
    )
    assert create_response.status_code == 201

    document = create_response.json()
    document_id = document["id"]

    upload_response = client.post(
        f"{settings.api_prefix}/documents/{document_id}/versions",
        files={
            "file": (
                "qa-manual.txt",
                b"Quality assurance policies and procedures.",
                "text/plain",
            )
        },
        data={
            "uploaded_by": "test-user",
            "make_active": "true",
        },
    )
    assert upload_response.status_code == 201

    version = upload_response.json()
    assert version["version_number"] == 1
    assert version["is_active"] is True
    assert version["checksum_sha256"]

    stored_path = Path(version["file_path"])
    assert stored_path.exists()
    assert stored_path.read_bytes() == b"Quality assurance policies and procedures."

    detail_response = client.get(
        f"{settings.api_prefix}/documents/{document_id}"
    )
    assert detail_response.status_code == 200
    detail = detail_response.json()
    assert detail["active_version_id"] == version["id"]
    assert len(detail["versions"]) == 1

    list_response = client.get(
        f"{settings.api_prefix}/documents",
        params={"search": "quality"},
    )
    assert list_response.status_code == 200
    assert list_response.json()["total"] == 1

    archive_response = client.post(
        f"{settings.api_prefix}/documents/{document_id}/archive"
    )
    assert archive_response.status_code == 200
    assert archive_response.json()["status"] == "archived"
    assert archive_response.json()["active_version_id"] is None


def test_reject_unsupported_upload(client: TestClient) -> None:
    create_response = client.post(
        f"{settings.api_prefix}/documents",
        json={"title": "Unsupported Upload Test"},
    )
    document_id = create_response.json()["id"]

    response = client.post(
        f"{settings.api_prefix}/documents/{document_id}/versions",
        files={
            "file": (
                "malware.exe",
                b"not really executable content",
                "application/octet-stream",
            )
        },
    )

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_reject_duplicate_version(client: TestClient) -> None:
    create_response = client.post(
        f"{settings.api_prefix}/documents",
        json={"title": "Duplicate Upload Test"},
    )
    document_id = create_response.json()["id"]

    upload = {
        "file": (
            "policy.txt",
            b"same policy contents",
            "text/plain",
        )
    }

    first_response = client.post(
        f"{settings.api_prefix}/documents/{document_id}/versions",
        files=upload,
    )
    assert first_response.status_code == 201

    second_response = client.post(
        f"{settings.api_prefix}/documents/{document_id}/versions",
        files={
            "file": (
                "policy-copy.txt",
                b"same policy contents",
                "text/plain",
            )
        },
    )

    assert second_response.status_code == 409
