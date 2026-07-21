"""Central application configuration.

This module loads values from the backend/.env file and exposes them through
one reusable `settings` object.

Usage:
    from app.core.config import settings

    print(settings.app_name)
    print(settings.chunk_size)
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Validated application settings loaded from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # Application
    app_name: str = Field(default="QA Office RAG Assistant")
    app_env: Literal["development", "testing", "production"] = "development"
    app_host: str = "127.0.0.1"
    app_port: int = Field(default=8000, ge=1, le=65535)
    app_reload: bool = True

    # API
    api_prefix: str = "/api/v1"
    cors_origins: list[str] = Field(
        default_factory=lambda: [
            "http://localhost:3000",
            "http://localhost:5173",
        ]
    )

    # Storage
    data_dir: Path = Path("./data")
    upload_dir: Path = Path("./data/uploads")
    archive_dir: Path = Path("./data/archive")
    chroma_dir: Path = Path("./data/chroma")
    sqlite_path: Path = Path("./data/qa_rag.db")

    # Document processing
    supported_extensions: set[str] = Field(
        default_factory=lambda: {".pdf", ".docx", ".txt", ".md"}
    )
    max_upload_size_mb: int = Field(default=25, ge=1)
    chunk_size: int = Field(default=700, ge=100)
    chunk_overlap: int = Field(default=120, ge=0)
    min_chunk_length: int = Field(default=80, ge=1)

    # Embeddings
    embedding_provider: Literal["sentence_transformers"] = "sentence_transformers"
    embedding_model: str = "BAAI/bge-small-en-v1.5"
    embedding_device: str = "cpu"
    embedding_batch_size: int = Field(default=32, ge=1)
    normalize_embeddings: bool = True

    # Vector database
    chroma_collection: str = "qa_office_documents"
    vector_distance_metric: Literal["cosine", "l2", "ip"] = "cosine"

    # Retrieval
    retrieval_top_k: int = Field(default=5, ge=1)
    retrieval_fetch_k: int = Field(default=15, ge=1)
    min_relevance_score: float = Field(default=0.25, ge=0.0, le=1.0)
    retrieve_active_versions_only: bool = True

    # Language model
    llm_provider: Literal["ollama"] = "ollama"
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    llm_temperature: float = Field(default=0.1, ge=0.0, le=2.0)
    llm_max_tokens: int = Field(default=700, ge=1)
    llm_request_timeout_seconds: int = Field(default=120, ge=1)

    # Prompt behavior
    system_prompt: str = (
        "You are the UP Diliman Quality Assurance Office assistant. "
        "Answer only from the provided context. If the context does not "
        "contain enough information, clearly say so. Cite the supplied sources."
    )
    max_context_characters: int = Field(default=16_000, ge=1_000)
    include_conversation_history: bool = True
    max_history_messages: int = Field(default=6, ge=0)

    # Logging
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Path = Path("./data/logs/app.log")

    # Security placeholders
    secret_key: str = "replace-this-with-a-long-random-secret"
    access_token_expire_minutes: int = Field(default=60, ge=1)
    default_access_level: str = "public"

    @field_validator("cors_origins", mode="before")
    @classmethod
    def parse_cors_origins(cls, value: object) -> object:
        """Convert comma-separated CORS origins from .env into a list."""
        if isinstance(value, str):
            return [item.strip() for item in value.split(",") if item.strip()]
        return value

    @field_validator("supported_extensions", mode="before")
    @classmethod
    def parse_supported_extensions(cls, value: object) -> object:
        """Convert extensions into a normalized lowercase set."""
        if isinstance(value, str):
            values = [item.strip() for item in value.split(",") if item.strip()]
        else:
            values = value

        if isinstance(values, (list, tuple, set)):
            return {
                extension.lower()
                if str(extension).startswith(".")
                else f".{str(extension).lower()}"
                for extension in values
            }

        return values

    @field_validator("api_prefix")
    @classmethod
    def normalize_api_prefix(cls, value: str) -> str:
        """Ensure the API prefix starts with one slash and has no trailing slash."""
        normalized = "/" + value.strip("/")
        return normalized if normalized != "/" else ""

    @model_validator(mode="after")
    def validate_retrieval_and_chunking(self) -> "Settings":
        """Validate settings that depend on other settings."""
        if self.chunk_overlap >= self.chunk_size:
            raise ValueError("CHUNK_OVERLAP must be smaller than CHUNK_SIZE.")

        if self.min_chunk_length > self.chunk_size:
            raise ValueError(
                "MIN_CHUNK_LENGTH must be less than or equal to CHUNK_SIZE."
            )

        if self.retrieval_fetch_k < self.retrieval_top_k:
            raise ValueError(
                "RETRIEVAL_FETCH_K must be greater than or equal to "
                "RETRIEVAL_TOP_K."
            )

        return self

    def create_runtime_directories(self) -> None:
        """Create local runtime directories required by the application."""
        directories = {
            self.data_dir,
            self.upload_dir,
            self.archive_dir,
            self.chroma_dir,
            self.sqlite_path.parent,
            self.log_file.parent,
        }

        for directory in directories:
            directory.mkdir(parents=True, exist_ok=True)


@lru_cache
def get_settings() -> Settings:
    """Create and cache one Settings instance for the process."""
    return Settings()


settings = get_settings()
