"""Dependency providers for API routes."""

from __future__ import annotations

from collections.abc import Iterator
from functools import lru_cache
from typing import Callable

from fastapi import Request
from sqlalchemy.orm import Session

from app.repositories.conversation import ConversationRepository
from app.rag.answer_processor import AnswerProcessor
from app.rag.llm import OllamaLLMProvider
from app.rag.prompt_builder import PromptBuilder
from app.rag.retriever import Retriever
from app.rag.vector_store import ChromaVectorStore
from app.services.conversation import ConversationService
from app.services.rag_query import RAGQueryService

from app.core.config import settings
from app.rag.embeddings import SentenceTransformerEmbeddingProvider


def get_rag_query_service(request: Request) -> RAGQueryService:
    """Return an app override or lazily construct the query service."""
    override = getattr(request.app.state, "rag_query_service", None)
    if override is not None:
        return override

    service = _build_default_rag_query_service()
    request.app.state.rag_query_service = service
    return service


def get_conversation_service(
    request: Request,
) -> Iterator[ConversationService]:
    """
    Provide a request-scoped conversation service and database session.

    Tests can override this dependency directly. Production requests use the
    SQLAlchemy session factory exposed by app.db.database.
    """
    override = getattr(request.app.state, "conversation_service", None)
    if override is not None:
        yield override
        return

    session_factory = _resolve_session_factory()
    session = session_factory()

    try:
        yield ConversationService(
            repository=ConversationRepository(session),
            rag_query_service=get_rag_query_service(request),
        )
    finally:
        session.close()

@lru_cache(maxsize=1)
def get_embedding_provider() -> SentenceTransformerEmbeddingProvider:
    """Return the shared production embedding provider."""
    return SentenceTransformerEmbeddingProvider(
        model_name=settings.embedding_model,
        device=settings.embedding_device,
        batch_size=settings.embedding_batch_size,
        normalize_embeddings=settings.normalize_embeddings,
    )


@lru_cache(maxsize=1)
def get_vector_store() -> ChromaVectorStore:
    """Return the shared persistent Chroma vector store."""
    return ChromaVectorStore(
        persist_directory=settings.chroma_dir,
        embedding_provider=get_embedding_provider(),
        collection_name=settings.chroma_collection,
    )


@lru_cache(maxsize=1)
def _build_default_rag_query_service() -> RAGQueryService:
    """Construct the production query service."""
    vector_store = get_vector_store()
    retriever = Retriever(vector_store=vector_store)

    llm_provider = OllamaLLMProvider(
        model=settings.ollama_model,
    )

    return RAGQueryService(
        retriever=retriever,
        prompt_builder=PromptBuilder(),
        llm_provider=llm_provider,
        answer_processor=AnswerProcessor(),
    )


@lru_cache(maxsize=1)
def _resolve_session_factory() -> Callable[[], Session]:
    """
    Locate the project's SQLAlchemy session factory.

    The project may expose it as SessionLocal, session_factory, or
    SessionFactory. Rename the candidate here if your database.py uses a
    different name.
    """
    from app.db import database

    for name in ("SessionLocal", "session_factory", "SessionFactory"):
        candidate = getattr(database, name, None)
        if candidate is not None and callable(candidate):
            return candidate

    raise RuntimeError(
        "No SQLAlchemy session factory was found in app.db.database. "
        "Expected SessionLocal, session_factory, or SessionFactory."
    )
