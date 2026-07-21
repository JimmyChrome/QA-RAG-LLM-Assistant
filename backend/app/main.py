"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes.conversations import router as conversations_router
from app.api.routes.documents import router as documents_router
from app.api.routes.query import router as query_router

from app.core.config import settings
from app.core.logging import get_logger, setup_logging
from app.db.database import initialize_database


setup_logging()
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(_: FastAPI) -> AsyncIterator[None]:
    """Run application startup and shutdown tasks."""
    settings.create_runtime_directories()
    initialize_database()

    logger.info(
        "Starting %s in %s mode.",
        settings.app_name,
        settings.app_env,
    )

    try:
        yield
    finally:
        logger.info("Shutting down %s.", settings.app_name)


app = FastAPI(
    title=settings.app_name,
    version="0.1.0",
    description=(
        "Backend API for the UP Diliman Quality Assurance Office "
        "Retrieval-Augmented Generation assistant."
    ),
    docs_url=f"{settings.api_prefix}/docs",
    redoc_url=f"{settings.api_prefix}/redoc",
    openapi_url=f"{settings.api_prefix}/openapi.json",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(documents_router)
app.include_router(query_router)
app.include_router(conversations_router)


@app.get("/", tags=["System"])
async def root() -> dict[str, str]:
    """Return a basic service message."""
    return {
        "message": f"{settings.app_name} backend is running.",
        "environment": settings.app_env,
    }


@app.get(f"{settings.api_prefix}/health", tags=["System"])
async def health_check() -> dict[str, str]:
    """Return the current backend health status."""
    return {
        "status": "healthy",
        "service": settings.app_name,
        "environment": settings.app_env,
    }
