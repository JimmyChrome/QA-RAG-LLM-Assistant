"""Document-version indexing orchestration."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.core.config import settings
from app.core.logging import get_logger
from app.db.models import DocumentVersion
from app.ingestion.loader import DocumentLoader
from app.ingestion.preprocessor import DocumentPreprocessor
from app.rag.chunker import DocumentChunker, TextChunk
from app.rag.vector_store import ChromaVectorStore
from app.repositories.documents import DocumentRepository


logger = get_logger(__name__)


class DocumentIndexingService:
    """Coordinate extraction, preprocessing, chunking, and vector indexing."""

    def __init__(
        self,
        db: Session,
        *,
        vector_store: ChromaVectorStore,
        loader: DocumentLoader | None = None,
        preprocessor: DocumentPreprocessor | None = None,
        chunker: DocumentChunker | None = None,
    ) -> None:
        self.repository = DocumentRepository(db)
        self.vector_store = vector_store
        self.loader = loader or DocumentLoader()
        self.preprocessor = preprocessor or DocumentPreprocessor()
        self.chunker = chunker or DocumentChunker(
            chunk_size=settings.chunk_size,
            overlap=settings.chunk_overlap,
        )

    def index_version(
        self,
        document_id: str,
        version_id: str,
    ) -> DocumentVersion:
        """
        Extract, preprocess, chunk, embed, and index one document version.

        This implementation runs synchronously. It is appropriate for the
        controlled TXT test and local MVP development.
        """
        version, job = self.repository.begin_indexing(
            document_id,
            version_id,
        )

        try:
            logger.info(
                "Starting indexing for document %s version %s.",
                document_id,
                version_id,
            )

            self.repository.update_indexing_job(
                job.id,
                stage="loading",
                progress_percent=10,
            )

            extracted = self.loader.load(version.file_path)

            self.repository.update_indexing_job(
                job.id,
                stage="preprocessing",
                progress_percent=30,
            )

            processed = self.preprocessor.process(extracted)

            if not processed.full_text.strip():
                raise ValueError(
                    "The document did not contain any extractable text."
                )

            self.repository.update_indexing_job(
                job.id,
                stage="chunking",
                progress_percent=50,
            )

            chunks = self.chunker.chunk(processed)
            chunks = self._filter_and_reindex_chunks(chunks)

            if not chunks:
                raise ValueError(
                    "The document did not produce any indexable text chunks."
                )

            self.repository.update_indexing_job(
                job.id,
                stage="embedding",
                progress_percent=70,
            )

            # Remove obsolete vectors before re-indexing this version.
            self.vector_store.delete_version(version_id)

            vector_ids = self.vector_store.add_chunks(
                chunks,
                document_id=document_id,
                version_id=version_id,
            )

            self.repository.update_indexing_job(
                job.id,
                stage="saving",
                progress_percent=90,
            )

            indexed_version = self.repository.complete_indexing(
                document_id=document_id,
                version_id=version_id,
                job_id=job.id,
                page_count=processed.page_count,
                character_count=processed.character_count,
                chunks=chunks,
                vector_ids=vector_ids,
            )

            logger.info(
                "Indexed document %s version %s with %s chunks.",
                document_id,
                version_id,
                len(chunks),
            )

            return indexed_version

        except Exception as exc:
            logger.exception(
                "Indexing failed for document %s version %s.",
                document_id,
                version_id,
            )

            # Best-effort cleanup if Chroma was partially updated.
            try:
                self.vector_store.delete_version(version_id)
            except Exception:
                logger.exception(
                    "Failed to clean partial vectors for version %s.",
                    version_id,
                )

            self.repository.fail_indexing(
                document_id=document_id,
                version_id=version_id,
                job_id=job.id,
                error_message=str(exc),
            )

            raise

    @staticmethod
    def _filter_and_reindex_chunks(
        chunks: list[TextChunk],
    ) -> list[TextChunk]:
        """Remove empty/tiny chunks and assign contiguous chunk indexes."""
        filtered = [
            chunk
            for chunk in chunks
            if len(chunk.text.strip()) >= settings.min_chunk_length
        ]

        for index, chunk in enumerate(filtered):
            chunk.chunk_index = index

        return filtered