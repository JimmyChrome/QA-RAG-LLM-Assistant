"""Combined preprocessing pipeline for extracted documents."""

from __future__ import annotations

from app.ingestion.cleaner import TextCleaner
from app.ingestion.metadata import MetadataExtractor
from app.ingestion.models import ExtractedDocument


class DocumentPreprocessor:
    """Clean extracted text and enrich it with derived metadata."""

    def __init__(
        self,
        cleaner: TextCleaner | None = None,
        metadata_extractor: MetadataExtractor | None = None,
    ) -> None:
        self.cleaner = cleaner or TextCleaner()
        self.metadata_extractor = metadata_extractor or MetadataExtractor()

    def process(self, document: ExtractedDocument) -> ExtractedDocument:
        cleaned = self.cleaner.clean_document(document)
        return self.metadata_extractor.enrich_document(cleaned)
