"""Base interface for file-specific document loaders."""
from __future__ import annotations
from abc import ABC, abstractmethod
from pathlib import Path
from app.ingestion.models import ExtractedDocument

class BaseDocumentLoader(ABC):
    supported_extensions: set[str] = set()

    def supports(self, extension: str) -> bool:
        return extension.lower() in self.supported_extensions

    @abstractmethod
    def load(self, path: Path) -> ExtractedDocument:
        raise NotImplementedError
