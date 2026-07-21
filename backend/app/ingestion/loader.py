"""Document loader dispatcher."""
from __future__ import annotations
from pathlib import Path
from app.ingestion.base import BaseDocumentLoader
from app.ingestion.docx_loader import DOCXDocumentLoader
from app.ingestion.exceptions import DocumentFileNotFoundError, UnsupportedDocumentTypeError
from app.ingestion.models import ExtractedDocument
from app.ingestion.pdf_loader import PDFDocumentLoader
from app.ingestion.text_loader import MarkdownDocumentLoader, TextDocumentLoader

class DocumentLoader:
    def __init__(self, loaders: list[BaseDocumentLoader] | None = None) -> None:
        self.loaders = loaders or [
            PDFDocumentLoader(),
            DOCXDocumentLoader(),
            TextDocumentLoader(),
            MarkdownDocumentLoader(),
        ]

    def load(self, file_path: str | Path) -> ExtractedDocument:
        path = Path(file_path)
        if not path.exists() or not path.is_file():
            raise DocumentFileNotFoundError(f"Document file '{path}' does not exist.")

        extension = path.suffix.lower()
        for loader in self.loaders:
            if loader.supports(extension):
                return loader.load(path)

        raise UnsupportedDocumentTypeError(
            f"No document loader is available for '{extension or '(none)'}'."
        )
