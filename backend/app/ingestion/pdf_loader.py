"""PDF text extraction using PyMuPDF."""
from __future__ import annotations
from pathlib import Path
import fitz
from app.ingestion.base import BaseDocumentLoader
from app.ingestion.exceptions import DocumentLoadError, EmptyDocumentError
from app.ingestion.models import ExtractedDocument, ExtractedPage

class PDFDocumentLoader(BaseDocumentLoader):
    supported_extensions = {".pdf"}

    def load(self, path: Path) -> ExtractedDocument:
        try:
            with fitz.open(path) as pdf:
                pages = [
                    ExtractedPage(
                        page_number=index + 1,
                        text=page.get_text("text"),
                        metadata={"width": float(page.rect.width), "height": float(page.rect.height)},
                    )
                    for index, page in enumerate(pdf)
                ]
                metadata = dict(pdf.metadata or {})
                metadata["page_count"] = len(pages)
        except Exception as exc:
            raise DocumentLoadError(f"Failed to read PDF file '{path.name}'.") from exc

        document = ExtractedDocument(
            source_path=str(path.resolve()),
            file_extension=path.suffix.lower(),
            pages=pages,
            metadata=metadata,
        )
        if not document.full_text.strip():
            raise EmptyDocumentError(f"No extractable text was found in PDF '{path.name}'.")
        return document
