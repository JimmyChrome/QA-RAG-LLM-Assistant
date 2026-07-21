"""Plain-text and Markdown document loaders."""
from __future__ import annotations
from pathlib import Path
from app.ingestion.base import BaseDocumentLoader
from app.ingestion.exceptions import DocumentLoadError, EmptyDocumentError
from app.ingestion.models import ExtractedDocument, ExtractedPage

class TextDocumentLoader(BaseDocumentLoader):
    supported_extensions = {".txt"}

    def load(self, path: Path) -> ExtractedDocument:
        try:
            text = path.read_text(encoding="utf-8-sig")
            encoding = "utf-8"
        except UnicodeDecodeError:
            try:
                text = path.read_text(encoding="cp1252")
                encoding = "cp1252"
            except Exception as exc:
                raise DocumentLoadError(f"Failed to decode text file '{path.name}'.") from exc
        except Exception as exc:
            raise DocumentLoadError(f"Failed to read text file '{path.name}'.") from exc

        if not text.strip():
            raise EmptyDocumentError(f"No extractable text was found in '{path.name}'.")

        return ExtractedDocument(
            source_path=str(path.resolve()),
            file_extension=path.suffix.lower(),
            pages=[ExtractedPage(page_number=1, text=text)],
            metadata={"encoding": encoding},
        )

class MarkdownDocumentLoader(TextDocumentLoader):
    supported_extensions = {".md"}

    def load(self, path: Path) -> ExtractedDocument:
        document = super().load(path)
        document.metadata["format"] = "markdown"
        return document
