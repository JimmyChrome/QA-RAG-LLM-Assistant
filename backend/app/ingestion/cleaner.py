"""Text normalization for extracted documents."""

from __future__ import annotations

import re
import unicodedata
from dataclasses import replace

from app.ingestion.models import ExtractedDocument, ExtractedPage


class TextCleaner:
    """Normalize extracted text while preserving page boundaries."""

    _space_pattern = re.compile(r"[ \t]+")
    _blank_line_pattern = re.compile(r"\n{3,}")
    _hyphenated_line_break_pattern = re.compile(r"(?<=\w)-\n(?=\w)")
    _line_break_pattern = re.compile(r"(?<!\n)\n(?!\n)")
    _control_character_pattern = re.compile(
        r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]"
    )

    def __init__(
        self,
        *,
        join_wrapped_lines: bool = True,
        normalize_unicode: bool = True,
    ) -> None:
        self.join_wrapped_lines = join_wrapped_lines
        self.normalize_unicode = normalize_unicode

    def clean_text(self, text: str) -> str:
        """Return normalized text suitable for chunking."""
        if self.normalize_unicode:
            text = unicodedata.normalize("NFKC", text)

        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = self._control_character_pattern.sub("", text)
        text = self._hyphenated_line_break_pattern.sub("", text)

        lines = [self._space_pattern.sub(" ", line).strip() for line in text.split("\n")]
        text = "\n".join(lines)

        if self.join_wrapped_lines:
            paragraphs = []
            for block in re.split(r"\n\s*\n", text):
                block = block.strip()
                if not block:
                    continue

                if self._looks_like_structured_block(block):
                    paragraphs.append(block)
                else:
                    paragraphs.append(self._line_break_pattern.sub(" ", block))

            text = "\n\n".join(paragraphs)

        text = self._blank_line_pattern.sub("\n\n", text)
        return text.strip()

    def clean_document(self, document: ExtractedDocument) -> ExtractedDocument:
        """Return a new document containing cleaned page text."""
        cleaned_pages = [
            ExtractedPage(
                page_number=page.page_number,
                text=self.clean_text(page.text),
                metadata=dict(page.metadata),
            )
            for page in document.pages
        ]

        metadata = dict(document.metadata)
        metadata["cleaned"] = True
        metadata["original_character_count"] = document.character_count

        cleaned_document = replace(
            document,
            pages=cleaned_pages,
            metadata=metadata,
        )
        cleaned_document.metadata["cleaned_character_count"] = (
            cleaned_document.character_count
        )
        return cleaned_document

    @staticmethod
    def _looks_like_structured_block(block: str) -> bool:
        """Preserve blocks that resemble lists, headings, or tables."""
        lines = block.splitlines()

        if len(lines) <= 1:
            return False

        structured_lines = 0
        for line in lines:
            stripped = line.strip()
            if (
                stripped.startswith(("-", "*", "•", "#", "[Table"))
                or re.match(r"^\d+[.)]\s+", stripped)
                or " | " in stripped
                or stripped.isupper()
            ):
                structured_lines += 1

        return structured_lines >= max(1, len(lines) // 2)
