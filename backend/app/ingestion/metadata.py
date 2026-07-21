"""Metadata extraction for cleaned documents."""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

from app.ingestion.models import ExtractedDocument


@dataclass(slots=True)
class MetadataExtractionResult:
    """Structured metadata produced from document contents."""

    title: str | None
    headings: list[str]
    word_count: int
    character_count: int
    page_count: int
    average_words_per_page: float
    detected_language: str
    source_filename: str
    file_extension: str
    extra: dict[str, str | int | float | bool | None]

    def as_dict(self) -> dict[str, object]:
        """Return a serializable metadata dictionary."""
        return {
            "title": self.title,
            "headings": self.headings,
            "word_count": self.word_count,
            "character_count": self.character_count,
            "page_count": self.page_count,
            "average_words_per_page": self.average_words_per_page,
            "detected_language": self.detected_language,
            "source_filename": self.source_filename,
            "file_extension": self.file_extension,
            **self.extra,
        }


class MetadataExtractor:
    """Derive lightweight metadata from extracted text and source data."""

    _word_pattern = re.compile(r"\b[\w'-]+\b", re.UNICODE)
    _markdown_heading_pattern = re.compile(r"^\s{0,3}#{1,6}\s+(.+?)\s*$")
    _numbered_heading_pattern = re.compile(
        r"^\s*(?:\d+(?:\.\d+)*)\s+([A-Z][^\n]{2,120})$"
    )

    def extract(self, document: ExtractedDocument) -> MetadataExtractionResult:
        text = document.full_text
        words = self._word_pattern.findall(text)
        page_count = max(document.page_count, 1)

        headings = self._extract_headings(document)
        title = self._select_title(document, headings)
        language = self._detect_language(words)

        existing = {
            key: value
            for key, value in document.metadata.items()
            if key not in {"title", "headings"}
        }

        return MetadataExtractionResult(
            title=title,
            headings=headings,
            word_count=len(words),
            character_count=len(text),
            page_count=document.page_count,
            average_words_per_page=round(len(words) / page_count, 2),
            detected_language=language,
            source_filename=Path(document.source_path).name,
            file_extension=document.file_extension,
            extra=existing,
        )

    def enrich_document(self, document: ExtractedDocument) -> ExtractedDocument:
        """Attach extracted metadata directly to a document."""
        result = self.extract(document)
        document.metadata.update(result.as_dict())
        return document

    def _extract_headings(self, document: ExtractedDocument) -> list[str]:
        headings: list[str] = []
        seen: set[str] = set()

        for page in document.pages:
            for raw_line in page.text.splitlines():
                line = raw_line.strip()
                if not line:
                    continue

                heading = self._heading_from_line(line)
                if heading and heading.lower() not in seen:
                    headings.append(heading)
                    seen.add(heading.lower())

                if len(headings) >= 50:
                    return headings

        return headings

    def _heading_from_line(self, line: str) -> str | None:
        markdown_match = self._markdown_heading_pattern.match(line)
        if markdown_match:
            return markdown_match.group(1).strip()

        numbered_match = self._numbered_heading_pattern.match(line)
        if numbered_match:
            return line.strip()

        if (
            3 <= len(line) <= 100
            and len(line.split()) <= 12
            and line.isupper()
        ):
            return line.title()

        return None

    @staticmethod
    def _select_title(
        document: ExtractedDocument,
        headings: list[str],
    ) -> str | None:
        existing_title = document.metadata.get("title")
        if isinstance(existing_title, str) and existing_title.strip():
            return existing_title.strip()

        if headings:
            return headings[0]

        for page in document.pages:
            for line in page.text.splitlines():
                candidate = line.strip()
                if 3 <= len(candidate) <= 150:
                    return candidate

        return None

    @staticmethod
    def _detect_language(words: list[str]) -> str:
        """Use a conservative stop-word heuristic for English and Filipino."""
        if not words:
            return "unknown"

        english = {
            "the", "and", "of", "to", "in", "for", "is", "are",
            "with", "this", "that", "from", "on",
        }
        filipino = {
            "ang", "ng", "mga", "sa", "ay", "at", "para", "ito",
            "na", "mula", "bilang", "may",
        }

        counts = Counter(word.lower() for word in words)
        english_score = sum(counts[word] for word in english)
        filipino_score = sum(counts[word] for word in filipino)

        if english_score == 0 and filipino_score == 0:
            return "unknown"
        if english_score > filipino_score:
            return "en"
        if filipino_score > english_score:
            return "fil"
        return "mixed"
