"""Shared data structures for document ingestion."""
from __future__ import annotations
from dataclasses import dataclass, field

@dataclass(slots=True)
class ExtractedPage:
    page_number: int
    text: str
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

@dataclass(slots=True)
class ExtractedDocument:
    source_path: str
    file_extension: str
    pages: list[ExtractedPage]
    metadata: dict[str, str | int | float | bool | None] = field(default_factory=dict)

    @property
    def full_text(self) -> str:
        return "\n\n".join(page.text.strip() for page in self.pages if page.text.strip())

    @property
    def page_count(self) -> int:
        return len(self.pages)

    @property
    def character_count(self) -> int:
        return len(self.full_text)
