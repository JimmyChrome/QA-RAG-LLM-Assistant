"""Tests for text cleaning and metadata extraction."""

from __future__ import annotations

from app.ingestion.cleaner import TextCleaner
from app.ingestion.metadata import MetadataExtractor
from app.ingestion.models import ExtractedDocument, ExtractedPage
from app.ingestion.preprocessor import DocumentPreprocessor


def build_document(text: str, *, extension: str = ".txt") -> ExtractedDocument:
    return ExtractedDocument(
        source_path=f"/tmp/sample{extension}",
        file_extension=extension,
        pages=[ExtractedPage(page_number=1, text=text)],
        metadata={},
    )


def test_clean_text_normalizes_spacing_and_line_breaks() -> None:
    cleaner = TextCleaner()

    cleaned = cleaner.clean_text(
        "Quality   assurance\r\nrequires\tclear procedures.\n\n\nNext section."
    )

    assert cleaned == (
        "Quality assurance requires clear procedures.\n\nNext section."
    )


def test_clean_text_repairs_hyphenated_line_breaks() -> None:
    cleaner = TextCleaner()

    cleaned = cleaner.clean_text("accredi-\ntation standards")

    assert cleaned == "accreditation standards"


def test_cleaner_preserves_page_boundaries_and_metadata() -> None:
    document = ExtractedDocument(
        source_path="/tmp/sample.pdf",
        file_extension=".pdf",
        pages=[
            ExtractedPage(
                page_number=1,
                text="First   page",
                metadata={"width": 100.0},
            ),
            ExtractedPage(
                page_number=2,
                text="Second\npage",
                metadata={"width": 100.0},
            ),
        ],
        metadata={"author": "QA Office"},
    )

    result = TextCleaner().clean_document(document)

    assert result.page_count == 2
    assert result.pages[0].page_number == 1
    assert result.pages[0].metadata["width"] == 100.0
    assert result.pages[0].text == "First page"
    assert result.pages[1].text == "Second page"
    assert result.metadata["author"] == "QA Office"
    assert result.metadata["cleaned"] is True


def test_extract_markdown_headings_and_statistics() -> None:
    document = build_document(
        "# Quality Assurance Manual\n\n"
        "This manual contains the policies and procedures of the office.\n\n"
        "## Internal Assessment\n\n"
        "The assessment is conducted every semester.",
        extension=".md",
    )

    result = MetadataExtractor().extract(document)

    assert result.title == "Quality Assurance Manual"
    assert result.headings == [
        "Quality Assurance Manual",
        "Internal Assessment",
    ]
    assert result.word_count > 10
    assert result.page_count == 1
    assert result.detected_language == "en"
    assert result.source_filename == "sample.md"


def test_detect_filipino_language() -> None:
    document = build_document(
        "Ang mga patakaran ng opisina ay para sa lahat ng mag-aaral "
        "at ito ay mula sa pamantasan."
    )

    result = MetadataExtractor().extract(document)

    assert result.detected_language == "fil"


def test_existing_document_title_has_priority() -> None:
    document = build_document("# Heading From Content\n\nBody text.")
    document.metadata["title"] = "Official Metadata Title"

    result = MetadataExtractor().extract(document)

    assert result.title == "Official Metadata Title"


def test_preprocessor_cleans_and_enriches_document() -> None:
    document = build_document(
        "# QA   Handbook\n\n"
        "The   handbook\ncontains office procedures."
    )

    result = DocumentPreprocessor().process(document)

    assert result.full_text == (
        "# QA Handbook\n\nThe handbook contains office procedures."
    )
    assert result.metadata["cleaned"] is True
    assert result.metadata["title"] == "QA Handbook"
    assert result.metadata["word_count"] == 7
    assert result.metadata["detected_language"] == "en"
