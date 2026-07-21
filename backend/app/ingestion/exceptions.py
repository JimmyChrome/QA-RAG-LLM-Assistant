"""Exceptions raised by document ingestion components."""

class DocumentLoadError(RuntimeError):
    """Base exception for document loading failures."""

class UnsupportedDocumentTypeError(DocumentLoadError):
    """Raised when no loader exists for a file extension."""

class DocumentFileNotFoundError(DocumentLoadError):
    """Raised when the requested document file does not exist."""

class EmptyDocumentError(DocumentLoadError):
    """Raised when no meaningful text can be extracted."""
