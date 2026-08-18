from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Page:
    """
    Represents one page of a document.
    """

    page_number: int
    text: str


@dataclass
class TextChunk:
    """
    Represents a chunk of text extracted from a document.
    """

    chunk_id: str
    text: str
    page_number: int


@dataclass
class DocumentImage:
    """
    Represents an image extracted from a document.
    """

    image_id: str
    page_number: int
    path: Path
    width: int
    height: int
    extension: str


@dataclass
class Document:
    """
    Represents an ingested document and all of its extracted content.
    """

    document_id: str
    filename: str

    pages: list[Page] = field(default_factory=list)

    text_chunks: list[TextChunk] = field(default_factory=list)

    images: list[DocumentImage] = field(default_factory=list)