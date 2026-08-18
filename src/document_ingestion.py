from pathlib import Path
from uuid import uuid4

from models import Document, DocumentImage
from pdf_image_extractor import extract_images_from_pdf
from pdf_reader import extract_pages_from_pdf
from text_chunker import chunk_pages


def ingest_document(
    pdf_path: str | Path,
    image_output_directory: str | Path = "data/extracted_images",
) -> Document:
    """
    Ingest a PDF and build a structured Document object.

    The ingestion process:
        1. Extracts PDF pages and text.
        2. Splits page text into chunks.
        3. Extracts embedded images.
        4. Combines everything into one Document object.

    Args:
        pdf_path: Path to the PDF file.
        image_output_directory: Directory for extracted images.

    Returns:
        A fully populated Document object.
    """

    pdf_path = Path(pdf_path)

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    document_id = str(uuid4())

    pages = extract_pages_from_pdf(pdf_path)

    text_chunks = chunk_pages(pages)

    extracted_images = extract_images_from_pdf(
        pdf_path,
        image_output_directory,
    )

    document_images = []

    for image in extracted_images:
        document_images.append(
            DocumentImage(
                image_id=(
                    f"page_{image['page_number']}"
                    f"_image_{image['image_index']}"
                ),
                page_number=image["page_number"],
                path=Path(image["path"]),
                width=image["width"],
                height=image["height"],
                extension=image["extension"],
            )
        )

    return Document(
        document_id=document_id,
        filename=pdf_path.name,
        pages=pages,
        text_chunks=text_chunks,
        images=document_images,
    )


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    document = ingest_document(pdf_path)

    print("\nDOCUMENT INGESTION SUCCESSFUL")
    print("=" * 60)

    print(f"Document ID: {document.document_id}")
    print(f"Filename: {document.filename}")
    print(f"Pages: {len(document.pages)}")
    print(f"Text chunks: {len(document.text_chunks)}")
    print(f"Images: {len(document.images)}")

    print("\nTEXT CHUNKS")
    print("-" * 60)

    for chunk in document.text_chunks[:5]:
        print(
            f"{chunk.chunk_id} | "
            f"Page {chunk.page_number}"
        )
        print(chunk.text[:200])
        print()

    print("IMAGES")
    print("-" * 60)

    for image in document.images:
        print(
            f"{image.image_id} | "
            f"Page {image.page_number} | "
            f"{image.width}x{image.height} | "
            f"{image.path}"
        )