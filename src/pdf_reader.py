from pathlib import Path

import pymupdf

from models import Page


def extract_pages_from_pdf(pdf_path: str | Path) -> list[Page]:
    """
    Extract text from a PDF while preserving page numbers.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        A list of Page objects.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text().strip()

        pages.append(
            Page(
                page_number=page_number,
                text=text,
            )
        )

    document.close()

    return pages


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    pages = extract_pages_from_pdf(pdf_path)

    print("\nPDF TEXT EXTRACTION SUCCESSFUL")
    print("=" * 60)

    for page in pages:
        print(f"\n--- Page {page.page_number} ---")
        print(page.text)