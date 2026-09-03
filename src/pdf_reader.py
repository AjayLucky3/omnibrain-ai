from pathlib import Path

import pymupdf

from src.models import Page


def extract_pages_from_pdf(
    pdf_path: str | Path,
) -> list[Page]:
    """
    Extract text from a PDF while preserving page numbers.
    """

    pdf_path = Path(
        pdf_path
    ).resolve()

    if not pdf_path.exists():
        raise FileNotFoundError(
            f"PDF file not found: {pdf_path}"
        )

    pages = []

    document = pymupdf.open(
        pdf_path
    )

    try:

        for page_number, page in enumerate(
            document,
            start=1,
        ):

            text = page.get_text().strip()

            pages.append(
                Page(
                    page_number=page_number,
                    text=text,
                )
            )

    finally:

        document.close()

    return pages


if __name__ == "__main__":

    pdf_path = input(
        "Enter PDF path: "
    ).strip()

    pages = extract_pages_from_pdf(
        pdf_path
    )

    print(
        "\nPDF TEXT EXTRACTION SUCCESSFUL"
    )
    print("=" * 60)

    print(
        f"Pages extracted: {len(pages)}"
    )

    for page in pages:

        print(
            f"\n--- Page {page.page_number} ---"
        )

        print(
            page.text
        )