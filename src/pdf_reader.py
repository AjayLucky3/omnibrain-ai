import pymupdf


def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract text from every page of a PDF.

    Args:
        pdf_path: Path to the PDF file.

    Returns:
        All extracted text combined into one string.
    """

    document = pymupdf.open(pdf_path)

    pages = []

    for page_number, page in enumerate(document, start=1):
        text = page.get_text()

        pages.append(
            f"\n--- Page {page_number} ---\n{text}"
        )

    document.close()

    return "\n".join(pages)


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    extracted_text = extract_text_from_pdf(pdf_path)

    print("\nPDF TEXT EXTRACTION SUCCESSFUL")
    print("=" * 60)
    print(extracted_text)