from src.models import Page, TextChunk


def chunk_pages(
    pages: list[Page],
    chunk_size: int = 1000,
    overlap: int = 200,
) -> list[TextChunk]:
    """
    Split document pages into smaller overlapping text chunks.

    Each chunk retains the page number it came from.
    """

    if chunk_size <= 0:

        raise ValueError(
            "chunk_size must be greater than 0"
        )

    if overlap < 0:

        raise ValueError(
            "overlap cannot be negative"
        )

    if overlap >= chunk_size:

        raise ValueError(
            "overlap must be smaller than chunk_size"
        )

    chunks = []

    for page in pages:

        text = page.text.strip()

        if not text:
            continue

        start = 0
        chunk_number = 1

        while start < len(text):

            end = start + chunk_size

            chunk_text = text[
                start:end
            ].strip()

            if chunk_text:

                chunk_id = (
                    f"page_{page.page_number}"
                    f"_chunk_{chunk_number}"
                )

                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        text=chunk_text,
                        page_number=page.page_number,
                    )
                )

                chunk_number += 1

            if end >= len(text):
                break

            start = end - overlap

    return chunks


if __name__ == "__main__":

    pages = [
        Page(
            page_number=1,
            text=(
                "NovaTech Industries reported revenue "
                "of 120 million US dollars in 2023. "
                "Revenue increased to 150 million "
                "in 2024."
            ),
        )
    ]

    chunks = chunk_pages(
        pages,
        chunk_size=80,
        overlap=20,
    )

    print(
        f"Created {len(chunks)} chunk(s).\n"
    )

    for chunk in chunks:

        print(
            f"--- {chunk.chunk_id} | "
            f"Page {chunk.page_number} ---"
        )

        print(
            chunk.text
        )

        print()