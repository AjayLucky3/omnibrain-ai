from document_ingestion import ingest_document
from embedder import TextEmbedder
from vector_store import VectorStore


def index_document(pdf_path: str) -> None:
    """
    Ingest a PDF, generate embeddings for its text chunks,
    and store those embeddings in Qdrant.
    """

    print("Starting document ingestion...")

    document = ingest_document(pdf_path)

    print(f"Document: {document.filename}")
    print(f"Pages: {len(document.pages)}")
    print(f"Text chunks: {len(document.text_chunks)}")
    print(f"Images: {len(document.images)}")

    print("\nGenerating text embeddings...")

    embedder = TextEmbedder()

    embedded_chunks = embedder.embed_chunks(
        document.text_chunks
    )

    print(
        f"Generated embeddings for "
        f"{len(embedded_chunks)} chunks."
    )

    print("\nStoring vectors in Qdrant...")

    vector_store = VectorStore()

    vector_store.add_chunks(
        chunks=embedded_chunks,
        document_id=document.document_id,
    )

    vector_store.close()

    print("\nDOCUMENT INDEXING SUCCESSFUL")
    print("=" * 60)
    print(f"Document ID: {document.document_id}")
    print(f"Indexed chunks: {len(embedded_chunks)}")


if __name__ == "__main__":
    pdf_path = input("Enter PDF path: ")

    index_document(pdf_path)