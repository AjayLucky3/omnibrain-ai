from embedder import TextEmbedder
from vector_store import VectorStore


class Retriever:
    """
    Handles semantic search over the OmniBrain vector store.
    """

    def __init__(self):
        self.embedder = TextEmbedder()
        self.vector_store = VectorStore()

    def search(
        self,
        query: str,
        limit: int = 5,
    ):
        """
        Convert a user question into an embedding
        and search Qdrant for similar document chunks.
        """

        query_vector = self.embedder.embed_text(query)

        results = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
        )

        return results

    def close(self) -> None:
        """
        Explicitly close the underlying vector store.
        """

        self.vector_store.close()


if __name__ == "__main__":
    retriever = Retriever()

    try:
        query = input("Enter your question: ")

        results = retriever.search(
            query,
            limit=3,
        )

        print("\nSEMANTIC SEARCH RESULTS")
        print("=" * 60)

        if not results:
            print("No results found.")

        for index, result in enumerate(results, start=1):
            print(f"\n--- Result {index} ---")
            print(f"Score: {result.score}")
            print(f"Chunk ID: {result.payload['chunk_id']}")
            print(f"Page: {result.payload['page_number']}")
            print(f"Text: {result.payload['text'][:500]}")

    finally:
        retriever.close()