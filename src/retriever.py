from src.embedder import TextEmbedder
from src.vector_store import VectorStore


# ============================================================
# CONFIGURATION
# ============================================================

DEFAULT_RETRIEVAL_LIMIT = 5

# Minimum similarity score required for a result
# to be considered relevant.
#
# Qdrant uses cosine similarity here.
# Higher score = more similar.
#
# Start with 0.30 and tune later based on testing.
SIMILARITY_THRESHOLD = 0.30


# ============================================================
# RETRIEVER
# ============================================================

class Retriever:
    """
    Handles semantic search over the OmniBrain vector store.

    Responsibilities:

    1. Convert user questions into embeddings.
    2. Search Qdrant for similar chunks.
    3. Remove results that are below the similarity threshold.
    """

    def __init__(self):

        self.embedder = TextEmbedder()

        self.vector_store = VectorStore()

        self.similarity_threshold = SIMILARITY_THRESHOLD

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        query: str,
        limit: int = DEFAULT_RETRIEVAL_LIMIT,
    ):
        """
        Convert a user question into an embedding
        and search Qdrant for relevant document chunks.

        Results below the similarity threshold are removed.
        """

        # ----------------------------------------------------
        # Validate query
        # ----------------------------------------------------

        query = query.strip()

        if not query:

            return []

        # ----------------------------------------------------
        # Create query embedding
        # ----------------------------------------------------

        query_vector = self.embedder.embed_text(
            query
        )

        # ----------------------------------------------------
        # Search Qdrant
        # ----------------------------------------------------

        results = self.vector_store.search(
            query_vector=query_vector,
            limit=limit,
        )

        # ----------------------------------------------------
        # Filter low-quality results
        # ----------------------------------------------------

        filtered_results = []

        for result in results:

            score = result.score

            if score >= self.similarity_threshold:

                filtered_results.append(
                    result
                )

        return filtered_results

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        """
        Explicitly close the underlying vector store.
        """

        self.vector_store.close()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    retriever = Retriever()

    try:

        query = input(
            "Enter your question: "
        )

        results = retriever.search(
            query=query,
            limit=5,
        )

        print()
        print("=" * 60)
        print("SEMANTIC SEARCH RESULTS")
        print("=" * 60)

        print(
            f"Query: {query}"
        )

        print(
            f"Similarity threshold: "
            f"{retriever.similarity_threshold}"
        )

        print(
            f"Results found: "
            f"{len(results)}"
        )

        if not results:

            print()
            print(
                "No relevant document chunks found."
            )

        else:

            for index, result in enumerate(
                results,
                start=1,
            ):

                payload = result.payload or {}

                print()
                print(
                    f"--- Result {index} ---"
                )

                print(
                    f"Score: "
                    f"{result.score}"
                )

                print(
                    f"Document ID: "
                    f"{payload.get('document_id')}"
                )

                print(
                    f"Filename: "
                    f"{payload.get('filename')}"
                )

                print(
                    f"Chunk ID: "
                    f"{payload.get('chunk_id')}"
                )

                print(
                    f"Page: "
                    f"{payload.get('page_number')}"
                )

                print(
                    f"Text: "
                    f"{payload.get('text', '')[:500]}"
                )

    finally:

        retriever.close()