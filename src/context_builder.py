from dataclasses import dataclass

from retriever import Retriever


@dataclass
class RetrievedContext:
    """
    Represents one piece of information retrieved
    from the vector database.
    """

    text: str
    page_number: int
    chunk_id: str
    document_id: str
    score: float


class ContextBuilder:
    """
    Converts raw vector search results into clean
    context objects that can later be passed to an LLM.
    """

    def __init__(self):
        self.retriever = Retriever()

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievedContext]:
        """
        Retrieve relevant chunks and convert them
        into structured context objects.
        """

        results = self.retriever.search(
            query=query,
            limit=limit,
        )

        contexts = []

        for result in results:
            payload = result.payload

            context = RetrievedContext(
                text=payload["text"],
                page_number=payload["page_number"],
                chunk_id=payload["chunk_id"],
                document_id=payload["document_id"],
                score=result.score,
            )

            contexts.append(context)

        return contexts

    def close(self) -> None:
        """
        Close the underlying retriever.
        """

        self.retriever.close()


if __name__ == "__main__":
    builder = ContextBuilder()

    try:
        query = input("Enter your question: ")

        contexts = builder.retrieve_context(
            query=query,
            limit=3,
        )

        print("\nRETRIEVED CONTEXT")
        print("=" * 60)

        if not contexts:
            print("No relevant context found.")

        for index, context in enumerate(contexts, start=1):
            print(f"\n--- Context {index} ---")
            print(f"Score: {context.score}")
            print(f"Document ID: {context.document_id}")
            print(f"Chunk ID: {context.chunk_id}")
            print(f"Page: {context.page_number}")
            print(f"Text: {context.text[:500]}")

    finally:
        builder.close()