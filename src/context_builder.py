from dataclasses import dataclass
import re

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
    Retrieves document chunks and reranks them using
    both semantic similarity and lexical relevance.
    """

    def __init__(self):
        self.retriever = Retriever()

    # ========================================================
    # TOKENIZATION
    # ========================================================

    @staticmethod
    def tokenize(text: str) -> set[str]:
        """
        Convert text into normalized words.
        """

        return set(
            re.findall(
                r"\b[a-zA-Z0-9]+\b",
                text.lower(),
            )
        )

    # ========================================================
    # LEXICAL SCORE
    # ========================================================

    def lexical_score(
        self,
        query: str,
        text: str,
    ) -> float:
        """
        Calculate how many query terms occur in the chunk.
        """

        query_words = self.tokenize(query)
        text_words = self.tokenize(text)

        if not query_words:
            return 0.0

        matches = query_words.intersection(text_words)

        return len(matches) / len(query_words)

    # ========================================================
    # RERANK
    # ========================================================

    def rerank(
        self,
        query: str,
        contexts: list[RetrievedContext],
    ) -> list[RetrievedContext]:
        """
        Combine semantic similarity with lexical relevance.

        Semantic similarity tells us whether the chunk is
        conceptually related.

        Lexical relevance tells us whether the chunk contains
        the actual words/entities from the question.
        """

        scored_contexts = []

        for context in contexts:

            lexical = self.lexical_score(
                query,
                context.text,
            )

            # Semantic score contributes 60%.
            # Lexical score contributes 40%.
            combined_score = (
                context.score * 0.60
                + lexical * 0.40
            )

            scored_contexts.append(
                (
                    combined_score,
                    lexical,
                    context,
                )
            )

        # Highest combined score first.
        scored_contexts.sort(
            key=lambda item: item[0],
            reverse=True,
        )

        return [
            context
            for _, _, context in scored_contexts
        ]

    # ========================================================
    # RETRIEVE CONTEXT
    # ========================================================

    def retrieve_context(
        self,
        query: str,
        limit: int = 5,
    ) -> list[RetrievedContext]:

        # Retrieve more candidates first.
        results = self.retriever.search(
            query=query,
            limit=max(limit, 5),
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

        # ----------------------------------------------------
        # RERANK
        # ----------------------------------------------------

        contexts = self.rerank(
            query=query,
            contexts=contexts,
        )

        # Return only requested number.
        return contexts[:limit]

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        """
        Close the underlying retriever.
        """

        self.retriever.close()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    builder = ContextBuilder()

    try:

        query = input(
            "Enter your question: "
        )

        contexts = builder.retrieve_context(
            query=query,
            limit=3,
        )

        print()
        print("=" * 60)
        print("RERANKED RETRIEVED CONTEXT")
        print("=" * 60)

        if not contexts:

            print("No relevant context found.")

        for index, context in enumerate(
            contexts,
            start=1,
        ):

            lexical = builder.lexical_score(
                query,
                context.text,
            )

            print()
            print(f"--- Context {index} ---")
            print(
                f"Semantic Score: "
                f"{context.score:.6f}"
            )
            print(
                f"Lexical Score: "
                f"{lexical:.6f}"
            )
            print(
                f"Document ID: "
                f"{context.document_id}"
            )
            print(
                f"Chunk ID: "
                f"{context.chunk_id}"
            )
            print(
                f"Page: "
                f"{context.page_number}"
            )
            print(
                f"Text: "
                f"{context.text[:1000]}"
            )

    finally:

        builder.close()