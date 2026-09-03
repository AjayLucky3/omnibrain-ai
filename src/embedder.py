from sentence_transformers import SentenceTransformer

from src.models import TextChunk


MODEL_NAME = "all-MiniLM-L6-v2"


class TextEmbedder:
    """
    Converts document text into numerical vector representations.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model = SentenceTransformer(model_name)

    def embed_text(self, text: str) -> list[float]:
        """
        Convert one piece of text into an embedding vector.
        """

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
        )

        return embedding.tolist()

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        """
        Convert multiple pieces of text into embedding vectors.
        """

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
        )

        return embeddings.tolist()

    def embed_chunks(
        self,
        chunks: list[TextChunk],
    ) -> list[TextChunk]:
        """
        Generate embeddings for TextChunk objects.

        The original chunk metadata is preserved.
        """

        if not chunks:
            return []

        texts = [chunk.text for chunk in chunks]

        embeddings = self.embed_texts(texts)

        for chunk, embedding in zip(chunks, embeddings):
            chunk.embedding = embedding

        return chunks


if __name__ == "__main__":
    embedder = TextEmbedder()

    sample_chunks = [
        TextChunk(
            chunk_id="page_1_chunk_1",
            text=(
                "NovaTech Industries revenue increased "
                "to 180 million US dollars in 2025."
            ),
            page_number=1,
        ),
        TextChunk(
            chunk_id="page_1_chunk_2",
            text=(
                "Operating income reached 32 million "
                "US dollars in 2025."
            ),
            page_number=1,
        ),
    ]

    embedded_chunks = embedder.embed_chunks(sample_chunks)

    print("CHUNK EMBEDDING SUCCESSFUL")
    print("=" * 60)

    for chunk in embedded_chunks:
        print(f"Chunk ID: {chunk.chunk_id}")
        print(f"Page: {chunk.page_number}")
        print(f"Vector dimensions: {len(chunk.embedding)}")
        print(f"First 5 values: {chunk.embedding[:5]}")
        print()