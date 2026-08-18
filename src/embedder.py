from sentence_transformers import SentenceTransformer


MODEL_NAME = "all-MiniLM-L6-v2"


class TextEmbedder:
    """
    Converts text into numerical vector representations.
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


if __name__ == "__main__":
    embedder = TextEmbedder()

    sample_text = (
        "NovaTech Industries revenue increased "
        "to 180 million US dollars in 2025."
    )

    embedding = embedder.embed_text(sample_text)

    print("EMBEDDING SUCCESSFUL")
    print("=" * 60)
    print(f"Vector dimensions: {len(embedding)}")
    print(f"First 10 values: {embedding[:10]}")