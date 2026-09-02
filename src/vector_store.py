from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from models import TextChunk


COLLECTION_NAME = "omnibrain_text"

VECTOR_SIZE = 384

STORAGE_PATH = Path("data/qdrant")


class VectorStore:
    """
    Local Qdrant vector database for OmniBrain.
    """

    def __init__(
        self,
        storage_path: str | Path = STORAGE_PATH,
        collection_name: str = COLLECTION_NAME,
    ):
        self.storage_path = Path(storage_path)
        self.collection_name = collection_name

        self.storage_path.mkdir(
            parents=True,
            exist_ok=True,
        )

        self.client = QdrantClient(
            path=str(self.storage_path),
        )

        self._create_collection()

    def _create_collection(self) -> None:
        """
        Create the collection if it doesn't already exist.
        """

        collections = self.client.get_collections()

        collection_names = [
            collection.name
            for collection in collections.collections
        ]

        if self.collection_name not in collection_names:
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=VECTOR_SIZE,
                    distance=Distance.COSINE,
                ),
            )

    def _create_point_id(
        self,
        document_id: str,
        chunk_id: str,
    ) -> str:
        """
        Create a deterministic UUID for a document chunk.

        The same document and chunk will always produce
        the same UUID.
        """

        unique_value = f"{document_id}:{chunk_id}"

        return str(
            uuid5(
                NAMESPACE_URL,
                unique_value,
            )
        )

    def add_chunks(
        self,
        chunks: list[TextChunk],
        document_id: str,
    ) -> None:
        """
        Store embedded text chunks in Qdrant.
        """

        points = []

        for chunk in chunks:
            if chunk.embedding is None:
                raise ValueError(
                    f"Chunk '{chunk.chunk_id}' has no embedding."
                )

            point_id = self._create_point_id(
                document_id=document_id,
                chunk_id=chunk.chunk_id,
            )

            points.append(
                PointStruct(
                    id=point_id,
                    vector=chunk.embedding,
                    payload={
                        "document_id": document_id,
                        "chunk_id": chunk.chunk_id,
                        "text": chunk.text,
                        "page_number": chunk.page_number,
                    },
                )
            )

        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
            )

    def search(
        self,
        query_vector: list[float],
        limit: int = 5,
    ):
        """
        Search for the most similar chunks.
        """

        results = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector,
            limit=limit,
        )

        return results.points

    def close(self) -> None:
        """
        Explicitly close the Qdrant client.
        """

        self.client.close()


if __name__ == "__main__":
    print("VECTOR STORE INITIALIZED")
    print("=" * 60)

    store = VectorStore()

    print(f"Collection: {store.collection_name}")
    print(f"Vector size: {VECTOR_SIZE}")
    print(f"Storage: {STORAGE_PATH}")

    store.close()