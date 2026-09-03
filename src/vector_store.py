from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams

from src.models import TextChunk


COLLECTION_NAME = "omnibrain_text"

VECTOR_SIZE = 384

BASE_DIR = Path(__file__).resolve().parent.parent
STORAGE_PATH = BASE_DIR / "data" / "qdrant"


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

    # ========================================================
    # COLLECTION
    # ========================================================

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

    # ========================================================
    # POINT ID
    # ========================================================

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

    # ========================================================
    # ADD CHUNKS
    # ========================================================

    def add_chunks(
        self,
        chunks: list[TextChunk],
        document_id: str,
        filename: str | None = None,
    ) -> None:
        """
        Store embedded text chunks in Qdrant.

        Each chunk stores:
            - document ID
            - filename
            - chunk ID
            - text
            - page number
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
                        "filename": filename,
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

    # ========================================================
    # SEARCH
    # ========================================================

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

    # ========================================================
    # DOCUMENTS
    # ========================================================

    def list_documents(self) -> list[dict]:
        """
        Return a summary of all indexed documents.
        """

        documents = {}

        offset = None

        while True:

            records, offset = self.client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for record in records:

                payload = record.payload or {}

                document_id = payload.get(
                    "document_id"
                )

                if not document_id:
                    continue

                if document_id not in documents:

                    documents[document_id] = {
                        "document_id": document_id,
                        "filename": payload.get(
                            "filename"
                        ),
                        "pages": set(),
                        "chunks": 0,
                    }

                documents[document_id]["chunks"] += 1

                page_number = payload.get(
                    "page_number"
                )

                if page_number is not None:
                    documents[document_id]["pages"].add(
                        page_number
                    )

            if offset is None:
                break

        result = []

        for document in documents.values():

            result.append(
                {
                    "document_id": document[
                        "document_id"
                    ],
                    "filename": document[
                        "filename"
                    ],
                    "pages": len(
                        document["pages"]
                    ),
                    "chunks": document[
                        "chunks"
                    ],
                }
            )

        result.sort(
            key=lambda item: item["filename"] or ""
        )

        return result

    # ========================================================
    # DELETE DOCUMENT
    # ========================================================

    def delete_document(
        self,
        document_id: str,
    ) -> None:
        """
        Delete every chunk belonging to a document.
        """

        from qdrant_client.models import (
            FieldCondition,
            Filter,
            MatchValue,
        )

        self.client.delete(
            collection_name=self.collection_name,
            points_selector=Filter(
                must=[
                    FieldCondition(
                        key="document_id",
                        match=MatchValue(
                            value=document_id
                        ),
                    )
                ]
            ),
        )

    # ========================================================
    # CLOSE
    # ========================================================

    def close(self) -> None:
        """
        Explicitly close the Qdrant client.
        """

        self.client.close()


# ============================================================
# TEST
# ============================================================

if __name__ == "__main__":

    print("VECTOR STORE INITIALIZED")
    print("=" * 60)

    store = VectorStore()

    print(
        f"Collection: "
        f"{store.collection_name}"
    )

    print(
        f"Vector size: "
        f"{VECTOR_SIZE}"
    )

    print(
        f"Storage: "
        f"{STORAGE_PATH}"
    )

    documents = store.list_documents()

    print(
        f"\nDocuments: "
        f"{len(documents)}"
    )

    total_chunks = sum(
        document["chunks"]
        for document in documents
    )

    print(
        f"Chunks: "
        f"{total_chunks}"
    )

    print()
    print("INDEXED DOCUMENTS")
    print("-" * 60)

    if not documents:

        print("No documents indexed.")

    else:

        for document in documents:

            print(
                f"Document ID: "
                f"{document['document_id']}"
            )

            print(
                f"Filename: "
                f"{document['filename']}"
            )

            print(
                f"Pages: "
                f"{document['pages']}"
            )

            print(
                f"Chunks: "
                f"{document['chunks']}"
            )

            print()

    store.close()