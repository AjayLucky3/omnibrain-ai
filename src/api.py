from pathlib import Path
from tempfile import NamedTemporaryFile

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.document_ingestion import ingest_document
from src.embedder import TextEmbedder
from src.rag_pipeline import RAGPipeline


# ============================================================
# CONFIGURATION
# ============================================================

API_TITLE = "OmniBrain AI"
API_VERSION = "1.0.0"

MAX_FILE_SIZE = 20 * 1024 * 1024

ALLOWED_EXTENSIONS = {
    ".pdf",
}


# ============================================================
# REQUEST / RESPONSE MODELS
# ============================================================

class ChatRequest(BaseModel):
    question: str
    limit: int = 5


class SourceResponse(BaseModel):
    document_id: str
    chunk_id: str
    page_number: int
    score: float
    text: str


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceResponse]


class UploadResponse(BaseModel):
    document_id: str
    filename: str
    pages: int
    text_chunks: int
    images: int
    indexed_chunks: int
    message: str


class DocumentResponse(BaseModel):
    document_id: str
    filename: str | None
    pages: int
    chunks: int


class DocumentListResponse(BaseModel):
    documents: list[DocumentResponse]
    total_documents: int
    total_chunks: int


# ============================================================
# FASTAPI APPLICATION
# ============================================================

app = FastAPI(
    title=API_TITLE,
    description=(
        "OmniBrain AI document question-answering API "
        "using RAG, Qdrant, embeddings, and Ollama."
    ),
    version=API_VERSION,
)


# ============================================================
# CORS
# ============================================================

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ============================================================
# RAG PIPELINE
# ============================================================

pipeline = RAGPipeline()


# ============================================================
# VECTOR STORE
# ============================================================

def get_vector_store():
    """
    Reuse the VectorStore already owned by the RAG pipeline.

    This is important because local Qdrant storage does not
    allow multiple QdrantClient instances to access the same
    storage folder simultaneously.
    """

    return (
        pipeline
        .context_builder
        .retriever
        .vector_store
    )


# ============================================================
# ROOT
# ============================================================

@app.get("/")
def root():
    return {
        "name": API_TITLE,
        "status": "running",
        "version": API_VERSION,
    }


# ============================================================
# HEALTH CHECK
# ============================================================

@app.get("/health")
def health():
    return {
        "status": "healthy",
    }


# ============================================================
# CHAT
# ============================================================

@app.post(
    "/api/v1/chat",
    response_model=ChatResponse,
)
def chat(request: ChatRequest):

    question = request.question.strip()

    if not question:

        raise HTTPException(
            status_code=400,
            detail="Question cannot be empty.",
        )

    if request.limit < 1 or request.limit > 20:

        raise HTTPException(
            status_code=400,
            detail="Limit must be between 1 and 20.",
        )

    try:

        original_limit = pipeline.retrieval_limit

        pipeline.retrieval_limit = request.limit

        try:

            result = pipeline.ask(
                question=question,
            )

        finally:

            pipeline.retrieval_limit = original_limit

        sources = [
            SourceResponse(
                document_id=context.document_id,
                chunk_id=context.chunk_id,
                page_number=context.page_number,
                score=context.score,
                text=context.text,
            )
            for context in result.contexts
        ]

        return ChatResponse(
            answer=result.answer,
            sources=sources,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=str(exc),
        )


# ============================================================
# PDF UPLOAD + INDEXING
# ============================================================

@app.post(
    "/api/v1/upload",
    response_model=UploadResponse,
)
async def upload_document(
    file: UploadFile = File(...),
):

    # --------------------------------------------------------
    # Validate filename
    # --------------------------------------------------------

    if not file.filename:

        raise HTTPException(
            status_code=400,
            detail="No filename provided.",
        )

    filename = Path(file.filename).name

    extension = Path(filename).suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:

        raise HTTPException(
            status_code=400,
            detail="Only PDF files are supported.",
        )

    temporary_path = None

    try:

        # ----------------------------------------------------
        # Read uploaded file
        # ----------------------------------------------------

        file_data = await file.read()

        if not file_data:

            raise HTTPException(
                status_code=400,
                detail="Uploaded file is empty.",
            )

        if len(file_data) > MAX_FILE_SIZE:

            raise HTTPException(
                status_code=413,
                detail="PDF file size cannot exceed 20 MB.",
            )

        # ----------------------------------------------------
        # Save temporary PDF
        # ----------------------------------------------------

        with NamedTemporaryFile(
            suffix=".pdf",
            delete=False,
        ) as temporary_file:

            temporary_file.write(file_data)

            temporary_path = Path(
                temporary_file.name
            )

        # ----------------------------------------------------
        # DOCUMENT INGESTION
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("DOCUMENT UPLOAD")
        print("=" * 60)

        print(
            f"Filename: {filename}"
        )

        document = ingest_document(
            pdf_path=temporary_path,
        )

        # Preserve original uploaded filename.
        document.filename = filename

        print(
            f"Pages extracted: "
            f"{len(document.pages)}"
        )

        print(
            f"Text chunks created: "
            f"{len(document.text_chunks)}"
        )

        print(
            f"Images extracted: "
            f"{len(document.images)}"
        )

        # ----------------------------------------------------
        # EMBEDDINGS
        # ----------------------------------------------------

        print()
        print("Generating embeddings...")

        embedder = TextEmbedder()

        embedded_chunks = embedder.embed_chunks(
            document.text_chunks
        )

        print(
            f"Generated embeddings for "
            f"{len(embedded_chunks)} chunks."
        )

        # ----------------------------------------------------
        # VECTOR STORE
        # ----------------------------------------------------

        print()
        print("Storing vectors in Qdrant...")

        vector_store = get_vector_store()

        vector_store.add_chunks(
            chunks=embedded_chunks,
            document_id=document.document_id,
            filename=document.filename,
        )

        print(
            "Vectors stored successfully."
        )

        # ----------------------------------------------------
        # RESPONSE
        # ----------------------------------------------------

        print()
        print("=" * 60)
        print("DOCUMENT UPLOAD SUCCESSFUL")
        print("=" * 60)

        return UploadResponse(
            document_id=document.document_id,
            filename=document.filename,
            pages=len(document.pages),
            text_chunks=len(document.text_chunks),
            images=len(document.images),
            indexed_chunks=len(embedded_chunks),
            message=(
                "Document uploaded, processed, "
                "embedded, and indexed successfully."
            ),
        )

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Document processing failed: {exc}",
        )

    finally:

        # ----------------------------------------------------
        # CLEANUP TEMPORARY FILE
        # ----------------------------------------------------

        if temporary_path is not None:

            try:

                temporary_path.unlink(
                    missing_ok=True
                )

            except Exception:

                pass


# ============================================================
# LIST DOCUMENTS
# ============================================================

@app.get(
    "/api/v1/documents",
    response_model=DocumentListResponse,
)
def list_documents():

    try:

        vector_store = get_vector_store()

        documents = vector_store.list_documents()

        total_chunks = sum(
            document["chunks"]
            for document in documents
        )

        formatted_documents = [
            DocumentResponse(
                document_id=document["document_id"],
                filename=document["filename"],
                pages=document["pages"],
                chunks=document["chunks"],
            )
            for document in documents
        ]

        return DocumentListResponse(
            documents=formatted_documents,
            total_documents=len(
                formatted_documents
            ),
            total_chunks=total_chunks,
        )

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not list documents: {exc}",
        )


# ============================================================
# DELETE DOCUMENT
# ============================================================

@app.delete(
    "/api/v1/documents/{document_id}",
)
def delete_document(
    document_id: str,
):

    try:

        vector_store = get_vector_store()

        documents = vector_store.list_documents()

        document_exists = any(
            document["document_id"] == document_id
            for document in documents
        )

        if not document_exists:

            raise HTTPException(
                status_code=404,
                detail="Document not found.",
            )

        vector_store.delete_document(
            document_id=document_id
        )

        return {
            "message": "Document deleted successfully.",
            "document_id": document_id,
        }

    except HTTPException:

        raise

    except Exception as exc:

        raise HTTPException(
            status_code=500,
            detail=f"Could not delete document: {exc}",
        )