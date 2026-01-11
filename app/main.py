"""FastAPI application for RAG pipeline."""
import logging
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from typing import List
import os

from app.models.schemas import (
    QueryRequest,
    QueryResponse,
    IngestionStatus,
    HealthCheck
)
from app.services.pdf_processor import PDFProcessor
from app.services.vector_store import VectorStore
from app.services.llm_service import OllamaService
from app.services.search import SearchService
from app.services.rag_pipeline import RAGPipeline


# Load environment variables from .env if present.
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Initialize FastAPI app
app = FastAPI(
    title="Agentic RAG System",
    description="Advanced RAG pipeline with hybrid search, hallucination detection, and citation support",
    version="1.0.0"
)


# CORS middleware for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Global instances
vector_store: VectorStore = None
llm_service: OllamaService = None
search_service: SearchService = None
rag_pipeline: RAGPipeline = None
pdf_processor: PDFProcessor = None


@app.on_event("startup")
async def startup_event():
    """Initialize services on startup."""
    global vector_store, llm_service, search_service, rag_pipeline, pdf_processor

    logger.info("Initializing RAG system...")

    # Initialize components
    vector_store = VectorStore(persist_dir="vector_store")
    llm_service = OllamaService()
    search_service = SearchService(vector_store, llm_service)
    rag_pipeline = RAGPipeline(llm_service, search_service)
    pdf_processor = PDFProcessor()

    # Check Groq + Ollama connectivity
    ollama_ok = llm_service.check_ollama()
    groq_ok = llm_service.check_groq_model()

    if ollama_ok and groq_ok:
        logger.info("Successfully connected to Groq and Ollama")
    else:
        logger.warning(
            "LLM or embeddings not ready (Groq: %s, Ollama: %s)",
            groq_ok,
            ollama_ok
        )

    logger.info(f"System initialized with {vector_store.size()} documents in vector store")


@app.on_event("shutdown")
async def shutdown_event():
    """Save data on shutdown."""
    logger.info("Shutting down...")
    if vector_store:
        vector_store.save()
    logger.info("Shutdown complete")


# Mount static files for UI
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=FileResponse)
async def read_root():
    """Serve the main UI page."""
    return FileResponse("static/index.html")


@app.get("/health", response_model=HealthCheck)
async def health_check():
    """
    Health check endpoint.

    Returns system status and component health.
    """
    ollama_connected = llm_service.check_ollama() if llm_service else False
    groq_connected = llm_service.check_groq_model() if llm_service else False

    return HealthCheck(
        status="healthy" if (ollama_connected and groq_connected) else "degraded",
        ollama_connected=ollama_connected,
        groq_connected=groq_connected,
        vector_store_initialized=vector_store is not None,
        total_documents=vector_store.size() if vector_store else 0
    )


@app.post("/api/ingest", response_model=IngestionStatus)
async def ingest_pdfs(files: List[UploadFile] = File(...)):
    """
    Ingest PDF files into the knowledge base.

    Args:
        files: List of PDF files to ingest

    Returns:
        Ingestion status with statistics
    """
    if not files:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No files provided"
        )

    try:
        # Validate file types
        for file in files:
            if not file.filename.endswith('.pdf'):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {file.filename} is not a PDF"
                )

        # Read PDF files
        pdf_files = []
        for file in files:
            content = await file.read()
            pdf_files.append((content, file.filename))

        logger.info(f"Processing {len(pdf_files)} PDF files...")

        # Process PDFs and extract chunks
        chunks = pdf_processor.process_multiple_pdfs(pdf_files)

        if not chunks:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="No text could be extracted from the PDFs"
            )

        # Generate embeddings
        logger.info(f"Generating embeddings for {len(chunks)} chunks...")
        texts = [chunk[0] for chunk in chunks]
        metadata = [chunk[1] for chunk in chunks]

        embeddings = llm_service.generate_embeddings_batch(texts)

        # Add to vector store and search index
        search_service.add_documents(texts, embeddings, metadata)

        # Save to disk
        vector_store.save()

        logger.info(f"Successfully ingested {len(chunks)} chunks from {len(pdf_files)} files")

        return IngestionStatus(
            status="success",
            message=f"Successfully processed {len(pdf_files)} files",
            files_processed=len(pdf_files),
            chunks_created=len(chunks),
            metadata={
                "filenames": [f[1] for f in pdf_files],
                "total_documents": vector_store.size()
            }
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error during ingestion: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing files: {str(e)}"
        )


@app.post("/api/query", response_model=QueryResponse)
async def query_knowledge_base(request: QueryRequest):
    """
    Query the knowledge base.

    Args:
        request: Query request with query text and parameters

    Returns:
        Query response with answer and citations
    """
    if not request.query or not request.query.strip():
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Query cannot be empty"
        )

    # Check if vector store has documents
    if vector_store.size() == 0:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="No documents in knowledge base. Please ingest PDFs first."
        )

    try:
        logger.info(f"Processing query: {request.query}")

        # Process query through RAG pipeline
        response = rag_pipeline.process_query(
            query=request.query,
            top_k=request.top_k,
            similarity_threshold=request.similarity_threshold
        )

        return response

    except Exception as e:
        logger.error(f"Error processing query: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error processing query: {str(e)}"
        )


@app.delete("/api/clear")
async def clear_knowledge_base():
    """
    Clear all documents from the knowledge base.

    Returns:
        Success message
    """
    try:
        vector_store.clear()
        search_service._rebuild_bm25_index()

        logger.info("Knowledge base cleared")

        return {
            "status": "success",
            "message": "Knowledge base cleared successfully"
        }

    except Exception as e:
        logger.error(f"Error clearing knowledge base: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error clearing knowledge base: {str(e)}"
        )


@app.get("/api/stats")
async def get_stats():
    """
    Get knowledge base statistics.

    Returns:
        Statistics about the knowledge base
    """
    return {
        "total_documents": vector_store.size(),
        "embedding_dimension": vector_store.embedding_dim,
        "source_files": list(set([
            meta.get("source_file", "unknown")
            for meta in vector_store.metadata
        ]))
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
