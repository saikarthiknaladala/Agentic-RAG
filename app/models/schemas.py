"""Pydantic models for API request/response schemas."""
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class QueryRequest(BaseModel):
    """Request model for querying the RAG system."""
    query: str = Field(..., description="User query text")
    top_k: int = Field(default=5, description="Number of top chunks to retrieve", ge=1, le=20)
    similarity_threshold: float = Field(default=0.3, description="Minimum similarity score", ge=0.0, le=1.0)


class Citation(BaseModel):
    """Citation information for a retrieved chunk."""
    chunk_id: str
    source_file: str
    page_number: Optional[int] = None
    similarity_score: float
    text_preview: str


class QueryResponse(BaseModel):
    """Response model for RAG queries."""
    answer: str
    intent: str
    query_transformed: Optional[str] = None
    citations: List[Citation] = []
    metadata: Dict[str, Any] = {}


class IngestionStatus(BaseModel):
    """Response model for PDF ingestion."""
    status: str
    message: str
    files_processed: int
    chunks_created: int
    metadata: Dict[str, Any] = {}


class HealthCheck(BaseModel):
    """Health check response."""
    status: str
    ollama_connected: bool
    groq_connected: bool
    vector_store_initialized: bool
    total_documents: int
