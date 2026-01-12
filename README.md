# Agentic RAG System

An advanced Retrieval-Augmented Generation (RAG) pipeline with hybrid search, hallucination detection, and intelligent query processing.

## System Architecture

### Overview

This RAG system implements a sophisticated pipeline that combines semantic and keyword-based search with LLM-based query processing, answer generation, and validation.

```
┌─────────────────────────────────────────────────────────────────────────┐
│                          RAG Pipeline Flow                               │
└─────────────────────────────────────────────────────────────────────────┘

User Query
    │
    ▼
┌──────────────────────┐
│ 1. Query Refusal     │  ← LLM checks for PII, legal, medical content
│    Check             │
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ 2. Intent Detection  │  ← LLM classifies intent (knowledge/greeting/chitchat)
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ 3. Query Transform   │  ← LLM expands/clarifies query for better retrieval
└──────────────────────┘
    │
    ▼
┌──────────────────────────────────────────────┐
│ 4. Hybrid Search                             │
│                                              │
│  ┌─────────────────┐  ┌──────────────────┐ │
│  │ Semantic Search │  │ Keyword Search   │ │
│  │ (Embeddings +   │  │ (BM25)           │ │
│  │  Cosine Sim)    │  │                  │ │
│  └─────────────────┘  └──────────────────┘ │
│           │                    │            │
│           └────────┬───────────┘            │
│                    ▼                        │
│           ┌──────────────────┐             │
│           │ Score Fusion     │             │
│           │ (Weighted Avg)   │             │
│           └──────────────────┘             │
└──────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────┐
│ 5. Re-ranking        │  ← LLM scores relevance of each chunk
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ 6. Citation Filter   │  ← Remove chunks below similarity threshold
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ 7. Answer Generation │  ← LLM generates answer from top chunks
└──────────────────────┘
    │
    ▼
┌──────────────────────┐
│ 8. Hallucination     │  ← LLM validates answer against evidence
│    Detection         │
└──────────────────────┘
    │
    ▼
Final Answer + Citations
```

### Core Components

#### 1. PDF Processing & Chunking ([pdf_processor.py](app/services/pdf_processor.py), [chunking.py](app/utils/chunking.py))

**Text Extraction:**
- Uses PyPDF2 for PDF text extraction
- Cleans artifacts and normalizes Unicode characters
- Preserves page numbers for citation tracking

**Chunking Strategy:**
- **Chunk size**: 512 tokens (~400 words)
  - Balances context preservation with retrieval precision
  - Large enough for semantic meaning, small enough for relevance
- **Overlap**: 128 tokens (~100 words)
  - Prevents information loss at chunk boundaries
  - Ensures context continuity
- **Sentence-aware splitting**: Splits at sentence boundaries to maintain coherence
- **Metadata tracking**: Source file, page number, chunk index

#### 2. Custom Vector Store ([vector_store.py](app/services/vector_store.py))

**Implementation:**
- Custom in-memory vector database using NumPy
- No external dependencies (no Pinecone, Weaviate, etc.)
- Cosine similarity search via normalized vectors
- Persistent storage to disk (JSON + NumPy arrays)

**Features:**
- Efficient batch operations
- Automatic vector normalization
- Metadata association
- Incremental document addition

#### 3. Hybrid Search System ([search.py](app/services/search.py))

**Semantic Search:**
- Embedding-based similarity using Nomic embeddings
- Cosine similarity computation
- Configurable similarity threshold

**Keyword Search:**
- Custom BM25 implementation ([bm25.py](app/utils/bm25.py))
- Probabilistic ranking function
- Parameters: k1=1.5, b=0.75
- Term frequency and inverse document frequency scoring

**Hybrid Fusion:**
- Weighted combination of semantic and keyword scores
- Default weights: 70% semantic, 30% keyword
- Min-max score normalization
- Configurable weight adjustment

**Re-ranking:**
- LLM-based relevance scoring
- Combines hybrid score (60%) with LLM relevance (40%)
- Improves precision of top results

#### 4. LLM Service ([llm_service.py](app/services/llm_service.py))

**Groq + Ollama Integration:**
- **LLM Model**: llama-3.1-8b-instant (Groq)
- **Embedding Model**: nomic-embed-text (Ollama)

**LLM-Based Features:**

1. **Intent Detection**
   - Classifies queries into categories
   - Determines if knowledge base search is needed
   - Returns confidence scores

2. **Query Transformation**
   - Expands abbreviations
   - Adds relevant synonyms
   - Rephrases for clarity

3. **PII & Refusal Detection**
   - Identifies personal information
   - Detects legal/medical queries requiring disclaimers
   - Flags harmful requests

4. **Hallucination Detection**
   - Post-generation validation
   - Sentence-by-sentence evidence checking
   - Identifies unsupported claims

#### 5. RAG Pipeline Orchestrator ([rag_pipeline.py](app/services/rag_pipeline.py))

**Pipeline Stages:**

1. **Query Refusal Check**: Reject inappropriate queries
2. **Intent Detection**: Classify query intent
3. **Query Transformation**: Optimize for retrieval
4. **Hybrid Search**: Retrieve candidate chunks
5. **Re-ranking**: Improve ranking precision
6. **Citation Filtering**: Remove low-confidence chunks
7. **Answer Generation**: Generate contextual answer
8. **Hallucination Detection**: Validate answer
9. **Answer Shaping**: Format based on intent

**Answer Templates:**
- Knowledge query: Standard Q&A format
- List queries: Bullet-point format
- Comparison: Structured comparison format
- Strict mode: Enhanced for hallucination prevention

#### 6. FastAPI Application ([main.py](app/main.py))

**Endpoints:**

- `GET /`: Web UI
- `GET /health`: System health check
- `POST /api/ingest`: Upload and process PDFs
- `POST /api/query`: Query the knowledge base
- `DELETE /api/clear`: Clear knowledge base
- `GET /api/stats`: Get statistics

## Installation

### Prerequisites

1. **Python 3.8+**
2. **Groq API key**
3. **Ollama** with models:
   - `nomic-embed-text` (embeddings)

### Install Ollama

```bash
# Download from https://ollama.ai/download

# Pull required model
ollama pull nomic-embed-text
```

### Configure Groq

Set your Groq API key (PowerShell):

```bash
$env:GROQ_API_KEY="your_key_here"
```

### Setup Application

```bash
# Clone repository
git clone <repository-url>
cd agentic_rag_karthik

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## Running the Application

### 1. Set Groq API Key

```bash
$env:GROQ_API_KEY="your_key_here"
```

### 2. Start Ollama

Make sure Ollama is running:

```bash
ollama serve
```

### 3. Start the FastAPI Server

```bash
# From project root
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 4. Access the Web UI

Open your browser and navigate to:

```
http://localhost:8000
```

## Usage Guide

### 1. Ingest Documents

1. Click "Upload PDFs" and select one or more PDF files
2. Click "Ingest Documents" to process them
3. Wait for the ingestion to complete (progress shown in notifications)

### 2. Query the System

1. Type your question in the input box
2. Adjust parameters if needed:
   - **Top K**: Number of chunks to retrieve (1-20)
   - **Similarity Threshold**: Minimum confidence (0.0-1.0)
3. Press Enter or click the send button

### 3. View Results

The system will display:
- **Answer**: Generated response
- **Intent**: Detected query intent
- **Transformed Query**: Optimized version of your query
- **Citations**: Source documents with page numbers and similarity scores
- **Metadata**: Retrieval statistics

## Key Features

### 1. Citation Support

- Refuses to answer if evidence is insufficient (below threshold)
- Returns "insufficient evidence" message instead of hallucinating
- Provides citations with:
  - Source file name
  - Page number
  - Similarity score
  - Text preview

### 2. Answer Shaping

- Switches prompt templates based on intent
- Formats answers appropriately:
  - Lists for enumeration queries
  - Structured format for comparisons
  - Standard Q&A for general queries

### 3. Hallucination Detection

- Post-generation validation
- Scans answer sentences for unsupported claims
- Regenerates with stricter prompt if hallucinations detected
- Logs unsupported claims for debugging

### 4. Query Refusal Policies

- **PII Detection**: Refuses queries with personal information
- **Legal Disclaimer**: Warns against providing legal advice
- **Medical Disclaimer**: Warns against providing medical advice
- **Harmful Content**: Blocks unethical requests
- All checks use LLM prompts (no keyword matching)

### 5. Hybrid Search

- Combines semantic and keyword search
- Semantic search captures meaning and context
- Keyword search ensures exact term matching
- Fusion provides balanced results

## Technical Highlights

### No External Libraries for RAG/Search

- ✅ Custom vector store implementation (NumPy only)
- ✅ Custom BM25 implementation
- ✅ No LangChain, LlamaIndex, or similar frameworks
- ✅ No Pinecone, Weaviate, ChromaDB, etc.

### LLM-Based Processing

- ✅ Intent detection via LLM (no keyword matching)
- ✅ Query transformation via LLM
- ✅ Refusal policies via LLM (no regex patterns)
- ✅ Hallucination detection via LLM

### Architecture Decisions

**Chunking:**
- 512 tokens balances context vs precision
- 128-token overlap prevents information loss
- Sentence-aware splitting maintains coherence

**Search Fusion:**
- 70/30 semantic/keyword weights empirically effective
- Re-ranking with LLM improves top-5 precision
- Min-max normalization ensures fair score combination

**Vector Store:**
- In-memory for speed
- Disk persistence for durability
- NumPy for efficient computation
- Normalized vectors enable cosine similarity via dot product

## Project Structure

```
agentic_rag_karthik/
├── app/
│   ├── __init__.py
│   ├── main.py                 # FastAPI application
│   ├── models/
│   │   ├── __init__.py
│   │   └── schemas.py          # Pydantic models
│   ├── services/
│   │   ├── __init__.py
│   │   ├── pdf_processor.py    # PDF extraction & chunking
│   │   ├── vector_store.py     # Custom vector database
│   │   ├── llm_service.py      # Groq + Ollama integration
│   │   ├── search.py           # Hybrid search
│   │   └── rag_pipeline.py     # RAG orchestrator
│   └── utils/
│       ├── __init__.py
│       ├── chunking.py         # Text chunking logic
│       └── bm25.py             # BM25 implementation
├── static/
│   ├── index.html              # Web UI
│   ├── style.css               # Styles
│   └── app.js                  # Frontend logic
├── data/                       # PDF storage (created at runtime)
├── vector_store/               # Vector DB persistence (created at runtime)
├── requirements.txt
├── .gitignore
└── README.md
```

## API Reference

### POST /api/ingest

Ingest PDF documents.

**Request:**
- Content-Type: `multipart/form-data`
- Body: List of PDF files

**Response:**
```json
{
  "status": "success",
  "message": "Successfully processed 2 files",
  "files_processed": 2,
  "chunks_created": 45,
  "metadata": {
    "filenames": ["doc1.pdf", "doc2.pdf"],
    "total_documents": 45
  }
}
```

### POST /api/query

Query the knowledge base.

**Request:**
```json
{
  "query": "What is machine learning?",
  "top_k": 5,
  "similarity_threshold": 0.3
}
```

**Response:**
```json
{
  "answer": "Machine learning is...",
  "intent": "knowledge_query",
  "query_transformed": "What is machine learning? ML definition artificial intelligence",
  "citations": [
    {
      "chunk_id": "12",
      "source_file": "ml_intro.pdf",
      "page_number": 3,
      "similarity_score": 0.87,
      "text_preview": "Machine learning is a subset of artificial intelligence..."
    }
  ],
  "metadata": {
    "chunks_retrieved": 10,
    "chunks_used": 5,
    "hallucination_check": {
      "is_supported": true,
      "unsupported_claims": [],
      "confidence": 0.95
    }
  }
}
```

## Libraries and Dependencies

### Core Dependencies

- **FastAPI** (0.109.0): Web framework
- **Uvicorn** (0.27.0): ASGI server
- **PyPDF2** (3.0.1): PDF text extraction
- **NumPy** (1.26.3): Vector operations
- **Scikit-learn** (1.4.0): Utility functions
- **Pydantic** (2.5.3): Data validation
- **Requests** (2.31.0): HTTP client for Groq and Ollama

### External Services

- **Groq**: LLM inference API
  - Website: https://console.groq.com
  - Model used: `llama-3.1-8b-instant`
- **Ollama**: Embedding inference engine
  - Website: https://ollama.ai
  - Models used: `nomic-embed-text`

## Performance Considerations

### Chunking

- **Token estimation**: ~1.3 tokens/word approximation
- **Processing speed**: ~1000 chunks/minute
- **Memory usage**: Minimal (streaming processing)

### Vector Operations

- **Embedding generation**: ~100 chunks/minute (depends on Ollama)
- **Search latency**: <100ms for 10,000 chunks
- **Memory**: ~4KB per chunk (embedding + metadata)

### LLM Calls

- **Intent detection**: ~1-2s
- **Query transformation**: ~1-2s
- **Answer generation**: ~3-5s
- **Hallucination check**: ~2-3s

**Total query latency**: 10-15 seconds (can be optimized with caching)

## Future Enhancements

- [ ] Batch embedding generation for faster ingestion
- [ ] Caching layer for repeated queries
- [ ] Multi-query retrieval for complex questions
- [ ] Document update/deletion support
- [ ] Conversation history tracking
- [ ] Fine-tuned re-ranking model
- [ ] Streaming response generation
- [ ] Advanced visualization of search results

## Troubleshooting

### Ollama Connection Issues

**Problem**: "Ollama not connected" error

**Solution**:
```bash
# Check if Ollama is running
ollama list

# Start Ollama if not running
ollama serve

# Verify models are installed
ollama list | grep nomic-embed-text
```

### Groq Connection Issues

**Problem**: "Groq not connected" error

**Solution**:
- Ensure `GROQ_API_KEY` is set in your environment
- Confirm the model `llama-3.1-8b-instant` is available in your Groq account

### Empty Responses

**Problem**: "Insufficient evidence" responses

**Solution**:
- Lower the similarity threshold (try 0.2 instead of 0.3)
- Increase top_k value
- Check if documents were properly ingested
- Verify PDF text extraction worked

### Slow Performance

**Problem**: Queries take too long

**Solution**:
- Reduce top_k value
- Check Ollama resource usage
- Consider using GPU acceleration for Ollama
- Reduce number of LLM calls (disable re-ranking)

## License

This project is for educational and evaluation purposes.

## Author

Developed as a technical assessment for advanced RAG pipeline implementation.

---

**Note**: This system is designed for demonstration purposes. For production use, consider adding:
- Authentication and authorization
- Rate limiting
- Comprehensive error handling
- Monitoring and logging
- Database backend for scalability
- Load balancing for multiple instances
