"""
Simple runner script for the RAG application.
"""
import uvicorn

if __name__ == "__main__":
    print("=" * 60)
    print("Starting Agentic RAG System")
    print("=" * 60)
    print("\nMake sure Groq is configured and Ollama is running with embeddings:")
    print("  - GROQ_API_KEY is set")
    print("  - Groq model: llama-3.1-8b-instant")
    print("  - Ollama model: nomic-embed-text")
    print("\nAccess the application at: http://localhost:8000")
    print("=" * 60)
    print()

    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )
