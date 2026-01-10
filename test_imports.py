"""
Test script to verify all imports work correctly.
"""
import sys

def test_imports():
    print("Testing imports...")
    print("-" * 60)

    try:
        print("✓ Testing standard library imports...")
        import json
        import logging
        import os
        from pathlib import Path
        print("  ✓ Standard library OK")

        print("\n✓ Testing third-party imports...")
        import fastapi
        print(f"  ✓ FastAPI {fastapi.__version__}")

        import uvicorn
        print(f"  ✓ Uvicorn OK")

        import PyPDF2
        print(f"  ✓ PyPDF2 {PyPDF2.__version__}")

        import numpy as np
        print(f"  ✓ NumPy {np.__version__}")

        import sklearn
        print(f"  ✓ Scikit-learn {sklearn.__version__}")

        import pydantic
        print(f"  ✓ Pydantic {pydantic.__version__}")

        import requests
        print(f"  ✓ Requests {requests.__version__}")

        print("\n✓ Testing application imports...")
        from app.models.schemas import QueryRequest, QueryResponse
        print("  ✓ Models OK")

        from app.utils.chunking import TextChunker
        from app.utils.bm25 import BM25
        print("  ✓ Utils OK")

        from app.services.pdf_processor import PDFProcessor
        from app.services.vector_store import VectorStore
        from app.services.llm_service import OllamaService
        from app.services.search import SearchService
        from app.services.rag_pipeline import RAGPipeline
        print("  ✓ Services OK")

        from app.main import app
        print("  ✓ Main application OK")

        print("\n" + "=" * 60)
        print("✓ All imports successful!")
        print("=" * 60)
        return True

    except ImportError as e:
        print(f"\n✗ Import error: {e}")
        print("\nMake sure you have:")
        print("  1. Activated the virtual environment")
        print("  2. Installed dependencies: pip install -r requirements.txt")
        return False
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        return False


def test_directory_structure():
    print("\nTesting directory structure...")
    print("-" * 60)

    required_dirs = [
        "app",
        "app/models",
        "app/services",
        "app/utils",
        "static",
        "data",
        "vector_store"
    ]

    required_files = [
        "requirements.txt",
        "README.md",
        "run.py",
        "app/__init__.py",
        "app/main.py",
        "app/models/__init__.py",
        "app/models/schemas.py",
        "app/services/__init__.py",
        "app/services/pdf_processor.py",
        "app/services/vector_store.py",
        "app/services/llm_service.py",
        "app/services/search.py",
        "app/services/rag_pipeline.py",
        "app/utils/__init__.py",
        "app/utils/chunking.py",
        "app/utils/bm25.py",
        "static/index.html",
        "static/style.css",
        "static/app.js"
    ]

    all_ok = True

    for dir_path in required_dirs:
        if os.path.isdir(dir_path):
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ Missing directory: {dir_path}/")
            all_ok = False

    for file_path in required_files:
        if os.path.isfile(file_path):
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ Missing file: {file_path}")
            all_ok = False

    if all_ok:
        print("\n✓ All required files and directories present")
    else:
        print("\n✗ Some files or directories are missing")

    return all_ok


def check_ollama():
    print("\nChecking Ollama connection...")
    print("-" * 60)

    try:
        import requests
        response = requests.get("http://localhost:11434/api/tags", timeout=5)

        if response.status_code == 200:
            print("  ✓ Ollama is running")

            tags = response.json()
            models = [model['name'] for model in tags.get('models', [])]

            required_models = ['nomic-embed-text']
            for model in required_models:
                # Check if model name starts with required model (handles versions)
                if any(m.startswith(model) for m in models):
                    print(f"  ✓ Model '{model}' is available")
                else:
                    print(f"  ✗ Model '{model}' not found. Run: ollama pull {model}")

            return True
        else:
            print("  ✗ Ollama responded with error")
            return False

    except requests.exceptions.ConnectionError:
        print("  ✗ Cannot connect to Ollama")
        print("  → Make sure Ollama is running: ollama serve")
        return False
    except Exception as e:
        print(f"  ✗ Error checking Ollama: {e}")
        return False


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("AGENTIC RAG SYSTEM - SETUP VERIFICATION")
    print("=" * 60 + "\n")

    structure_ok = test_directory_structure()
    imports_ok = test_imports()
    ollama_ok = check_ollama()

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Directory Structure: {'✓ PASS' if structure_ok else '✗ FAIL'}")
    print(f"Python Imports:      {'✓ PASS' if imports_ok else '✗ FAIL'}")
    print(f"Ollama Connection:   {'✓ PASS' if ollama_ok else '✗ FAIL'}")

    if structure_ok and imports_ok and ollama_ok:
        print("\n✓ System is ready! Run: python run.py")
    else:
        print("\n✗ Please fix the issues above before running the application")

    print("=" * 60 + "\n")
