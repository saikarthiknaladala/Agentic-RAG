# Quick Setup Instructions

## Prerequisites

1. **Python 3.8 or higher**
2. **Groq API key**
3. **Ollama installed and running** (for embeddings)

## Step-by-Step Setup

### 1. Install Ollama

Download and install Ollama from: https://ollama.ai/download

### 2. Configure Groq

Set your Groq API key (PowerShell):

```bash
$env:GROQ_API_KEY="your_key_here"
```

Default model: `llama-3.1-8b-instant`

### 3. Pull Required Ollama Model

Open a terminal and run:

```bash
ollama pull nomic-embed-text
```

Wait for both models to download completely.

### 4. Start Ollama Server

```bash
ollama serve
```

Keep this terminal window open.

### 5. Setup Python Environment

Open a new terminal in the project directory:

```bash
# Create virtual environment
python -m venv venv

# Activate it
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 6. Run the Application

```bash
# Option 1: Using the run script
python run.py

# Option 2: Using uvicorn directly
python -m uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

### 7. Access the Web UI

Open your browser and go to:

```
http://localhost:8000
```

## Quick Test

1. Upload a PDF document using the "Upload PDFs" button
2. Click "Ingest Documents" and wait for processing
3. Ask a question in the chat input
4. View the answer with citations

## Common Issues

### Ollama Not Connected

**Error**: Red status indicator showing "Ollama not connected"

**Fix**: Make sure Ollama is running (`ollama serve`) and the embedding model is pulled.

### Groq Not Connected

**Error**: "Groq not connected"

**Fix**: Ensure `GROQ_API_KEY` is set and the model `llama-3.1-8b-instant` is available.

### Import Errors

**Error**: `ModuleNotFoundError`

**Fix**: Make sure virtual environment is activated and dependencies are installed.

### No Text Extracted from PDF

**Error**: "No text could be extracted from the PDFs"

**Fix**: Ensure PDFs contain extractable text (not scanned images).

## System Requirements

- **RAM**: Minimum 8GB (16GB recommended for larger documents)
- **Storage**: ~5GB for Ollama models
- **CPU**: Modern multi-core processor (GPU optional but recommended)

## Next Steps

- Read the full README.md for detailed documentation
- Explore the code in the `app/` directory
- Check the API documentation at http://localhost:8000/docs
