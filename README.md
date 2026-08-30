# Atlas RAG - Document Intelligence

Atlas RAG is a deployment-ready document question-answering application. Upload PDFs, index them with Gemini embeddings and ChromaDB, then ask natural-language questions that receive grounded answers and page-level source citations.

## Architecture

`PDF upload -> PyMuPDF extraction -> LangChain chunking -> Gemini embeddings -> ChromaDB -> semantic retrieval -> Gemini grounded answer -> citations`

## Features

- Multi-document PDF upload and persistent ChromaDB vector storage.
- LangChain orchestration and recursive text splitting.
- Gemini embeddings and generation via `langchain-google-genai`.
- Document-scoped questions, relevance threshold guardrails, and cited answers.
- Production-style Streamlit chat UI plus FastAPI endpoints.
- Dockerfile and Streamlit deployment configuration.

## Configure

Create `.env` in the project root from `.env.example` and set your Gemini API key:

```env
GEMINI_API_KEY=your_key_here
GEMINI_CHAT_MODEL=gemini-3.6-flash
GEMINI_EMBEDDING_MODEL=models/gemini-embedding-001
```

Never put the key in source code or commit `.env`.

## Run locally

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m streamlit run streamlit_app.py
```

Open the URL Streamlit prints, typically `http://localhost:8501`.

To run the API instead:

```powershell
.\.venv\Scripts\python.exe -m uvicorn app.main:app --reload
```

Open API documentation at `http://127.0.0.1:8000/docs`.

## Deployment

The supplied `Dockerfile` starts Streamlit on port 8501. Set `GEMINI_API_KEY` as a secret/environment variable on your hosting platform - not in the image or repository.
