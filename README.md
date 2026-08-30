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

