import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Settings:
    gemini_api_key: str | None = os.getenv("GEMINI_API_KEY")
    chat_model: str = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.6-flash")
    embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "models/gemini-embedding-001")
    data_dir: Path = Path(os.getenv("RAG_DATA_DIR", "data"))
    collection_name: str = os.getenv("RAG_COLLECTION_NAME", "gemini_document_qa")
    chunk_size: int = int(os.getenv("RAG_CHUNK_SIZE", "1000"))
    chunk_overlap: int = int(os.getenv("RAG_CHUNK_OVERLAP", "200"))
    retrieval_k: int = int(os.getenv("RAG_RETRIEVAL_K", "5"))
    min_relevance: float = float(os.getenv("RAG_MIN_RELEVANCE", "0.35"))


settings = Settings()
