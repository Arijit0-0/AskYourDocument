import re
import uuid
from datetime import UTC, datetime
from pathlib import Path

from langchain_chroma import Chroma
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from .config import Settings

NOT_FOUND_ANSWER = "I couldn't find that answer in the uploaded documents."


class ConfigurationError(RuntimeError):
    pass


class RagService:
    def __init__(self, settings: Settings):
        if not settings.gemini_api_key:
            raise ConfigurationError("GEMINI_API_KEY is required. Add it to .env, then restart the app.")
        self.settings = settings
        self.uploads = settings.data_dir / "uploads"
        self.uploads.mkdir(parents=True, exist_ok=True)
        self.store = Chroma(collection_name=settings.collection_name, persist_directory=str(settings.data_dir / "chroma"), embedding_function=GoogleGenerativeAIEmbeddings(model=settings.embedding_model, google_api_key=settings.gemini_api_key))
        self.splitter = RecursiveCharacterTextSplitter(chunk_size=settings.chunk_size, chunk_overlap=settings.chunk_overlap)
        self.llm = ChatGoogleGenerativeAI(model=settings.chat_model, google_api_key=settings.gemini_api_key, temperature=0)

    def ingest_pdf(self, filename: str, payload: bytes) -> dict[str, object]:
        if not filename.lower().endswith(".pdf"):
            raise ValueError("Only PDF files are supported.")
        document_id = str(uuid.uuid4())
        path = self.uploads / f"{document_id}_{re.sub(r'[^A-Za-z0-9._-]', '_', Path(filename).name)}"
        path.write_bytes(payload)
        pages = PyMuPDFLoader(str(path)).load()
        if not pages:
            raise ValueError("No extractable text was found in this PDF.")
        indexed_at = datetime.now(UTC).isoformat()
        for page in pages:
            page.metadata.update({"document_id": document_id, "filename": filename, "page": int(page.metadata.get("page", 0)) + 1, "indexed_at": indexed_at})
        chunks = self.splitter.split_documents(pages)
        self.store.add_documents(chunks, ids=[str(uuid.uuid4()) for _ in chunks])
        return {"document_id": document_id, "filename": filename, "pages_indexed": len(pages), "chunks_indexed": len(chunks)}

    def list_documents(self) -> list[dict[str, object]]:
        grouped: dict[str, dict[str, object]] = {}
        for metadata in self.store.get(include=["metadatas"]).get("metadatas", []):
            if metadata:
                item = grouped.setdefault(metadata["document_id"], {"document_id": metadata["document_id"], "filename": metadata["filename"], "pages": set(), "chunks_indexed": 0, "indexed_at": metadata.get("indexed_at", "")})
                item["pages"].add(metadata["page"]); item["chunks_indexed"] += 1
        return [{**item, "pages_indexed": len(item.pop("pages"))} for item in grouped.values()]

    def delete_document(self, document_id: str) -> None:
        self.store.delete(where={"document_id": document_id})

    def ask(self, question: str, document_ids: list[str] | None = None):
        filter_ = {"document_id": {"$in": document_ids}} if document_ids else None
        found = self.store.similarity_search_with_relevance_scores(question, k=self.settings.retrieval_k, filter=filter_)
        if not found or found[0][1] < self.settings.min_relevance:
            return NOT_FOUND_ANSWER, []
        sources = [{"citation": i + 1, "filename": doc.metadata["filename"], "page": doc.metadata["page"], "score": round(float(score), 4), "excerpt": doc.page_content[:300]} for i, (doc, score) in enumerate(found)]
        context = "\n\n".join(f"[{source['citation']}] {doc.page_content}" for (doc, _), source in zip(found, sources))
        prompt = ChatPromptTemplate.from_messages([("system", "Answer only from supplied context. If unsupported, reply exactly: " + NOT_FOUND_ANSWER), ("human", "Context:\n{context}\n\nQuestion: {question}")])
        answer = (prompt | self.llm | StrOutputParser()).invoke({"context": context, "question": question}).strip()
        return answer or NOT_FOUND_ANSWER, sources
