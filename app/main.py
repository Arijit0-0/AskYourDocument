from functools import lru_cache
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field
from .config import settings
from .rag import RagService, ConfigurationError

app = FastAPI(title='Enterprise Document Q&A RAG API', version='1.0.0')

@lru_cache
def get_service(): return RagService(settings)

class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=4000)
    document_ids: list[str] | None = None

@app.get('/health')
def health(): return {'status': 'ok', 'provider': 'gemini', 'vector_store': 'chroma'}

@app.get('/documents')
def list_documents():
    try: return get_service().list_documents()
    except ConfigurationError as exc: raise HTTPException(503, str(exc))

@app.delete('/documents/{document_id}', status_code=204)
def delete_document(document_id: str):
    try: get_service().delete_document(document_id)
    except ConfigurationError as exc: raise HTTPException(503, str(exc))

@app.post('/documents', status_code=201)
async def upload_documents(files: list[UploadFile] = File(...)):
    try: return [get_service().ingest_pdf(file.filename or 'document.pdf', await file.read()) for file in files]
    except (ConfigurationError, ValueError) as exc: raise HTTPException(503 if isinstance(exc, ConfigurationError) else 400, str(exc))

@app.post('/ask')
def ask(request: AskRequest):
    try:
        answer, sources = get_service().ask(request.question, request.document_ids)
        return {'answer': answer, 'sources': sources}
    except ConfigurationError as exc: raise HTTPException(503, str(exc))
