from uuid import uuid4
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from .config import LLM_PROVIDER, MAX_UPLOAD_MB, DEFAULT_ANSWER_LANGUAGE, SUPPORTED_ANSWER_LANGUAGES, EMBEDDING_MODEL
from .document_loader import extract_document, SUPPORTED_EXTENSIONS
from .llm import generate_answer
from .rag import index_pages, retrieve, IncompatibleIndexError
from .schemas import ChatRequest, AnswerLanguage

app = FastAPI(title="DocuMind AI - Document Intelligence Chatbot API", version="0.2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory document metadata storage for runtime session
DOCUMENT_META: dict[str, dict] = {}


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "provider": LLM_PROVIDER,
        "default_answer_language": DEFAULT_ANSWER_LANGUAGE,
        "supported_languages": SUPPORTED_ANSWER_LANGUAGES,
        "embedding_model": EMBEDDING_MODEL,
    }


@app.post("/api/documents/upload")
async def upload_document(file: UploadFile = File(...)):
    filename = file.filename or "uploaded-file"
    from pathlib import Path
    extension = Path(filename).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise HTTPException(400, f"Unsupported file type. Allowed: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")
    data = await file.read()
    if not data:
        raise HTTPException(400, "The uploaded file is empty.")
    if len(data) > MAX_UPLOAD_MB * 1024 * 1024:
        raise HTTPException(413, f"File exceeds the {MAX_UPLOAD_MB} MB upload limit.")
    try:
        pages = extract_document(filename, data)
        document_id = str(uuid4())
        chunk_count = index_pages(document_id, pages)
    except Exception as exc:
        # Avoid leaking stack traces or secrets to the client.
        raise HTTPException(422, f"Could not process document: {str(exc)}") from exc

    DOCUMENT_META[document_id] = {
        "document_id": document_id,
        "filename": filename,
        "pages_with_text": len(pages),
        "chunks": chunk_count,
        "preview": "\n\n".join(p["text"][:500] for p in pages[:2])[:900],
    }
    return DOCUMENT_META[document_id]


@app.post("/api/chat")
async def chat(request: ChatRequest):
    if request.document_id not in DOCUMENT_META:
        raise HTTPException(404, "Document not found. Upload it again if the backend restarted.")
    try:
        chunks = retrieve(request.document_id, request.question)
        result = await generate_answer(
            request.question,
            chunks,
            request.history,
            provider=request.provider,
            answer_language=request.answer_language,
        )
    except KeyError as exc:
        raise HTTPException(404, str(exc)) from exc
    except IncompatibleIndexError as exc:
        raise HTTPException(409, str(exc)) from exc
    except Exception as exc:
        raise HTTPException(502, f"Could not generate an answer: {str(exc)}") from exc

    sources = [
        {"page": c.get("page"), "kind": c.get("kind"), "text": c["text"], "score": round(c["score"], 3)}
        for c in chunks
    ]
    return {
        "answer": result["answer"],
        "provider": result.get("provider", request.provider),
        "answer_language": result.get("answer_language", request.answer_language.value),
        "answers": result.get("answers", {}),
        "sources": sources,
    }
