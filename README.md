# DocuMind AI - Document Intelligence Chatbot

**DocuMind** is a production-grade full-stack RAG (Retrieval-Augmented Generation) application for chatting with PDFs, Word docs, text files, and scanned images with verifiable source citations. It indexes files using FAISS and Sentence Transformers, and generates answers using Google Gemini, local offline Ollama, or both side-by-side — now with native **Malayalam (മലയാളം) Answer Mode**!

---

## ✨ Features

- **Multi-Format Ingestion**: Supports PDF, DOCX, TXT, PNG, JPG, JPEG, and WEBP.
- **Table & OCR Extraction**: Extracts text and tables using PyMuPDF and python-docx, with Tesseract OCR fallback for scanned documents.
- **Semantic Vector Search**: Fast chunking and vector indexing with Sentence Transformers (`all-MiniLM-L6-v2`) and FAISS.
- **🌐 Bilingual Answer Mode (English & മലയാളം)**:
  - Toggle between **English** and **മലയാളം (Malayalam)** on the fly with a single click.
  - Answers in Malayalam use natural Malayalam script (മലയാളം ലിപി), preserving names, numbers, and technical terms.
  - Retains source citations `[Page X]` or `[പേജ് X]` without translating the underlying document or requiring re-indexing.
- **Multi-LLM Switching**:
  - 🌟 **Google Gemini**: High-speed, intelligent cloud inference.
  - 🦙 **Local Ollama**: 100% private, offline inference on your own PC.
  - ⚡ **Dual Execution (Side-by-Side)**: Compare responses from both engines simultaneously.
- **Accurate Citations**: Every answer includes source page and excerpt references with similarity scores.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Optional: [Ollama](https://ollama.com/) for local offline model execution (`qwen2.5:3b`).

---

### 1. Backend Setup

```bash
cd backend
python -m venv .venv

# Windows PowerShell:
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope Process -Force
.\.venv\Scripts\Activate.ps1

# Install dependencies:
pip install -r requirements.txt

# Configure environment:
copy .env.example .env
```

Edit `backend/.env` with your preferred settings:
- **For Google Gemini**: Set `LLM_PROVIDER=openai`, set `OPENAI_API_KEY=<your_key>`, and set `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`.
- **For Local Ollama**: Set `LLM_PROVIDER=ollama`, install Ollama, and run `ollama pull qwen2.5:3b`.
- **Default Language**: `DEFAULT_ANSWER_LANGUAGE=en` (or `ml`).

Start the backend:
```bash
uvicorn app.main:app --reload
```
API Documentation: [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)

---

### 2. Frontend Setup

Open a new terminal:
```bash
cd frontend
npm install
npm run dev
```
Open [http://localhost:5173](http://localhost:5173) in your browser.

---

## 🌐 Malayalam Answer Mode & API Contract

### How to Use the UI Toggle
1. Upload any supported document (PDF, DOCX, TXT, image).
2. Use the language toggle in the **Topbar** (`English` | `മലയാളം`) or the quick-toggle pill in the **Chat Composer** footer.
3. Ask your question in English or Malayalam.
4. The system retrieves document evidence and responds in the selected language with page citations.

### API Contract (`POST /api/chat`)

```json
{
  "document_id": "uuid-string",
  "question": "Who created this document?",
  "provider": "gemini",
  "answer_language": "ml",
  "history": []
}
```

- **`answer_language`**: Accepts `"en"` (default) or `"ml"`. Also accepts aliases `"english"` and `"malayalam"`.
- Requests omitting `answer_language` default to English for backward compatibility.
- Invalid language requests return HTTP 422 with a clear validation error.

Response:
```json
{
  "answer": "ഈ രേഖ നിർമ്മിച്ചത് ബിലാൽ അഹമ്മദ് ആണ് [Page 1].",
  "provider": "gemini",
  "answer_language": "ml",
  "answers": {
    "gemini": "ഈ രേഖ നിർമ്മിച്ചത് ബിലാൽ അഹമ്മദ് ആണ് [Page 1]."
  },
  "sources": [
    {
      "page": 1,
      "kind": "page",
      "text": "...",
      "score": 0.884
    }
  ]
}
```

---

## 🧪 Automated Testing

### Backend Tests (pytest)
Runs 23 automated unit and integration tests covering:
- English & Malayalam prompt generation
- Schema validation & invalid language rejection (422)
- Provider routing (Gemini, Ollama, Both)
- Malayalam RAG evaluation & citation preservation
- Backward compatibility for legacy requests

```bash
cd backend
.\.venv\Scripts\python.exe -m pytest -v
```

### Frontend Tests & Production Build
```bash
cd frontend
npm test
npm run build
```

---

## 🛠️ Tech Stack
- **Frontend**: React 18, TypeScript, Vite, Lucide Icons, Vanilla CSS
- **Backend**: FastAPI, Uvicorn, Python 3.12, Pydantic v2
- **Embeddings & Vector Search**: Sentence Transformers (`all-MiniLM-L6-v2`), FAISS
- **Extraction**: PyMuPDF, python-docx, Pillow, PyTesseract
- **LLM Integrations**: Google Gemini API, Ollama (`qwen2.5:3b`)
