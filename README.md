# DocuMind AI - Document Intelligence Chatbot

**DocuMind** is a full-stack RAG (Retrieval-Augmented Generation) application for chatting with PDFs, Word docs, and images with source citations. It indexes files using FAISS and Sentence Transformers, allowing you to answer questions using Google Gemini, local Ollama, or both side-by-side.

---

## ✨ Features
- **Multi-Format Ingestion**: Supports PDF, DOCX, TXT, PNG, JPG, JPEG, and WEBP.
- **Table & OCR Extraction**: Extracts text and tables using PyMuPDF and python-docx, with Tesseract OCR for scanned documents.
- **Semantic Vector Search**: Fast chunking and vector indexing with Sentence Transformers (`all-MiniLM-L6-v2`) and FAISS.
- **Multi-LLM Switching**:
  - 🌟 **Google Gemini**: High-speed, intelligent cloud inference.
  - 🦙 **Local Ollama**: 100% private, offline inference on your own PC.
  - ⚡ **Dual Execution (Side-by-Side)**: Compare responses from both engines simultaneously.
- **Accurate Citations**: Every answer includes source page and excerpt references.

---

## 🚀 Getting Started

### Prerequisites
- Python 3.11+
- Node.js 20+
- Optional: [Ollama](https://ollama.com/) for local offline model execution.

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
- **For Google Gemini**: Set `LLM_PROVIDER=openai`, add your Gemini API key in `OPENAI_API_KEY`, and set `OPENAI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/`.
- **For Local Ollama**: Set `LLM_PROVIDER=ollama`, install Ollama, and run `ollama pull qwen2.5:3b`.

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

## 🛠️ Tech Stack
- **Frontend**: React 18, TypeScript, Vite, Lucide Icons, Vanilla CSS
- **Backend**: FastAPI, Uvicorn, Python 3.12
- **Embeddings & Vector Search**: Sentence Transformers, FAISS
- **Extraction**: PyMuPDF, python-docx, Pillow, PyTesseract
- **LLM Integrations**: Google Gemini API, Ollama
