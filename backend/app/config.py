import os
from dotenv import load_dotenv

load_dotenv()

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "ollama").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_BASE_URL = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "qwen2.5:1.5b")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)
MAX_UPLOAD_MB = int(os.getenv("MAX_UPLOAD_MB", "20"))
DEFAULT_ANSWER_LANGUAGE = os.getenv("DEFAULT_ANSWER_LANGUAGE", "en").lower()
SUPPORTED_ANSWER_LANGUAGES = ["en", "ml"]
OLLAMA_NUM_GPU = int(os.getenv("OLLAMA_NUM_GPU", "-1"))
