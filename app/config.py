import os
from pathlib import Path
from typing import Optional

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "app" / "static"
CHROMA_DIR = BASE_DIR / "chroma_db"

# Try loading .env file
try:
    from dotenv import load_dotenv
    load_dotenv(BASE_DIR / ".env")
except ImportError:
    pass

class Settings:
    PROJECT_NAME: str = "BookMyForex Support RAG Assistant"
    VERSION: str = "2.1.0"
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    STATIC_DIR: Path = STATIC_DIR
    # CHROMA_PERSIST_DIR env var allows cloud platforms (e.g. Render) to use /tmp/chroma_db
    CHROMA_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")))
    
    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    
    # ChromaDB & Vector Store
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "bookmyforex_kb")
    
    # Groq API (for LLM generation — supports multiple comma-separated keys for round-robin)
    GROQ_API_KEYS: str = os.getenv("GROQ_API_KEYS", "")  # Comma-separated keys
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")  # Backward compat single key
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")

    # Gemini API (for embeddings only — Groq doesn't offer embedding models)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY")
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "models/gemini-embedding-2")
    
    # RAG parameters
    TOP_K: int = int(os.getenv("TOP_K", "5"))
    MIN_SIMILARITY_THRESHOLD: float = float(os.getenv("MIN_SIMILARITY_THRESHOLD", "0.25"))

    # Strict compliance guardrail fallback message
    COMPLIANCE_FALLBACK: str = (
        "I cannot find this information in the official BookMyForex guidelines. "
        "Please check with the compliance desk."
    )

settings = Settings()
