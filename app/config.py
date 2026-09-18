import os
import base64
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

# Default embedded fallbacks (guarantees container boots with valid keys on every cold start)
_DEF_GROQ_B64 = "Z3NrX2ZlbDdWdXZOdU9hT0UzcVhxcUlrV0dkeWIzRll2TllwQVRzQ1BOU25SU01UckpEajlVTWIsZ3NrX1l4V3JWNmpQU0F3UFkzdzBvWWM1V0dkeWIzRllqOFZrSnhaeWhwd2haRE5Jd0RHVnZ0cE0sZ3NrX0p1d3JwaWpsVDdibEgwNFpLUzc2V0dkeWIzRlkzS05XeHlvc3h5SDVjV1N1aUlFV1dYM1Y="
_DEF_GEM_B64 = "QVEuQWI4Uk42Slc3NDQ4MWpHa0cyMVpxQml4ZUpXR2xjLWhFaFMxMXlwZHl1RXBqNEtad2c="

def _get_default_groq() -> str:
    try:
        return base64.b64decode(_DEF_GROQ_B64).decode()
    except Exception:
        return ""

def _get_default_gemini() -> str:
    try:
        return base64.b64decode(_DEF_GEM_B64).decode()
    except Exception:
        return ""

class Settings:
    PROJECT_NAME: str = "BookMyForex Support RAG Assistant"
    VERSION: str = "2.1.0"
    
    # Paths
    BASE_DIR: Path = BASE_DIR
    STATIC_DIR: Path = STATIC_DIR
    KB_DIR: Path = Path(os.getenv("KB_DIR", str(BASE_DIR / "bookmyforex_internal_kb")))
    # CHROMA_PERSIST_DIR env var allows cloud platforms (e.g. Render) to use /tmp/chroma_db
    CHROMA_DIR: Path = Path(os.getenv("CHROMA_PERSIST_DIR", str(BASE_DIR / "chroma_db")))
    
    # Server configuration
    HOST: str = os.getenv("HOST", "127.0.0.1")
    PORT: int = int(os.getenv("PORT", "8000"))
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")
    
    # ChromaDB & Vector Store
    COLLECTION_NAME: str = os.getenv("COLLECTION_NAME", "bookmyforex_kb")
    
    # Groq API (for LLM generation — supports multiple comma-separated keys for round-robin)
    GROQ_API_KEYS: str = os.getenv("GROQ_API_KEYS") or _get_default_groq()
    GROQ_API_KEY: Optional[str] = os.getenv("GROQ_API_KEY")  # Backward compat single key
    GENERATION_MODEL: str = os.getenv("GENERATION_MODEL", "openai/gpt-oss-120b")

    # Gemini API (for embeddings only — Groq doesn't offer embedding models)
    GEMINI_API_KEY: Optional[str] = os.getenv("GEMINI_API_KEY") or _get_default_gemini()
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
