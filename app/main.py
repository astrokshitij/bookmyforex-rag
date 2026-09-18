import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from app.config import settings
from app.ingestion import load_and_chunk_all_markdown
from app.vector_store import vector_store
from app.rag_engine import rag_engine

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("bookmyforex_rag")

app = FastAPI(
    title=settings.PROJECT_NAME,
    version=settings.VERSION,
    description="Local internal RAG application for BookMyForex support teams."
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Robust static directory path resolution relative to this file
STATIC_DIR = Path(__file__).resolve().parent / "static"
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

from typing import Optional, Dict, Any, List

# In-memory escalation & feedback logs
ESCALATIONS_LOG: List[Dict[str, Any]] = []

# Request Models
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Support agent query")
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to retrieve")
    document_type: Optional[str] = Field(default=None, description="Filter by document type")
    history: Optional[List[Dict[str, str]]] = Field(default=None, description="Recent conversation history")

class SettingsRequest(BaseModel):
    groq_api_keys: Optional[str] = Field(default=None, description="Groq API Key(s), comma-separated for multiple")
    groq_api_key: Optional[str] = Field(default=None, description="Single Groq API Key")
    gemini_api_key: Optional[str] = Field(default=None, description="Gemini API Key for embeddings")

class FeedbackRequest(BaseModel):
    query: str = Field(..., description="The query that was answered")
    answer: str = Field(..., description="The response provided")
    rating: str = Field(..., description="'positive' or 'negative'")
    comment: Optional[str] = Field(default=None, description="Optional agent feedback note")

@app.on_event("startup")
async def startup_event():
    """Checks vector database on startup and performs auto-ingestion if empty or BM25 unindexed."""
    stats = vector_store.get_stats()
    logger.info(f"Vector Store Status on Startup: {stats}")
    if stats["total_chunks"] == 0 or stats["bm25_indexed_chunks"] == 0:
        logger.info("Initializing markdown ingestion and building ChromaDB + BM25 indices...")
        try:
            chunks = load_and_chunk_all_markdown()
            count = vector_store.index_chunks(chunks, reset=(stats["total_chunks"] == 0))
            logger.info(f"Successfully indexed {count} chunks for hybrid search.")
        except Exception as e:
            logger.error(f"Error during startup auto-ingestion: {e}")

@app.get("/", response_class=HTMLResponse)
@app.get("/index.html", response_class=HTMLResponse)
async def serve_ui():
    """Serves the main chat application UI."""
    index_file = STATIC_DIR / "index.html"
    if not index_file.exists():
        # Fallback to settings.STATIC_DIR
        index_file = settings.STATIC_DIR / "index.html"
    if not index_file.exists():
        raise HTTPException(status_code=404, detail="UI index.html not found.")
    return FileResponse(str(index_file), media_type="text/html")

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    """Silences browser favicon 404 requests on cloud platforms."""
    return HTMLResponse(content="", status_code=204)

@app.post("/api/chat")
async def chat(req: ChatRequest):
    """
    RAG Chat endpoint for support queries.
    Strictly answers using BookMyForex guidelines or triggers compliance fallback.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    try:
        response = rag_engine.generate_response(
            query=req.query,
            top_k=req.top_k,
            filter_type=req.document_type,
            history=req.history
        )
        # Automatically log compliance escalations for auditing
        if response.get("fallback_triggered"):
            import time
            ESCALATIONS_LOG.append({
                "timestamp": time.time(),
                "query": req.query,
                "reason": "compliance_fallback_triggered",
                "citations_count": len(response.get("citations", []))
            })
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/chat/stream")
async def chat_stream(req: ChatRequest):
    """
    Real-time Server-Sent Events (SSE) streaming endpoint for low-latency chat.
    Yields initial citations immediately, then streaming tokens.
    """
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    return StreamingResponse(
        rag_engine.generate_response_stream(
            query=req.query,
            top_k=req.top_k,
            filter_type=req.document_type,
            history=req.history
        ),
        media_type="text/event-stream"
    )

@app.post("/api/feedback")
async def submit_feedback(fb: FeedbackRequest):
    """Logs agent feedback (thumbs up / down) for answer quality monitoring."""
    import time
    entry = {
        "timestamp": time.time(),
        "query": fb.query,
        "answer_snippet": fb.answer[:200],
        "rating": fb.rating,
        "comment": fb.comment
    }
    ESCALATIONS_LOG.append(entry)
    logger.info(f"Agent Feedback Recorded: {fb.rating} for query '{fb.query[:50]}'")
    return {"status": "success", "message": "Feedback recorded. Thank you for improving the assistant!"}

@app.get("/api/escalations")
async def list_escalations():
    """Returns the compliance and agent feedback escalation audit log."""
    return {
        "total_logged": len(ESCALATIONS_LOG),
        "escalations": ESCALATIONS_LOG[-50:]
    }

@app.post("/api/ingest")
async def trigger_ingestion(reset: bool = True):
    """
    Reads all markdown files in workspace, parses YAML frontmatter,
    performs clause-preserving chunking, and indexes into ChromaDB and BM25.
    Also flushes the in-memory response cache.
    """
    try:
        chunks = load_and_chunk_all_markdown()
        count = vector_store.index_chunks(chunks, reset=reset)
        rag_engine.cache.clear()
        stats = vector_store.get_stats()
        return {
            "status": "success",
            "message": f"Successfully indexed {count} clause-preserved chunks into Hybrid Vector & BM25 indices.",
            "stats": stats
        }
    except Exception as e:
        logger.error(f"Ingestion failed: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Ingestion error: {str(e)}")

@app.get("/api/documents")
async def list_documents():
    """Lists all available documents and their indexed metadata."""
    try:
        chunks = load_and_chunk_all_markdown()
        doc_summary: Dict[str, Dict[str, Any]] = {}
        for c in chunks:
            sf = c.source_file
            if sf not in doc_summary:
                doc_summary[sf] = {
                    "source_file": sf,
                    "document_title": c.document_title,
                    "document_type": c.document_type,
                    "last_updated": c.last_updated,
                    "chunks_count": 0,
                    "sections": []
                }
            doc_summary[sf]["chunks_count"] += 1
            if c.section_title not in doc_summary[sf]["sections"]:
                doc_summary[sf]["sections"].append(c.section_title)

        return {
            "total_documents": len(doc_summary),
            "total_chunks": len(chunks),
            "documents": list(doc_summary.values())
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/api/health")
async def health_check():
    """Returns application health and configuration status."""
    stats = vector_store.get_stats()
    return {
        "status": "healthy",
        "app_name": settings.PROJECT_NAME,
        "version": settings.VERSION,
        "groq_keys_configured": rag_engine.key_count,
        "gemini_embedding_key_configured": bool(settings.GEMINI_API_KEY),
        "embedding_model": settings.EMBEDDING_MODEL,
        "generation_model": settings.GENERATION_MODEL,
        "llm_provider": "groq",
        "vector_store": stats,
        "cache": rag_engine.cache.get_stats()
    }

@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    """Allows setting Groq API key(s) and/or Gemini API key at runtime."""
    updated_items = []
    
    # Handle Groq keys
    groq_input = (req.groq_api_keys or req.groq_api_key or "").strip()
    if groq_input:
        rag_engine.reload_api_keys(groq_input)
        updated_items.append(f"{rag_engine.key_count} Groq key(s)")
        
        # Persist to .env
        env_file = settings.BASE_DIR / ".env"
        try:
            lines = []
            if env_file.exists():
                with open(env_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            key_found = False
            new_lines = []
            for line in lines:
                if line.startswith("GROQ_API_KEYS=") or line.startswith("GROQ_API_KEY="):
                    if not key_found:
                        new_lines.append(f"GROQ_API_KEYS={groq_input}\n")
                        key_found = True
                else:
                    new_lines.append(line)
            if not key_found:
                new_lines.append(f"\nGROQ_API_KEYS={groq_input}\n")

            with open(env_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            logger.warning(f"Could not persist Groq keys to .env: {e}")

    # Handle Gemini key for embeddings
    gemini_input = (req.gemini_api_key or "").strip()
    if gemini_input:
        settings.GEMINI_API_KEY = gemini_input
        vector_store.reload_api_key(gemini_input)
        updated_items.append("Gemini embedding key")
        
        # Persist to .env
        env_file = settings.BASE_DIR / ".env"
        try:
            lines = []
            if env_file.exists():
                with open(env_file, "r", encoding="utf-8") as f:
                    lines = f.readlines()
            
            key_found = False
            new_lines = []
            for line in lines:
                if line.startswith("GEMINI_API_KEY="):
                    new_lines.append(f"GEMINI_API_KEY={gemini_input}\n")
                    key_found = True
                else:
                    new_lines.append(line)
            if not key_found:
                new_lines.append(f"\nGEMINI_API_KEY={gemini_input}\n")

            with open(env_file, "w", encoding="utf-8") as f:
                f.writelines(new_lines)
        except Exception as e:
            logger.warning(f"Could not persist Gemini key to .env: {e}")

    if not updated_items:
        raise HTTPException(status_code=400, detail="No valid API key provided to update.")

    return {
        "status": "success",
        "message": f"Successfully updated: {', '.join(updated_items)}!",
        "groq_keys_count": rag_engine.key_count,
        "gemini_configured": bool(settings.GEMINI_API_KEY)
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", settings.PORT))
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("RENDER") else settings.HOST)
    uvicorn.run("app.main:app", host=host, port=port, reload=settings.DEBUG)
