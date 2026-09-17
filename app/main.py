import os
import logging
from typing import Optional, Dict, Any
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
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

# Request Models
class ChatRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Support agent query")
    top_k: Optional[int] = Field(default=None, description="Number of context chunks to retrieve")
    document_type: Optional[str] = Field(default=None, description="Filter by document type")

class SettingsRequest(BaseModel):
    groq_api_key: str = Field(..., min_length=5, description="Groq API Key")

@app.on_event("startup")
async def startup_event():
    """Checks vector database on startup and performs auto-ingestion if empty."""
    stats = vector_store.get_stats()
    logger.info(f"Vector Store Status on Startup: {stats}")
    if stats["total_chunks"] == 0:
        logger.info("Vector store is empty. Triggering initial markdown ingestion...")
        try:
            chunks = load_and_chunk_all_markdown()
            count = vector_store.index_chunks(chunks)
            logger.info(f"Auto-ingested {count} chunks from workspace markdown files.")
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
            filter_type=req.document_type
        )
        return response
    except Exception as e:
        logger.error(f"Chat endpoint error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/ingest")
async def trigger_ingestion(reset: bool = True):
    """
    Reads all markdown files in workspace, parses YAML frontmatter,
    performs clause-preserving chunking, and indexes into ChromaDB.
    """
    try:
        chunks = load_and_chunk_all_markdown()
        count = vector_store.index_chunks(chunks, reset=reset)
        stats = vector_store.get_stats()
        return {
            "status": "success",
            "message": f"Successfully indexed {count} clause-preserved chunks.",
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
        "groq_api_key_configured": bool(rag_engine.groq_api_key),
        "gemini_embedding_key_configured": bool(settings.GEMINI_API_KEY),
        "embedding_model": settings.EMBEDDING_MODEL,
        "generation_model": settings.GENERATION_MODEL,
        "llm_provider": "groq",
        "vector_store": stats
    }

@app.post("/api/settings")
async def update_settings(req: SettingsRequest):
    """Allows setting the Groq API key at runtime from the UI."""
    key = req.groq_api_key.strip()
    if not key:
        raise HTTPException(status_code=400, detail="API key cannot be empty.")
    
    rag_engine.reload_api_key(key)
    
    # Save key to .env so it persists across restarts
    env_file = settings.BASE_DIR / ".env"
    try:
        lines = []
        if env_file.exists():
            with open(env_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
        
        key_found = False
        new_lines = []
        for line in lines:
            if line.startswith("GROQ_API_KEY="):
                new_lines.append(f"GROQ_API_KEY={key}\n")
                key_found = True
            else:
                new_lines.append(line)
        if not key_found:
            new_lines.append(f"\nGROQ_API_KEY={key}\n")

        with open(env_file, "w", encoding="utf-8") as f:
            f.writelines(new_lines)
            
    except Exception as e:
        logger.warning(f"Could not persist key to .env: {e}")

    return {
        "status": "success",
        "message": "Groq API key updated successfully!",
        "has_groq_key": True
    }

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", settings.PORT))
    host = os.getenv("HOST", "0.0.0.0" if os.getenv("RENDER") else settings.HOST)
    uvicorn.run("app.main:app", host=host, port=port, reload=settings.DEBUG)
