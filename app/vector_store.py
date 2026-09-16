import os
import math
import logging
from typing import List, Dict, Any, Optional
import chromadb
from chromadb.api.types import Documents, EmbeddingFunction, Embeddings
from app.config import settings
from app.ingestion import DocumentChunk

logger = logging.getLogger("vector_store")
logger.setLevel(logging.INFO)

class GeminiEmbeddingFunction(EmbeddingFunction):
    """
    Embedding function using Google Gemini's text-embedding-004 model.
    Falls back gracefully to a deterministic local vectorizer if GEMINI_API_KEY
    is not yet configured, allowing testing and offline operation.
    """
    def __init__(self, api_key: Optional[str] = None, model_name: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = model_name or settings.EMBEDDING_MODEL
        self._genai_client = None
        self._init_client()

    def _init_client(self):
        if not self.api_key:
            return
        
        # Try google.genai (new SDK) first, then fallback to google.generativeai
        try:
            from google import genai
            self._genai_client = genai.Client(api_key=self.api_key)
            self._sdk_type = "google_genai"
            logger.info("Initialized Google GenAI SDK client for embeddings.")
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._genai_client = gai
                self._sdk_type = "google_generativeai"
                logger.info("Initialized google.generativeai SDK client for embeddings.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini embedding client: {e}")
                self._genai_client = None

    def __call__(self, input: Documents) -> Embeddings:
        # If API key is present and client initialized, use Gemini embeddings
        if self._genai_client and self.api_key:
            try:
                return self._embed_with_gemini(input)
            except Exception as e:
                logger.error(f"Gemini embedding call failed: {e}. Falling back to local vectorizer.")

        # Fallback local deterministic hash-based embedding (dimension = 768)
        return self._local_hash_embedding(input)

    def _embed_with_gemini(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for text in texts:
            if self._sdk_type == "google_genai":
                response = self._genai_client.models.embed_content(
                    model=self.model_name,
                    contents=text
                )
                embeddings.append(response.embeddings[0].values)
            else:
                # google.generativeai
                res = self._genai_client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document"
                )
                embeddings.append(res["embedding"])
        return embeddings

    def embed_query(self, input: Any) -> Embeddings:
        """Embeds query inputs (str or list of str) conforming to ChromaDB's protocol."""
        if isinstance(input, str):
            texts = [input]
        else:
            texts = list(input)

        if self._genai_client and self.api_key:
            try:
                embeddings = []
                for query in texts:
                    if self._sdk_type == "google_genai":
                        response = self._genai_client.models.embed_content(
                            model=self.model_name,
                            contents=query
                        )
                        embeddings.append(response.embeddings[0].values)
                    else:
                        res = self._genai_client.embed_content(
                            model=self.model_name,
                            content=query,
                            task_type="retrieval_query"
                        )
                        embeddings.append(res["embedding"])
                return embeddings
            except Exception as e:
                logger.warning(f"Query embedding via Gemini failed: {e}")
        
        return self._local_hash_embedding(texts)

    def _local_hash_embedding(self, texts: List[str], dim: int = 768) -> List[List[float]]:
        """
        Deterministic word-frequency hash embedding vector for offline testing or fallback.
        Produces consistent, normalized 768-dimensional vectors.
        """
        import re
        import hashlib
        results = []
        for text in texts:
            words = re.findall(r'\b\w+\b', text.lower())
            vec = [0.0] * dim
            for w in words:
                h = int(hashlib.md5(w.encode('utf-8')).hexdigest(), 16)
                idx = h % dim
                sign = 1.0 if ((h >> 4) % 2 == 0) else -1.0
                vec[idx] += sign
            
            # Normalize vector
            norm = math.sqrt(sum(x * x for x in vec))
            if norm > 0:
                vec = [x / norm for x in vec]
            else:
                vec = [0.0] * dim
            results.append(vec)
        return results


class VectorStoreManager:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(settings.CHROMA_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = GeminiEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )

    def reload_api_key(self, api_key: str):
        """Updates embedding function with a newly provided API key."""
        self.embedding_fn.api_key = api_key
        self.embedding_fn._init_client()

    def index_chunks(self, chunks: List[DocumentChunk], reset: bool = False) -> int:
        """
        Indexes chunks into the ChromaDB collection.
        If reset is True, clears existing items in the collection first.
        """
        if reset:
            self.client.delete_collection(settings.COLLECTION_NAME)
            self.collection = self.client.get_or_create_collection(
                name=settings.COLLECTION_NAME,
                embedding_function=self.embedding_fn,
                metadata={"hnsw:space": "cosine"}
            )

        if not chunks:
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # Chroma upsert handles duplicates safely
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k matching chunks with similarity scores and metadata.
        """
        results = self.collection.query(
            query_texts=[query_text],
            n_results=top_k,
            where=filter_metadata if filter_metadata else None,
            include=["documents", "metadatas", "distances"]
        )

        retrieved = []
        if results and "documents" in results and results["documents"]:
            docs = results["documents"][0]
            metas = results["metadatas"][0] if "metadatas" in results else [{}] * len(docs)
            dists = results["distances"][0] if "distances" in results else [0.0] * len(docs)

            for doc, meta, dist in zip(docs, metas, dists):
                # Cosine distance to similarity: similarity = 1 - distance
                similarity = max(0.0, 1.0 - dist)
                retrieved.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": round(similarity, 4)
                })

        return retrieved

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the vector store index."""
        count = self.collection.count()
        # Retrieve sample of metadatas to find unique source files
        unique_sources = set()
        if count > 0:
            sample = self.collection.get(limit=count, include=["metadatas"])
            for m in sample.get("metadatas", []):
                if m and "source_file" in m:
                    unique_sources.add(m["source_file"])
                    
        return {
            "total_chunks": count,
            "unique_documents": len(unique_sources),
            "sources": sorted(list(unique_sources)),
            "collection_name": settings.COLLECTION_NAME,
            "has_gemini_key": bool(self.embedding_fn.api_key)
        }

# Global singleton instance
vector_store = VectorStoreManager()
