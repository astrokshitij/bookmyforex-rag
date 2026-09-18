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
                    contents=text,
                    config={"task_type": "RETRIEVAL_DOCUMENT", "output_dimensionality": 768}
                )
                embeddings.append(response.embeddings[0].values)
            else:
                # google.generativeai
                res = self._genai_client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document",
                    output_dimensionality=768
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
                            contents=query,
                            config={"task_type": "RETRIEVAL_QUERY", "output_dimensionality": 768}
                        )
                        embeddings.append(response.embeddings[0].values)
                    else:
                        res = self._genai_client.embed_content(
                            model=self.model_name,
                            content=query,
                            task_type="retrieval_query",
                            output_dimensionality=768
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


BM25_STOP_WORDS = {
    "a", "about", "above", "after", "again", "against", "all", "am", "an", "and",
    "any", "are", "aren't", "as", "at", "be", "because", "been", "before", "being",
    "below", "between", "both", "but", "by", "can", "can't", "cannot", "could",
    "couldn't", "did", "didn't", "do", "does", "doesn't", "doing", "don't", "down",
    "during", "each", "few", "for", "from", "further", "had", "hadn't", "has",
    "hasn't", "have", "haven't", "having", "he", "her", "here", "hers", "herself",
    "him", "himself", "his", "how", "how's", "i", "i'd", "i'll", "i'm", "i've",
    "if", "in", "into", "is", "isn't", "it", "it's", "its", "itself", "let's",
    "me", "more", "most", "mustn't", "my", "myself", "no", "nor", "not", "of",
    "off", "on", "once", "only", "or", "other", "ought", "our", "ours", "ourselves",
    "out", "over", "own", "same", "shan't", "she", "should", "shouldn't", "so",
    "some", "such", "than", "that", "the", "their", "theirs", "them", "themselves",
    "then", "there", "these", "they", "this", "those", "through", "to", "too",
    "under", "until", "up", "very", "was", "wasn't", "we", "were", "weren't",
    "what", "when", "where", "which", "while", "who", "whom", "why", "with",
    "won't", "would", "wouldn't", "you", "your", "yours", "yourself", "yourselves",
    "tell", "give", "avail", "get", "details", "info", "information", "please",
    "show", "know", "check", "want", "need", "customer", "asks", "ask"
}

SYNONYMS = {
    "sim": ["sim", "esim", "data", "poshvine", "talktime"],
    "esim": ["sim", "esim", "data", "poshvine", "talktime"],
    "lounge": ["lounge", "lounges", "airport", "razorpay", "1200"],
    "lounges": ["lounge", "lounges", "airport", "razorpay", "1200"],
    "flight": ["flight", "flights", "airline"],
    "flights": ["flight", "flights", "airline"],
    "hotel": ["hotel", "hotels", "stay"],
    "hotels": ["hotel", "hotels", "stay"],
    "remittance": ["remittance", "remit", "transfer", "outward"],
    "transfer": ["transfer", "transfers", "remittance", "outward"],
    "internation": ["international", "internation", "overseas", "global"],
    "international": ["international", "internation", "overseas", "global"],
    "founder": ["founder", "ceo", "sudarshan", "motwani", "leadership", "founded"],
    "ceo": ["founder", "ceo", "sudarshan", "motwani", "leadership"],
    "makemytrip": ["makemytrip", "tripmoney", "mmt", "acquisition", "acquired", "parent", "stake"],
    "cancellation": ["cancellation", "cancel", "refund", "refunds", "cancelled", "return"],
    "refund": ["refund", "refunds", "cancellation", "cancel", "reversal"],
    "refunds": ["refund", "refunds", "cancellation", "cancel", "reversal"],
    "tuition": ["tuition", "university", "education", "s0305", "remittance", "fees"],
    "tcs": ["tcs", "tax", "collected", "source", "lrs", "loan", "education"],
    "jetsetter": ["jetsetter", "bonus", "poshvine", "reward", "rewards", "10000"],
    "isic": ["isic", "student", "card", "complimentary", "digital"],
    "voucher": ["voucher", "cab", "airport", "ride", "makemytrip", "amazon", "swiggy", "zomato"],
    "ride": ["ride", "cab", "voucher", "airport", "transfer", "makemytrip"],
    "doorstep": ["doorstep", "delivery", "timeline", "hours", "same-day"],
    "rate": ["rate", "rates", "lock", "guaranteed", "interbank", "live"],
}


class BM25Index:
    """
    Lightweight BM25Okapi inverted index for keyword search with stop-word filtering
    and stem normalization for high precision on domain terms.
    """
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1 = k1
        self.b = b
        self.doc_len: List[int] = []
        self.avgdl: float = 0.0
        self.doc_freqs: List[Dict[str, int]] = []
        self.idf: Dict[str, float] = {}
        self.corpus_size: int = 0
        self.chunks: List[DocumentChunk] = []

    def _stem(self, w: str) -> str:
        w = w.lower().strip()
        if w.startswith("esim"):
            return "sim"
        if w.endswith("ies") and len(w) > 4:
            return w[:-3] + "y"
        if w.endswith("es") and len(w) > 4:
            return w[:-2]
        if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
            return w[:-1]
        if w.endswith("ing") and len(w) > 5:
            return w[:-3]
        if w.endswith("tional") and len(w) > 8:
            return w[:-6]
        if w.endswith("tion") and len(w) > 6:
            return w[:-4]
        return w

    def _tokenize(self, text: str, expand_synonyms: bool = False) -> List[str]:
        import re
        raw_tokens = [w for w in re.findall(r'[a-zA-Z0-9_\-₹$€£]+', text.lower()) if len(w) > 1]
        tokens = []
        for t in raw_tokens:
            if t in BM25_STOP_WORDS:
                continue
            stemmed = self._stem(t)
            tokens.append(stemmed)
            if stemmed != t:
                tokens.append(t)
            if expand_synonyms and t in SYNONYMS:
                tokens.extend(SYNONYMS[t])

        # If all tokens were stopwords, fallback to keeping raw tokens to avoid empty query
        if not tokens:
            tokens = [self._stem(t) for t in raw_tokens if len(t) > 1]
        return tokens

    def build_index(self, chunks: List[DocumentChunk]):
        self.chunks = chunks
        self.corpus_size = len(chunks)
        if self.corpus_size == 0:
            self.doc_len = []
            self.avgdl = 0.0
            self.doc_freqs = []
            self.idf = {}
            return

        self.doc_len = []
        self.doc_freqs = []
        df: Dict[str, int] = {}

        for chunk in chunks:
            tokens = self._tokenize(chunk.text, expand_synonyms=False)
            self.doc_len.append(len(tokens))
            freqs: Dict[str, int] = {}
            for t in tokens:
                freqs[t] = freqs.get(t, 0) + 1
            self.doc_freqs.append(freqs)
            for t in freqs.keys():
                df[t] = df.get(t, 0) + 1

        self.avgdl = sum(self.doc_len) / self.corpus_size if self.corpus_size > 0 else 0.0

        self.idf = {}
        for term, freq in df.items():
            self.idf[term] = math.log((self.corpus_size - freq + 0.5) / (freq + 0.5) + 1.0)
        logger.info(f"Built BM25 Index over {self.corpus_size} chunk(s) (vocabulary: {len(self.idf)} terms).")

    def search(
        self,
        query: str,
        top_k: int = 10,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        if self.corpus_size == 0:
            return []

        q_tokens = self._tokenize(query, expand_synonyms=True)
        if not q_tokens:
            return []

        scores: List[Tuple[int, float]] = []

        for idx, chunk in enumerate(self.chunks):
            # Apply metadata filtering if specified
            if filter_metadata:
                skip = False
                for k, v in filter_metadata.items():
                    if v and chunk.metadata.get(k) != v:
                        skip = True
                        break
                if skip:
                    continue

            doc_len = self.doc_len[idx]
            freqs = self.doc_freqs[idx]
            score = 0.0

            for q in q_tokens:
                if q not in freqs:
                    continue
                tf = freqs[q]
                idf = self.idf.get(q, 0.0)
                numerator = tf * (self.k1 + 1.0)
                denominator = tf + self.k1 * (1.0 - self.b + self.b * (doc_len / (self.avgdl or 1.0)))
                score += idf * (numerator / denominator)

            if score > 0.0:
                scores.append((idx, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        results = []
        for idx, score in scores[:top_k]:
            c = self.chunks[idx]
            results.append({
                "chunk_id": c.chunk_id,
                "content": c.text,
                "metadata": c.metadata,
                "bm25_score": round(score, 4)
            })
        return results


class VectorStoreManager:
    def __init__(self, persist_dir: Optional[str] = None):
        self.persist_dir = persist_dir or str(settings.CHROMA_DIR)
        os.makedirs(self.persist_dir, exist_ok=True)
        self.client = chromadb.PersistentClient(path=self.persist_dir)
        self.embedding_fn = GeminiEmbeddingFunction()
        self.bm25_index = BM25Index()
        self.collection = self.client.get_or_create_collection(
            name=settings.COLLECTION_NAME,
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": "cosine"}
        )
        self._ensure_initialized()

    def _ensure_initialized(self):
        """Ensures ChromaDB and BM25 are populated with KB documents upon instantiation."""
        try:
            from app.ingestion import load_and_chunk_all_markdown
            if self.collection.count() == 0 or self.bm25_index.corpus_size == 0:
                chunks = load_and_chunk_all_markdown()
                if chunks:
                    self.index_chunks(chunks, reset=(self.collection.count() == 0))
        except Exception as e:
            logger.warning(f"Vector store auto-init skipped: {e}")

    def reload_api_key(self, api_key: str):
        """Updates embedding function with a newly provided API key."""
        self.embedding_fn.api_key = api_key
        self.embedding_fn._init_client()

    def index_chunks(self, chunks: List[DocumentChunk], reset: bool = False) -> int:
        """
        Indexes chunks into both ChromaDB collection and BM25 sparse index.
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
            self.bm25_index.build_index([])
            return 0

        ids = [c.chunk_id for c in chunks]
        documents = [c.text for c in chunks]
        metadatas = [c.metadata for c in chunks]

        # 1. Update ChromaDB
        self.collection.upsert(
            ids=ids,
            documents=documents,
            metadatas=metadatas
        )

        # 2. Update BM25 Index
        self.bm25_index.build_index(chunks)
        return len(chunks)

    def query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieves top_k matching chunks with similarity scores and metadata.
        Chunks below MIN_SIMILARITY_THRESHOLD are filtered out.
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
                if similarity < settings.MIN_SIMILARITY_THRESHOLD:
                    logger.debug(
                        f"Chunk filtered (similarity={similarity:.4f} < threshold={settings.MIN_SIMILARITY_THRESHOLD}): "
                        f"{meta.get('source_file', '?')} | {meta.get('section_title', '?')}"
                    )
                    continue
                retrieved.append({
                    "content": doc,
                    "metadata": meta,
                    "distance": dist,
                    "similarity": round(similarity, 4)
                })

        return retrieved

    def hybrid_query(
        self,
        query_text: str,
        top_k: int = 5,
        filter_metadata: Optional[Dict[str, Any]] = None
    ) -> List[Dict[str, Any]]:
        """
        Executes Hybrid Search combining Dense Vector Search (Chroma) + Sparse BM25.
        Fuses candidate rankings using Reciprocal Rank Fusion (RRF, k=60).
        """
        candidate_k = max(top_k * 2, 8)

        # 1. Dense retrieval
        dense_results = self.query(
            query_text=query_text,
            top_k=candidate_k,
            filter_metadata=filter_metadata
        )

        # 2. Sparse BM25 retrieval
        bm25_results = self.bm25_index.search(
            query=query_text,
            top_k=candidate_k,
            filter_metadata=filter_metadata
        )

        # If BM25 is not built yet or returns nothing, fall back to dense
        if not bm25_results:
            return dense_results[:top_k]

        # If dense returned nothing, fall back to BM25
        if not dense_results:
            bm25_converted = []
            for r in bm25_results[:top_k]:
                bm25_converted.append({
                    "content": r["content"],
                    "metadata": r["metadata"],
                    "distance": 0.0,
                    "similarity": 0.5,
                    "bm25_score": r["bm25_score"]
                })
            return bm25_converted

        # 3. Reciprocal Rank Fusion (RRF)
        RRF_K = 60
        fused: Dict[str, Dict[str, Any]] = {}

        for rank, item in enumerate(dense_results):
            cid = item["metadata"].get("chunk_id") or item["content"][:80]
            rrf_score = 1.0 / (RRF_K + rank + 1)
            fused[cid] = {
                "item": item,
                "score": rrf_score,
                "dense_rank": rank + 1,
                "bm25_rank": None
            }

        for rank, item in enumerate(bm25_results):
            cid = item["metadata"].get("chunk_id") or item["content"][:80]
            rrf_score = 1.0 / (RRF_K + rank + 1)
            if cid in fused:
                fused[cid]["score"] += rrf_score
                fused[cid]["bm25_rank"] = rank + 1
            else:
                fused[cid] = {
                    "item": {
                        "content": item["content"],
                        "metadata": item["metadata"],
                        "distance": 0.0,
                        "similarity": 0.5
                    },
                    "score": rrf_score,
                    "dense_rank": None,
                    "bm25_rank": rank + 1
                }

        # Sort by fused score descending
        sorted_candidates = sorted(fused.values(), key=lambda x: x["score"], reverse=True)

        final_results = []
        for c in sorted_candidates[:top_k]:
            entry = c["item"]
            entry["rrf_score"] = round(c["score"], 5)
            entry["dense_rank"] = c["dense_rank"]
            entry["bm25_rank"] = c["bm25_rank"]
            final_results.append(entry)

        logger.info(
            f"Hybrid RRF search retrieved {len(final_results)} fused chunk(s) for: '{query_text[:80]}'"
        )
        return final_results

    def get_stats(self) -> Dict[str, Any]:
        """Returns statistics on the vector store and BM25 index."""
        count = self.collection.count()
        unique_sources = set()
        if count > 0:
            sample = self.collection.get(limit=count, include=["metadatas"])
            for m in sample.get("metadatas", []):
                if m and "source_file" in m:
                    unique_sources.add(m["source_file"])

        return {
            "total_chunks": count,
            "bm25_indexed_chunks": self.bm25_index.corpus_size,
            "unique_documents": len(unique_sources),
            "sources": sorted(list(unique_sources)),
            "collection_name": settings.COLLECTION_NAME,
            "has_gemini_key": bool(self.embedding_fn.api_key)
        }

# Global singleton instance
vector_store = VectorStoreManager()
