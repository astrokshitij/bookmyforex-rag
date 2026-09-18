import time
import logging
import threading
from typing import List, Dict, Any, Optional
from app.config import settings
from app.vector_store import vector_store

logger = logging.getLogger("rag_engine")
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = """You are the official BookMyForex Internal Support AI Assistant. Your role is to provide precise, grounded operational, promotional, and procedural guidance to BookMyForex customer support agents, operations staff, and compliance representatives.

GROUNDING & GUARDRAILS:
1. Grounding & Semantic Reasoning: Answer using the provided Context below. You may use semantic reasoning — if the context addresses the topic through related terms, synonyms, or equivalent concepts (e.g., airport transfer / cab voucher), treat it as relevant. If the context contains relevant facts (such as offer existence, eligibility, discount amount, or bundled perks) but lacks exhaustive step-by-step instructions, provide all known facts clearly based on the context. Do NOT use outside knowledge or speculate beyond what is documented.
2. Compliance Fallback: ONLY use the fallback if the provided context genuinely contains NO information that is relevant to the query — not even indirectly or partially. When you must fall back, respond with EXACTLY this statement:
"{fallback_statement}"
Do not add pleasantries or partial guesses before or after this fallback sentence.
3. Source File Citations: Every factual answer MUST cite the source file and section from which the facts were obtained (e.g., `[Offers.md: Section 3.A]` or `[gemini-code-1789542027610.md: Section 1]`).
4. Operational Alerts: When applicable, explicitly highlight critical DOs and DON'Ts, eligibility caveats, deadline windows (e.g. 60-day claim / 30-day payout), and mandatory documentation (e.g. physical FIR requirement).
5. Clean Output: NEVER output internal technical details, chunk IDs, similarity scores, match percentages, distance values, or database metadata in your response.
6. Format: Use clean markdown with clear bullet points, bold key terms, and code blocks for promo codes. At the bottom of valid answers, include a "📚 Sources Cited" section listing the referenced documents and sections.
"""


class ResponseCache:
    """
    Thread-safe in-memory cache with TTL expiration and LRU eviction.
    """
    def __init__(self, ttl_seconds: int = 3600, max_size: int = 500):
        self.ttl_seconds = ttl_seconds
        self.max_size = max_size
        self._cache: Dict[str, Dict[str, Any]] = {}
        self._lock = threading.Lock()
        self.hits = 0
        self.misses = 0

    def _make_key(self, query: str, filter_type: Optional[str] = None, history_len: int = 0) -> str:
        clean_q = " ".join(query.lower().strip().split())
        f_type = (filter_type or "").lower().strip()
        return f"{clean_q}__filter_{f_type}__hist_{history_len}"

    def get(self, query: str, filter_type: Optional[str] = None, history_len: int = 0) -> Optional[Dict[str, Any]]:
        key = self._make_key(query, filter_type, history_len)
        now = time.time()
        with self._lock:
            if key in self._cache:
                entry = self._cache[key]
                if now - entry["timestamp"] < self.ttl_seconds:
                    entry["last_accessed"] = now
                    self.hits += 1
                    cached_data = dict(entry["data"])
                    cached_data["cached"] = True
                    return cached_data
                else:
                    del self._cache[key]
            self.misses += 1
            return None

    def set(self, query: str, data: Dict[str, Any], filter_type: Optional[str] = None, history_len: int = 0):
        if data.get("response_path") in ("llm_rate_limited", "llm_all_retries_failed"):
            return

        key = self._make_key(query, filter_type, history_len)
        now = time.time()
        with self._lock:
            if len(self._cache) >= self.max_size and key not in self._cache:
                oldest_key = min(self._cache.keys(), key=lambda k: self._cache[k]["last_accessed"])
                del self._cache[oldest_key]

            self._cache[key] = {
                "data": data,
                "timestamp": now,
                "last_accessed": now
            }

    def clear(self):
        with self._lock:
            self._cache.clear()
            logger.info("Cleared RAG response cache.")

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total = self.hits + self.misses
            return {
                "cached_entries": len(self._cache),
                "max_size": self.max_size,
                "ttl_seconds": self.ttl_seconds,
                "hits": self.hits,
                "misses": self.misses,
                "hit_ratio": round(self.hits / total, 3) if total > 0 else 0.0
            }


class RAGEngine:
    """
    RAG Engine with Hybrid Search (BM25 + Chroma Dense RRF),
    multi-turn memory, query rewriting, and multi-key round-robin Groq LLM failover.
    """

    def __init__(self):
        self.model_name = settings.GENERATION_MODEL
        self._api_keys: List[str] = []
        self._clients: Dict[str, Any] = {}
        self._key_index = 0
        self._lock = threading.Lock()
        self.cache = ResponseCache(ttl_seconds=3600, max_size=500)
        self._load_keys()

    def _load_keys(self):
        """Load API keys from GROQ_API_KEYS (comma-separated) or single GROQ_API_KEY."""
        keys = []
        if settings.GROQ_API_KEYS:
            keys = [k.strip() for k in settings.GROQ_API_KEYS.split(",") if k.strip()]
        if settings.GROQ_API_KEY and settings.GROQ_API_KEY not in keys:
            keys.append(settings.GROQ_API_KEY)

        self._api_keys = keys
        self._init_clients()
        logger.info(f"Loaded {len(self._api_keys)} Groq API key(s) for round-robin rotation.")

    def _init_clients(self):
        """Initialize a Groq client for each API key."""
        self._clients = {}
        if not self._api_keys:
            return

        try:
            from groq import Groq
            for key in self._api_keys:
                self._clients[key] = Groq(api_key=key)
            logger.info(f"Initialized {len(self._clients)} Groq client(s) (model: {self.model_name}).")
        except ImportError:
            logger.error("Groq SDK not installed. Run: pip install groq")
        except Exception as e:
            logger.warning(f"Failed to initialize Groq clients: {e}")

    def _next_client(self):
        """Thread-safe round-robin key selection. Returns (key_label, client)."""
        if not self._api_keys:
            return None, None
        with self._lock:
            idx = self._key_index % len(self._api_keys)
            self._key_index += 1
        key = self._api_keys[idx]
        return f"key_{idx + 1}/{len(self._api_keys)}", self._clients.get(key)

    def reload_api_keys(self, keys_csv: str):
        """Updates Groq clients with new comma-separated API keys."""
        self._api_keys = [k.strip() for k in keys_csv.split(",") if k.strip()]
        self._key_index = 0
        self._init_clients()

    def reload_api_key(self, key: str):
        self.reload_api_keys(key)

    @property
    def groq_api_key(self):
        return self._api_keys[0] if self._api_keys else None

    @property
    def key_count(self):
        return len(self._api_keys)

    def _contextualize_query(self, query: str, history: Optional[List[Dict[str, str]]]) -> str:
        """
        Rewrites conversational follow-up queries using prior turns so vector/BM25 search
        captures full context (e.g., 'what about for existing users?' -> 'what about for existing users? [Context: Big Forex Sale]').
        """
        if not history or len(history) == 0:
            return query

        recent_user_turns = [h["content"] for h in history if h.get("role") == "user"]
        if not recent_user_turns:
            return query

        last_user_query = recent_user_turns[-1].strip()
        ref_words = (
            "it", "this", "that", "these", "those", "same", "offer", "discount",
            "cashback", "slab", "promo", "code", "what about", "and for", "how much",
            "eligibility", "terms", "limit", "charges", "fees", "rate", "procedure",
            "process", "timeline", "validity", "applicable", "requirement", "documents"
        )
        query_lower = query.lower()

        needs_context = (len(query.split()) <= 8) or any(w in query_lower for w in ref_words)

        if needs_context and last_user_query:
            return f"{query} [Topic context: {last_user_query[:120]}]"

        return query

    def generate_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ) -> Dict[str, Any]:
        """
        Executes the optimized RAG pipeline:
        1. Check In-Memory TTL/LRU Response Cache (Instant Return)
        2. Contextual Query Rewriting for Multi-Turn Follow-Ups
        3. Hybrid Search (Chroma Dense Vectors + BM25 Sparse with RRF)
        4. Strict Grounding Guardrail Prompting with Conversation Memory
        5. Groq LLM Generation with multi-key round-robin rotation & 429 failover
        6. Store in Cache & Return
        """
        history_list = history or []
        history_len = len(history_list)

        # 1. Check Cache
        cached_resp = self.cache.get(query, filter_type=filter_type, history_len=history_len)
        if cached_resp:
            logger.info(f"Cache HIT for query: '{query[:60]}'")
            return cached_resp

        k = top_k or settings.TOP_K
        filter_dict = {"document_type": filter_type} if filter_type else None

        # 2. Contextualize query for search
        search_query = self._contextualize_query(query, history_list)

        # 3. Hybrid Retrieval (BM25 + Dense Chroma with Reciprocal Rank Fusion)
        retrieved_chunks = vector_store.hybrid_query(
            query_text=search_query,
            top_k=k,
            filter_metadata=filter_dict
        )

        # Structure clean citations for UI
        structured_citations = []
        context_parts = []

        for item in retrieved_chunks:
            meta = item.get("metadata", {})
            content = item.get("content", "")

            structured_citations.append({
                "source_file": meta.get("source_file", "Unknown"),
                "document_title": meta.get("document_title", "Document"),
                "document_type": meta.get("document_type", "kb"),
                "section_title": meta.get("section_title", "General"),
                "last_updated": meta.get("last_updated", ""),
                "snippet": content[:300] + "..." if len(content) > 300 else content
            })

            context_parts.append(
                f"--- SOURCE: {meta.get('source_file')} | SECTION: {meta.get('section_title')} ---\n{content}\n"
            )

        context_str = "\n".join(context_parts)

        # If hybrid search returned no relevant chunks
        if not retrieved_chunks:
            result = {
                "answer": settings.COMPLIANCE_FALLBACK,
                "grounded": False,
                "fallback_triggered": True,
                "citations": [],
                "model_used": self.model_name,
                "retrieval_count": 0,
                "response_path": "no_chunks_retrieved",
                "cached": False
            }
            self.cache.set(query, result, filter_type=filter_type, history_len=history_len)
            return result

        # Check if any Groq API keys are configured
        if not self._api_keys or not self._clients:
            answer_text = (
                f"⚠️ **Groq API Key is not configured.**\n\n"
                f"Please set `GROQ_API_KEY` or `GROQ_API_KEYS` in your `.env` file "
                f"or click the ⚙️ **Settings** button to input your key(s).\n\n"
                f"### Retrieved Context Preview from Vector Store:\n"
            )
            for c in structured_citations[:3]:
                answer_text += f"\n- **{c['source_file']}** ({c['section_title']}):\n> {c['snippet']}\n"

            return {
                "answer": answer_text,
                "grounded": True,
                "fallback_triggered": False,
                "citations": structured_citations,
                "model_used": "offline_preview",
                "cached": False
            }

        # 4. Prepare LLM prompt with grounded context and multi-turn history
        formatted_sys_prompt = SYSTEM_PROMPT.format(
            fallback_statement=settings.COMPLIANCE_FALLBACK
        )

        messages = [
            {"role": "system", "content": formatted_sys_prompt}
        ]

        # Insert recent conversation history (up to last 4 turns)
        if history_list:
            for turn in history_list[-4:]:
                r = turn.get("role")
                c = turn.get("content")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})

        user_content = (
            f"KNOWLEDGE BASE CONTEXT:\n\n{context_str}\n\n"
            f"CUSTOMER SUPPORT QUERY: {query}\n\n"
            f"Using the context above, provide an accurate, grounded answer with source citations. "
            f"If the context addresses the topic — even through synonyms or related concepts — synthesize "
            f"a clear answer from it. Only use the compliance fallback if the context contains genuinely "
            f"no relevant information: \"{settings.COMPLIANCE_FALLBACK}\""
        )
        messages.append({"role": "user", "content": user_content})

        # 5. Round-robin Groq call with failover across all keys
        total_attempts = len(self._api_keys) * 2
        last_exception = None
        keys_exhausted = set()

        for attempt in range(1, total_attempts + 1):
            key_label, client = self._next_client()

            if not client:
                break

            try:
                response = client.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    temperature=0.2,
                    max_tokens=2048
                )
                answer = response.choices[0].message.content.strip()
                fallback_triggered = (settings.COMPLIANCE_FALLBACK.lower() in answer.lower())

                result = {
                    "answer": answer,
                    "grounded": not fallback_triggered,
                    "fallback_triggered": fallback_triggered,
                    "citations": structured_citations,
                    "model_used": self.model_name,
                    "retrieval_count": len(retrieved_chunks),
                    "response_path": "llm_success",
                    "keys_available": len(self._api_keys),
                    "cached": False
                }

                # Store in cache
                self.cache.set(query, result, filter_type=filter_type, history_len=history_len)
                return result

            except Exception as e:
                last_exception = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

                if is_rate_limit:
                    keys_exhausted.add(key_label)
                    logger.info(f"Groq {key_label} rate-limited, rotating to next key... "
                                f"({len(keys_exhausted)}/{len(self._api_keys)} exhausted)")

                    if len(keys_exhausted) >= len(self._api_keys):
                        logger.warning("All Groq keys rate-limited. Backing off 5s...")
                        time.sleep(5)
                        keys_exhausted.clear()
                else:
                    logger.warning(f"Groq {key_label} attempt {attempt} failed: {err_str}")
                    time.sleep(1)

        # All attempts exhausted
        logger.error(f"All Groq API attempts failed ({total_attempts} tries): {last_exception}")
        err_msg = str(last_exception) if last_exception else ""
        is_rate_limited = "429" in err_msg or "rate_limit" in err_msg.lower()

        if is_rate_limited:
            answer_text = (
                f"⚠️ **All {len(self._api_keys)} Groq API key(s) are rate-limited.** "
                f"Please wait a moment and try again.\n\n"
                "Your question was received and the relevant knowledge base context was found — "
                "the AI just couldn't generate a response due to rate limiting."
            )
        else:
            answer_text = settings.COMPLIANCE_FALLBACK

        return {
            "answer": answer_text,
            "grounded": False,
            "fallback_triggered": not is_rate_limited,
            "citations": structured_citations,
            "model_used": self.model_name,
            "retrieval_count": len(retrieved_chunks),
            "response_path": "llm_rate_limited" if is_rate_limited else "llm_all_retries_failed",
            "error": err_msg[:500] if err_msg else None,
            "keys_available": len(self._api_keys),
            "cached": False
        }

# Global singleton instance
rag_engine = RAGEngine()
