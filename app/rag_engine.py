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
1. Grounding: Answer using the provided Context below. You may use semantic reasoning — if the context addresses the topic through related terms, synonyms, or equivalent concepts, treat it as relevant and answer based on it. Do NOT use outside knowledge, speculate, or extrapolate beyond what can be reasonably inferred from the documents.
2. Compliance Fallback: ONLY use the fallback if the provided context genuinely contains NO information that is relevant to the query — not even indirectly. If there is relevant context, synthesize and answer from it. When you must fall back, respond with EXACTLY this statement:
"{fallback_statement}"
Do not add pleasantries or partial guesses before or after this fallback sentence.
3. Source File Citations: Every factual answer MUST cite the source file and section from which the facts were obtained (e.g., `[Offers.md: Section 3.A]` or `[gemini-code-1789542027610.md: Section 1]`).
4. Operational Alerts: When applicable, explicitly highlight critical DOs and DON'Ts, eligibility caveats, deadline windows (e.g. 60-day claim / 30-day payout), and mandatory documentation (e.g. physical FIR requirement).
5. Clean Output: NEVER output internal technical details, chunk IDs, similarity scores, match percentages, distance values, or database metadata in your response.
6. Format: Use clean markdown with clear bullet points, bold key terms, and code blocks for promo codes. At the bottom of valid answers, include a "📚 Sources Cited" section listing the referenced documents and sections.
"""


class RAGEngine:
    """
    RAG Engine with multi-key round-robin Groq API support.
    
    Supports multiple Groq API keys for higher effective rate limits:
    - Round-robin distributes requests evenly across keys
    - On 429 rate limit, instantly fails over to the next key (zero wait)
    - Only backs off if ALL keys are exhausted
    """

    def __init__(self):
        self.model_name = settings.GENERATION_MODEL
        self._api_keys: List[str] = []
        self._clients: Dict[str, Any] = {}
        self._key_index = 0
        self._lock = threading.Lock()  # Thread-safe round-robin
        self._load_keys()

    def _load_keys(self):
        """Load API keys from GROQ_API_KEYS (comma-separated) or single GROQ_API_KEY."""
        keys = []

        # Multi-key: GROQ_API_KEYS=key1,key2,key3
        if settings.GROQ_API_KEYS:
            keys = [k.strip() for k in settings.GROQ_API_KEYS.split(",") if k.strip()]

        # Backward compat: single GROQ_API_KEY (add if not already in multi-key list)
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

    # Backward compat alias
    def reload_api_key(self, key: str):
        self.reload_api_keys(key)

    @property
    def groq_api_key(self):
        """Backward compat: returns first key or None."""
        return self._api_keys[0] if self._api_keys else None

    @property
    def key_count(self):
        return len(self._api_keys)

    def generate_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the RAG pipeline:
        1. Retrieval from ChromaDB Vector Store (Gemini embeddings)
        2. Strict Grounding Guardrail Prompting
        3. Groq LLM generation with round-robin key rotation & failover
        """
        k = top_k or settings.TOP_K
        filter_dict = {"document_type": filter_type} if filter_type else None

        # 1. Retrieve relevant chunks
        retrieved_chunks = vector_store.query(
            query_text=query,
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

        # If vector store returned no relevant chunks
        if not retrieved_chunks:
            return {
                "answer": settings.COMPLIANCE_FALLBACK,
                "grounded": False,
                "fallback_triggered": True,
                "citations": [],
                "model_used": self.model_name,
                "retrieval_count": 0,
                "response_path": "no_chunks_retrieved"
            }

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
                "model_used": "offline_preview"
            }

        # 2. Prepare LLM prompt
        formatted_sys_prompt = SYSTEM_PROMPT.format(
            fallback_statement=settings.COMPLIANCE_FALLBACK
        )

        user_content = (
            f"KNOWLEDGE BASE CONTEXT:\n\n{context_str}\n\n"
            f"CUSTOMER SUPPORT QUERY: {query}\n\n"
            f"Using the context above, provide an accurate, grounded answer with source citations. "
            f"If the context addresses the topic — even through synonyms or related concepts — synthesize "
            f"a clear answer from it. Only use the compliance fallback if the context contains genuinely "
            f"no relevant information: \"{settings.COMPLIANCE_FALLBACK}\""
        )

        messages = [
            {"role": "system", "content": formatted_sys_prompt},
            {"role": "user", "content": user_content}
        ]

        # 3. Round-robin Groq call with failover across all keys
        #    On 429: instantly try next key (zero wait)
        #    On other errors: retry same key with short backoff
        #    Only fail after ALL keys exhausted + retry attempts

        total_attempts = len(self._api_keys) * 2  # 2 full rotations max
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

                return {
                    "answer": answer,
                    "grounded": not fallback_triggered,
                    "fallback_triggered": fallback_triggered,
                    "citations": structured_citations,
                    "model_used": self.model_name,
                    "retrieval_count": len(retrieved_chunks),
                    "response_path": "llm_success",
                    "keys_available": len(self._api_keys)
                }

            except Exception as e:
                last_exception = e
                err_str = str(e)
                is_rate_limit = "429" in err_str or "rate_limit" in err_str.lower()

                if is_rate_limit:
                    # Rate limited — instantly try next key (zero delay)
                    keys_exhausted.add(key_label)
                    logger.info(f"Groq {key_label} rate-limited, rotating to next key... "
                                f"({len(keys_exhausted)}/{len(self._api_keys)} exhausted)")

                    if len(keys_exhausted) >= len(self._api_keys):
                        # ALL keys exhausted — short backoff before second rotation
                        logger.warning("All Groq keys rate-limited. Backing off 5s...")
                        time.sleep(5)
                        keys_exhausted.clear()
                else:
                    # Non-rate-limit error — short backoff and retry
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
            "keys_available": len(self._api_keys)
        }

# Global singleton instance
rag_engine = RAGEngine()
