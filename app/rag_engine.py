import time
import logging
import threading
from typing import List, Dict, Any, Optional, Tuple
from app.config import settings
from app.vector_store import vector_store

logger = logging.getLogger("rag_engine")
logger.setLevel(logging.INFO)

SYSTEM_PROMPT = """You are the official BookMyForex Internal Support AI Assistant. Your role is to provide precise, grounded operational, promotional, and procedural guidance to BookMyForex customer support agents, operations staff, and compliance representatives.

GROUNDING & GUARDRAILS:
1. Grounding & Semantic Reasoning: Answer using the provided Context below. You may use semantic reasoning — if the context addresses the topic through related terms, synonyms, or equivalent concepts (e.g., airport transfer / cab voucher), treat it as relevant. If the context contains relevant facts (such as offer existence, eligibility, discount amount, or bundled perks) but lacks exhaustive step-by-step instructions, provide all known facts clearly based on the context. Do NOT use outside knowledge or speculate beyond what is documented.
2. Exhaustive Multi-Offer Synthesis (No Exceptions): When asked to list, show, or summarize "all" offers, promotions, promo codes, or perks, you MUST list EVERY SINGLE offer, campaign, bundled card deliverable, and partner perk present across the provided context without skipping any:
   - Category 1: Campaign & Promo-Code Offers (`BIGFXSALE` / India's Biggest Forex Sale, `REMPITSPL` / `REMITSPL` / Education Remittance Special, Zero-Fee Remittance Offer)
   - Category 2: Partner & Visa Value-Added Perks (Free International Airport Lounges, Free International SIM / eSIM, ₹500 First Payment Voucher, Complimentary Digital ISIC Student Card, Visa Power Travel Rewards & ₹10,000 Jetsetter Bonus, Zero-Surcharge Allpoint ATMs, Medical Tourism & City Experiences)
   - Category 3: New-Card Bundled Travel Deliverables (MakeMyTrip ₹500 Airport Transfer Cab Voucher, Up to ₹6,000 off Flights, Up to 30% off Hotels, Up to 25% off Tours & Attractions, ₹250 Visa Services Gift Card)
   For every entry, include the exact Promo Code (or N/A), Product / Service, Minimum Spend / Transfer, Key Benefits, and Expiry / Validity Date.
   CRITICAL EXPIRY / VALIDITY RULE: If an explicit calendar expiry date is not stated in the document for an item (such as ongoing bundled new-card deliverables), state "Ongoing / Bundled with New Card (verify live page)" — NEVER omit or drop an offer simply because an expiry date is not listed!
3. Compliance Fallback: ONLY use the fallback if the provided context genuinely contains NO information that is relevant to the query — not even indirectly or partially. When you must fall back, respond with EXACTLY this statement:
"{fallback_statement}"
Do not add pleasantries or partial guesses before or after this fallback sentence.
4. Source File Citations: Every factual answer MUST cite the source file and section from which the facts were obtained (e.g., `[offers.md: Section 3.A]` or `[current-offers.md: Section 2]`).
5. Operational Alerts: When applicable, explicitly highlight critical DOs and DON'Ts, eligibility caveats, deadline windows (e.g. 60-day claim / 30-day payout, bookings till 15th Sep vs from 16th Sep MyCash), and mandatory documentation.
6. Clean Output: NEVER output internal technical details, chunk IDs, similarity scores, match percentages, distance values, or database metadata in your response.
7. Format: Use clean markdown with clear tables or bullet points, bold key terms, and code blocks for promo codes. At the bottom of valid answers, include a "📚 Sources Cited" section listing the referenced documents and sections.
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
            import httpx
            from groq import Groq
            http_client = httpx.Client(verify=False, timeout=30.0)
            for key in self._api_keys:
                self._clients[key] = Groq(api_key=key, http_client=http_client)
            logger.info(f"Initialized {len(self._clients)} Groq client(s) (model: {self.model_name}).")
        except ImportError:
            logger.error("Groq or httpx SDK not installed. Run: pip install groq httpx")
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

    def _get_aggregation_chunks(self, query: str) -> List[Dict[str, Any]]:
        """
        Detects broad domain aggregation questions (e.g., all offers, all KYC documents,
        all TCS rules, all fees, all card variants) and returns canonical chunks.
        Guarantees 100% recall across multi-document knowledge topics without relying on top_k cutoffs.
        """
        import re
        q = query.lower()
        canonical_chunks: List[Dict[str, Any]] = []

        # 1. Broad Offers / Deals / Perks / Discounts Aggregation
        is_offers_agg = (
            any(phrase in q for phrase in (
                "all offer", "all the offer", "every offer", "list all offer", "all deal",
                "all promo", "all discount", "all perk", "all benefit", "all coupon",
                "current offer", "active offer", "available offer", "offers and their expiry",
                "offers currently live", "what are the offer", "what are all the offer",
                "give me all offer", "show all offer", "list of all the offer", "all current",
                "list the offer", "list all the offer", "what offers", "what are all offers",
                "give me list of all the offer"
            ))
            or (re.search(r'\b(all|every|list|summary|overview|what\s+are)\b', q) and re.search(r'\b(offer|offers|deal|deals|perk|perks|promo|promos|discount|discounts|cashback)\b', q))
        )
        if is_offers_agg:
            canonical_chunks.extend(
                vector_store.get_canonical_chunks(
                    source_files=["offers.md", "current-offers.md"],
                    exclude_sections=["Sources"]
                )
            )

        # 2. Broad KYC / Documentation Aggregation
        is_kyc_agg = (
            re.search(r'\b(all\s+kyc|all\s+documents|all\s+the\s+documents|mandatory\s+documents|documents\s+needed|documents\s+required|kyc\s+requirements|paperwork)\b', q)
            or (re.search(r'\b(all|what|list|every)\b', q) and re.search(r'\b(kyc|document|documents|proof|proofs)\b', q))
        )
        if is_kyc_agg:
            canonical_chunks.extend(
                vector_store.get_canonical_chunks(
                    source_files=["money-transfer.md", "currency-exchange.md", "forex-card.md", "tcs-and-regulations.md"],
                    section_keywords=["document", "kyc", "mandatory", "eligibility", "purpose"]
                )
            )

        # 3. Broad TCS / Regulatory Tax Aggregation
        is_tcs_agg = (
            re.search(r'\b(all\s+tcs|tcs\s+rules|tcs\s+rates|tcs\s+slabs|tcs\s+structure|tax\s+rules|lrs\s+limit|all\s+tax)\b', q)
            or (re.search(r'\b(all|what|list|overview|summary)\b', q) and re.search(r'\b(tcs|tax|taxes)\b', q))
        )
        if is_tcs_agg:
            canonical_chunks.extend(
                vector_store.get_canonical_chunks(
                    source_files=["tcs-and-regulations.md"],
                    exclude_sections=["Sources"]
                )
            )

        # 4. Broad Fees & Charges Aggregation
        is_fees_agg = (
            re.search(r'\b(all\s+fees|all\s+charges|fee\s+structure|charges\s+list|schedule\s+of\s+charges|all\s+costs)\b', q)
            or (re.search(r'\b(all|what|list|overview)\b', q) and re.search(r'\b(fee|fees|charge|charges|cost|costs)\b', q))
        )
        if is_fees_agg:
            canonical_chunks.extend(
                vector_store.get_canonical_chunks(
                    source_files=["fees-and-charges.md"],
                    exclude_sections=["Sources"]
                )
            )

        # 5. Broad Card Variants / Comparison Aggregation
        is_cards_agg = (
            re.search(r'\b(all\s+cards|all\s+forex\s+cards|card\s+variants|compare\s+cards|which\s+cards|different\s+cards)\b', q)
        )
        if is_cards_agg:
            canonical_chunks.extend(
                vector_store.get_canonical_chunks(
                    source_files=["forex-card.md", "offers.md"],
                    section_keywords=["variant", "specifications", "multi-currency", "global usd"]
                )
            )

        return canonical_chunks

    def _retrieve_and_assemble_context(
        self,
        query: str,
        history_list: Optional[List[Dict[str, str]]] = None,
        top_k: Optional[int] = None,
        filter_type: Optional[str] = None
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], str]:
        """
        Unified retrieval and context assembly combining:
        1. Contextual query rewriting
        2. Domain-specific query expansion
        3. Deterministic canonical chunk retrieval for broad aggregation queries
        4. Dense + Sparse Hybrid Search with Reciprocal Rank Fusion (RRF)
        5. Deduplication and structured citation formatting
        """
        normalized_query = self._normalize_query_terms(query)
        context_query = self._contextualize_query(normalized_query, history_list or [])
        expanded_query = self._expand_query_intent(context_query)
        filter_dict = {"document_type": filter_type} if filter_type else None

        # 1. Fetch any domain aggregation chunks for broad queries
        canonical_chunks = self._get_aggregation_chunks(normalized_query)
        is_broad_query = bool(canonical_chunks) or any(
            w in query.lower() for w in ("all", "list", "every", "summary", "overview", "offers", "perks",
                                         "promotions", "discounts", "codes", "cashback", "compare", "deals")
        )
        k = top_k or (18 if is_broad_query else 6)

        # 2. Multi-Query Hybrid Retrieval
        hybrid_chunks = vector_store.hybrid_query(
            query_text=expanded_query,
            top_k=k,
            filter_metadata=filter_dict
        )

        # 3. Merge canonical chunks + hybrid chunks, preserving uniqueness by chunk_id
        seen_chunk_ids = set()
        retrieved_chunks: List[Dict[str, Any]] = []

        # Canonical chunks first (guarantees 100% presence)
        for c in canonical_chunks:
            cid = c.get("chunk_id")
            if cid and cid not in seen_chunk_ids:
                seen_chunk_ids.add(cid)
                retrieved_chunks.append(c)

        # Hybrid search chunks appended up to max limit
        max_total = max(k, len(canonical_chunks) + 4)
        for c in hybrid_chunks:
            cid = c.get("chunk_id")
            if cid and cid not in seen_chunk_ids and len(retrieved_chunks) < max_total:
                seen_chunk_ids.add(cid)
                retrieved_chunks.append(c)

        # 4. Structure clean citations and context string
        structured_citations: List[Dict[str, Any]] = []
        context_parts: List[str] = []

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
        return retrieved_chunks, structured_citations, context_str

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
        2. Unified Hybrid & Deterministic Canonical Retrieval
        3. Strict Grounding Guardrail Prompting with Conversation Memory
        4. Groq LLM Generation with multi-key round-robin rotation & 429 failover
        5. Store in Cache & Return
        """
        history_list = history or []
        history_len = len(history_list)

        # 1. Check Cache
        cached_resp = self.cache.get(query, filter_type=filter_type, history_len=history_len)
        if cached_resp:
            logger.info(f"Cache HIT for query: '{query[:60]}'")
            return cached_resp

        # 2. Retrieve & assemble context
        retrieved_chunks, structured_citations, context_str = self._retrieve_and_assemble_context(
            query=query,
            history_list=history_list,
            top_k=top_k,
            filter_type=filter_type
        )

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
                    max_tokens=4096
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

        # Try Gemini fallback before failing
        gemini_answer = self._generate_with_gemini(
            system_instruction=formatted_sys_prompt,
            prompt_text=user_content
        )
        if gemini_answer:
            fallback_triggered = (settings.COMPLIANCE_FALLBACK.lower() in gemini_answer.lower())
            result = {
                "answer": gemini_answer,
                "grounded": not fallback_triggered,
                "fallback_triggered": fallback_triggered,
                "citations": structured_citations,
                "model_used": "gemini-3.6-flash (fallback)",
                "retrieval_count": len(retrieved_chunks),
                "response_path": "gemini_fallback_success",
                "keys_available": len(self._api_keys),
                "cached": False
            }
            self.cache.set(query, result, filter_type=filter_type, history_len=history_len)
            return result

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

    def _generate_with_gemini(self, system_instruction: str, prompt_text: str) -> Optional[str]:
        """Fallback LLM generation using Google Gemini if Groq keys are temporarily unavailable."""
        gemini_key = settings.GEMINI_API_KEY
        if not gemini_key:
            return None

        try:
            from google import genai
            client = genai.Client(api_key=gemini_key)
            response = client.models.generate_content(
                model="gemini-2.5-flash",
                contents=prompt_text,
                config={"system_instruction": system_instruction, "temperature": 0.2}
            )
            if response and response.text:
                return response.text.strip()
        except Exception as e:
            logger.warning(f"Gemini fallback generation failed via google.genai: {e}")
            try:
                import google.generativeai as gai
                gai.configure(api_key=gemini_key)
                model = gai.GenerativeModel(
                    model_name="gemini-1.5-flash",
                    system_instruction=system_instruction
                )
                res = model.generate_content(prompt_text)
                if res and res.text:
                    return res.text.strip()
            except Exception as e2:
                logger.error(f"Gemini fallback generation failed completely: {e2}")
        return None

    def _normalize_query_terms(self, query: str) -> str:
        """Fixes common typos and normalizes domain terms."""
        import re
        q = query
        q = re.sub(r'\binternation\b', 'international', q, flags=re.IGNORECASE)
        q = re.sub(r'\be-sim\b', 'esim', q, flags=re.IGNORECASE)
        q = re.sub(r'\blounges\b', 'lounge', q, flags=re.IGNORECASE)
        q = re.sub(r'\bcanceltion\b', 'cancellation', q, flags=re.IGNORECASE)
        q = re.sub(r'\bremitence\b', 'remittance', q, flags=re.IGNORECASE)
        return q

    def _expand_query_intent(self, query: str) -> str:
        """
        Intelligent intent expander that translates human colloquial/slang phrasing,
        typos, and high-level questions into rich domain-specific search vectors.
        """
        import re
        q = query.lower()

        INTENT_RULES = [
            # 1. Emergency & Card Safety
            (r'\b(stolen|lost|stuck|eaten|eat|swallowed|swallow|block|blocked|theft|robbed|emergency|misplaced|captured)\b',
             'emergency SOP block card replace card physical FIR local police report 24x7 helpline zero liability insurance'),
            
            # 2. Student & Education Remittance
            (r'\b(student|study|studies|studying|university|college|tuition|semester|gic|admission|daughter|son|overseas studies|abroad education|campus)\b',
             'education remittance S0305 REMPITSPL REMITSPL Form A2 offer letter ISIC student card TCS 0.5% 0% education loan GIC Canada'),
            
            # 3. Perks, Freebies, Lounges, SIM & Rewards
            (r'\b(free|freebie|freebies|perk|perks|benefit|benefits|reward|rewards|lounge|lounges|sim|esim|gift|voucher|bonus|jetsetter|allpoint|cab|ride)\b',
             'partner perks free international SIM eSIM airport lounge USD 1200 razorpay 500 voucher ISIC student card Jetsetter bonus Allpoint ATM surcharge'),
            
            # 4. Promo Codes, Cashback & All Offers
            (r'\b(all\s+offer|all\s+the\s+offer|all\s+deal|all\s+promo|all\s+discount|list\s+all|every\s+offer|coupons|coupon|promo\s+code|cashback|mycash|savings|deals|all\s+perk|all\s+benefit)\b',
             'BIGFXSALE India Biggest Forex Sale REMPITSPL Education Remittance Special Zero Fee Remittance MakeMyTrip Airport Transfers Flights Hotels Tours Voucher Free International SIM eSIM Airport Lounges 500 First Payment ISIC student card Jetsetter bonus Allpoint ATM surcharge expiry validity'),
            
            # 5. TCS & Government Taxes
            (r'\b(tax|taxes|tcs|tcx|deduction|govt charge|government charge|7\s*lakh|20%|5%|exemption|pan)\b',
             'TCS Tax Collected at Source LRS 250000 USD limit education loan 0% 5% above 7 lakh 20% remittance'),
            
            # 6. Currency Cash Notes, Doorstep & Delivery
            (r'\b(cash|notes|doorstep|delivery|rate\s*lock|live\s*rate|interbank|markup|exchange\s*rate|cut-off|same-day)\b',
             'currency notes doorstep delivery same-day cut-off guaranteed rate lock interbank zero markup KYC passport PAN air ticket'),
            
            # 7. Card Comparison & Charges
            (r'\b(compare|difference|which card|yes bank|global usd|instarem|visa|reload|unload|atm charge|hidden fee|annual fee|markup fee)\b',
             'YES Bank Multi-Currency Forex Card Global USD Forex Card Instarem zero markup interbank rate zero reload unload fee'),
            
            # 8. Founder, Leadership & Company Information
            (r'\b(who\s+(is|are|owns|started|founded|runs)|founder|ceo|ownership|owner|headquarter|history|makemytrip|tripmoney|acquisition|cities|started)\b',
             'Sudarshan Motwani Founder CEO Nitin Motwani CTO 2012 MakeMyTrip TripMoney 2022 acquisition Gurugram 650 cities')
        ]

        expansions = []
        for pattern, keywords in INTENT_RULES:
            if re.search(pattern, q):
                expansions.append(keywords)

        if expansions:
            return f"{query} [{' | '.join(expansions)}]"
        return query

    def clean_for_customer(self, answer_text: str) -> str:
        """Strips internal citation brackets and compliance headers for clean customer pasting."""
        import re
        text = answer_text
        # Remove markdown citation footers like '📚 Sources Cited...'
        text = re.split(r'📚\s*\*?\*?Sources Cited\*?\*?', text)[0]
        # Remove [file.md: Section ...] brackets
        text = re.sub(r'\[[a-zA-Z0-9_\-\.]+\.md:[^\]]+\]', '', text)
        # Remove trailing dividers
        text = text.rstrip(" -\n*#")
        return text.strip()

    def generate_response_stream(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_type: Optional[str] = None,
        history: Optional[List[Dict[str, str]]] = None
    ):
        """
        Streaming generator yielding SSE JSON chunks:
        1. {"type": "init", "citations": [...], "retrieval_count": ...}
        2. {"type": "token", "delta": "..."}
        3. {"type": "done", "grounded": bool, "fallback_triggered": bool, "model_used": str, "citations": [...]}
        """
        import json

        history_list = history or []
        history_len = len(history_list)

        # 1. Check Cache
        cached_resp = self.cache.get(query, filter_type=filter_type, history_len=history_len)
        if cached_resp:
            logger.info(f"Stream Cache HIT for query: '{query[:60]}'")
            yield json.dumps({"type": "init", "citations": cached_resp.get("citations", []), "cached": True}) + "\n"
            yield json.dumps({"type": "token", "delta": cached_resp.get("answer", "")}) + "\n"
            yield json.dumps({
                "type": "done",
                "grounded": cached_resp.get("grounded", True),
                "fallback_triggered": cached_resp.get("fallback_triggered", False),
                "model_used": cached_resp.get("model_used", self.model_name),
                "citations": cached_resp.get("citations", []),
                "cached": True
            }) + "\n"
            return

        # 2. Retrieve & assemble context
        retrieved_chunks, structured_citations, context_str = self._retrieve_and_assemble_context(
            query=query,
            history_list=history_list,
            top_k=top_k,
            filter_type=filter_type
        )

        # Send initial event with citations immediately (<100ms)
        yield json.dumps({
            "type": "init",
            "citations": structured_citations,
            "retrieval_count": len(retrieved_chunks)
        }) + "\n"

        if not retrieved_chunks:
            fallback = settings.COMPLIANCE_FALLBACK
            yield json.dumps({"type": "token", "delta": fallback}) + "\n"
            yield json.dumps({
                "type": "done",
                "grounded": False,
                "fallback_triggered": True,
                "model_used": self.model_name,
                "citations": []
            }) + "\n"
            return

        formatted_sys_prompt = SYSTEM_PROMPT.format(fallback_statement=settings.COMPLIANCE_FALLBACK)
        user_content = (
            f"KNOWLEDGE BASE CONTEXT:\n\n{context_str}\n\n"
            f"CUSTOMER SUPPORT QUERY: {query}\n\n"
            f"Using the context above, provide an accurate, grounded answer with source citations. "
            f"If the context addresses the topic — even through synonyms or related concepts — synthesize "
            f"a clear answer from it. Only use the compliance fallback if the context contains genuinely "
            f"no relevant information: \"{settings.COMPLIANCE_FALLBACK}\""
        )

        messages = [{"role": "system", "content": formatted_sys_prompt}]
        if history_list:
            for turn in history_list[-4:]:
                r = turn.get("role")
                c = turn.get("content")
                if r in ("user", "assistant") and c:
                    messages.append({"role": r, "content": c})
        messages.append({"role": "user", "content": user_content})

        accumulated_answer = []
        stream_success = False

        # Attempt streaming via Groq
        if self._api_keys and self._clients:
            for _ in range(len(self._api_keys)):
                key_label, client = self._next_client()
                if not client:
                    break
                try:
                    stream = client.chat.completions.create(
                        model=self.model_name,
                        messages=messages,
                        temperature=0.2,
                        max_tokens=4096,
                        stream=True
                    )
                    for chunk in stream:
                        delta = chunk.choices[0].delta.content if chunk.choices and chunk.choices[0].delta else None
                        if delta:
                            accumulated_answer.append(delta)
                            yield json.dumps({"type": "token", "delta": delta}) + "\n"
                    stream_success = True
                    break
                except Exception as e:
                    logger.warning(f"Groq stream on {key_label} failed: {e}")

        # Fallback to Gemini streaming / generation if Groq didn't complete
        if not stream_success:
            gemini_ans = self._generate_with_gemini(formatted_sys_prompt, user_content)
            if gemini_ans:
                accumulated_answer = [gemini_ans]
                yield json.dumps({"type": "token", "delta": gemini_ans}) + "\n"
                stream_success = True

        full_text = "".join(accumulated_answer).strip() or settings.COMPLIANCE_FALLBACK
        fallback_triggered = (settings.COMPLIANCE_FALLBACK.lower() in full_text.lower())

        # Cache result
        result = {
            "answer": full_text,
            "grounded": not fallback_triggered,
            "fallback_triggered": fallback_triggered,
            "citations": structured_citations,
            "model_used": self.model_name,
            "retrieval_count": len(retrieved_chunks),
            "cached": False
        }
        self.cache.set(query, result, filter_type=filter_type, history_len=history_len)

        yield json.dumps({
            "type": "done",
            "grounded": not fallback_triggered,
            "fallback_triggered": fallback_triggered,
            "model_used": self.model_name,
            "citations": structured_citations
        }) + "\n"

    def _generate_with_gemini(self, system_instruction: str, prompt_text: str) -> Optional[str]:
        """Fallback LLM generation using Google Gemini when Groq is unavailable."""
        if not settings.GEMINI_API_KEY:
            return None
        try:
            from google import genai
            client = genai.Client(api_key=settings.GEMINI_API_KEY)
            for attempt in range(3):
                try:
                    resp = client.models.generate_content(
                        model="gemini-3.6-flash",
                        contents=prompt_text,
                        config={"system_instruction": system_instruction, "temperature": 0.2}
                    )
                    if resp and resp.text:
                        return resp.text.strip()
                except Exception as ex:
                    logger.warning(f"Gemini generation attempt {attempt + 1} failed: {ex}")
                    time.sleep(1.5 * (attempt + 1))
            return None
        except Exception as e:
            logger.warning(f"Gemini fallback generation failed: {e}")
            return None

# Global singleton instance
rag_engine = RAGEngine()
