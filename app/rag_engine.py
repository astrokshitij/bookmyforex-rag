import time
import logging
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
    def __init__(self):
        self.groq_api_key = settings.GROQ_API_KEY
        self.model_name = settings.GENERATION_MODEL
        self._llm_client = None
        self._init_llm()

    def _init_llm(self):
        if not self.groq_api_key:
            logger.warning("GROQ_API_KEY is not set. LLM generation will be unavailable.")
            return

        try:
            from groq import Groq
            self._llm_client = Groq(api_key=self.groq_api_key)
            logger.info(f"Initialized Groq client for generation (model: {self.model_name}).")
        except ImportError:
            logger.error("Groq SDK not installed. Run: pip install groq")
            self._llm_client = None
        except Exception as e:
            logger.warning(f"Failed to initialize Groq client: {e}")
            self._llm_client = None

    def reload_api_key(self, groq_api_key: str):
        """Updates LLM client with a new Groq API key."""
        self.groq_api_key = groq_api_key
        self._init_llm()

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
        3. Groq LLM generation with automatic retry & citations
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

        # Check if Groq API Key is available
        if not self.groq_api_key or not self._llm_client:
            answer_text = (
                f"⚠️ **Groq API Key is not configured.**\n\n"
                f"Please set `GROQ_API_KEY` in your `.env` file or click the ⚙️ **Settings** button in the top bar to input your key.\n\n"
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

        # 3. Call Groq LLM with retry logic
        max_retries = 3
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self._llm_client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": formatted_sys_prompt},
                        {"role": "user", "content": user_content}
                    ],
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
                    "response_path": "llm_success"
                }

            except Exception as e:
                last_exception = e
                err_str = str(e)
                logger.warning(f"Groq API attempt {attempt}/{max_retries} failed: {err_str}")
                
                if attempt < max_retries:
                    if "429" in err_str or "rate_limit" in err_str.lower():
                        backoff_delay = 10 * attempt
                        logger.info(f"Rate limited. Backing off {backoff_delay}s before retry...")
                    else:
                        backoff_delay = 2 ** (attempt - 1)
                    time.sleep(backoff_delay)

        # If all retry attempts failed
        logger.error(f"All {max_retries} Groq API retry attempts failed: {last_exception}")
        err_msg = str(last_exception) if last_exception else ""
        is_rate_limited = "429" in err_msg or "rate_limit" in err_msg.lower()

        if is_rate_limited:
            answer_text = (
                "⚠️ **Groq API rate limit reached.** Please wait a moment and try again.\n\n"
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
            "error": err_msg[:500] if err_msg else None
        }

# Global singleton instance
rag_engine = RAGEngine()
