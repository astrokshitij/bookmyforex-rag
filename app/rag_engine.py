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
    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or settings.GEMINI_API_KEY
        self.model_name = settings.GENERATION_MODEL
        self._llm_client = None
        self._init_llm()

    def _init_llm(self):
        if not self.api_key:
            return

        try:
            from google import genai
            self._llm_client = genai.Client(api_key=self.api_key)
            self._sdk_type = "google_genai"
            logger.info("Initialized Google GenAI client for generation.")
        except Exception:
            try:
                import google.generativeai as gai
                gai.configure(api_key=self.api_key)
                self._llm_client = gai
                self._sdk_type = "google_generativeai"
                logger.info("Initialized google.generativeai client for generation.")
            except Exception as e:
                logger.warning(f"Failed to initialize Gemini generation client: {e}")
                self._llm_client = None

    def reload_api_key(self, api_key: str):
        """Updates LLM client with a new API key."""
        self.api_key = api_key
        self._init_llm()
        vector_store.reload_api_key(api_key)

    def generate_response(
        self,
        query: str,
        top_k: Optional[int] = None,
        filter_type: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes the RAG pipeline:
        1. Retrieval from Vector Store
        2. Strict Grounding Guardrail Prompting
        3. Gemini generation with automatic retry & citations
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

        # If vector store is empty
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

        # Check if API Key is available
        if not self.api_key or not self._llm_client:
            answer_text = (
                f"⚠️ **Gemini API Key is not configured.**\n\n"
                f"Please set `GEMINI_API_KEY` in your `.env` file or click the ⚙️ **Settings** button in the top bar to input your key.\n\n"
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

        # 3. Call Gemini LLM with automatic exponential backoff retry (up to 3 attempts)
        max_retries = 3
        last_exception = None

        for attempt in range(1, max_retries + 1):
            try:
                if self._sdk_type == "google_genai":
                    response = self._llm_client.models.generate_content(
                        model=self.model_name,
                        contents=user_content,
                        config={
                            "system_instruction": formatted_sys_prompt,
                            "temperature": 0.2
                        }
                    )
                    answer = response.text.strip()
                else:
                    model = self._llm_client.GenerativeModel(
                        model_name=self.model_name,
                        system_instruction=formatted_sys_prompt,
                        generation_config={"temperature": 0.2}
                    )
                    response = model.generate_content(user_content)
                    answer = response.text.strip()

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
                logger.warning(f"Gemini API attempt {attempt}/{max_retries} failed: {err_str}")
                
                # Check for 503 / 429 / rate limit / high demand errors and retry with exponential backoff
                if attempt < max_retries:
                    backoff_delay = 2 ** (attempt - 1)  # 1s, 2s, 4s...
                    time.sleep(backoff_delay)

        # If all retry attempts failed, log error and return user-friendly fallback
        logger.error(f"All {max_retries} Gemini API retry attempts failed: {last_exception}")
        return {
            "answer": settings.COMPLIANCE_FALLBACK,
            "grounded": False,
            "fallback_triggered": True,
            "citations": structured_citations,
            "model_used": self.model_name,
            "retrieval_count": len(retrieved_chunks),
            "response_path": "llm_all_retries_failed",
            "error": str(last_exception)[:500] if last_exception else None
        }

# Global singleton instance
rag_engine = RAGEngine()
