# BookMyForex Internal Support RAG Application

An internal, strictly grounded Retrieval-Augmented Generation (RAG) assistant designed for BookMyForex customer support, operations, and compliance teams.

The system ingests all Markdown policy guidelines, campaign offers, product rules, and operational SOPs, indexes them into a local persistent ChromaDB vector database using Gemini's embedding model, and generates cited, compliance-verified answers using Gemini LLM.

---

## 🌟 Key Features

1. **Frontmatter & Clause-Aware Ingestion**:
   - Parses YAML frontmatter (`document_type`, `document_title`, `last_updated`, `status`).
   - Semantically chunks documents by markdown headings and section breaks without severing legal clauses, cashback slabs, or numbered SOP workflows.
   - Embeds hierarchical document breadcrumbs into each chunk for accurate contextual retrieval.

2. **ChromaDB Vector Store & Gemini Embeddings**:
   - Uses Google Gemini's `text-embedding-004` model to index chunks into local persistent vector storage (`./chroma_db`).
   - Includes automatic offline vectorizer fallback so you can preview chunking and test the app before supplying an API key.

3. **Strict Grounding Guardrails & Compliance Fallback**:
   - **Zero-Hallucination Policy**: Answers strictly from the provided knowledge base.
   - **Exact Compliance Escalate Rule**: If an answer is missing or cannot be authoritatively resolved, the assistant outputs:
     > *"I cannot find this information in the official BookMyForex guidelines. Please check with the compliance desk."*
   - **Mandatory Source Citations**: Every claim cites the source markdown file and section (e.g., `[Offers.md: Section 3.A]`).

4. **Internal Support UI**:
   - BookMyForex branded interface with quick inquiry pills for frequent support scenarios (cashback slabs, card stuck in ATM, insurance claim rules, 3 wrong PIN block duration).
   - Expandable source citation drawers showing exact matched text snippets and similarity scores.
   - Live Knowledge Base inspector and settings dialog to configure API keys dynamically.

---

## 📁 Project Structure

```
c:\RAG BookMyForex\
├── .env.example                     # Environment template (GEMINI_API_KEY, HOST, PORT)
├── requirements.txt                 # Python dependencies
├── run.bat                          # One-click Windows runner
├── run.ps1                          # PowerShell launcher
├── README.md                        # Documentation
│
├── app/
│   ├── __init__.py
│   ├── config.py                    # App configuration & environment settings
│   ├── ingestion.py                 # YAML parser & clause-preserving chunker
│   ├── vector_store.py              # ChromaDB client & Gemini text-embedding-004 integration
│   ├── rag_engine.py                # Gemini generation, strict grounding prompt & fallback
│   ├── main.py                      # FastAPI server & endpoints
│   │
│   └── static/                      # Web UI for internal support agents
│       ├── index.html               # Chat layout, quick inquiry pills, citations viewer
│       ├── style.css                # BookMyForex navy/emerald branding, card styling
│       └── app.js                   # Client logic, live markdown rendering, settings
│
├── chroma_db/                       # Local persistent Chroma vector store (auto-generated)
│
└── [Knowledge Base Markdown Documents]
    ├── Offers.md                    # Campaign offers, promo codes (BIGFXSALE, REMPITSPL), perks
    ├── gemini-code-1789542020503.md # Company profile, history, leadership, core services
    ├── gemini-code-1789542023580.md # YES Bank Multi-Currency Forex Card guide & procedures
    ├── gemini-code-1789542025658.md # Instarem / Global USD Forex Card guide & procedures
    └── gemini-code-1789542027610.md # Forex card emergency actions, dispute handling & SOP
```

---

## 🚀 Quick Start

### 1. Configure Gemini API Key
Copy `.env.example` to `.env` and add your Google Gemini API key:
```bash
copy .env.example .env
```
Edit `.env`:
```env
GEMINI_API_KEY=your_gemini_api_key_here
```
*(Alternatively, you can launch the app and enter the key directly in the web UI via the ⚙️ Settings dialog).*

### 2. Launch the Application

#### Option A: Windows Batch (Command Prompt)
Double-click `run.bat` or run:
```cmd
run.bat
```

#### Option B: PowerShell
```powershell
.\run.ps1
```

#### Option C: Manual Launch
```bash
# 1. Create and activate virtual environment
python -m venv venv
.\venv\Scripts\activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run FastAPI backend
python -m uvicorn app.main:app --host 127.0.0.1 --port 8000 --reload
```

### 3. Access the Support Agent Portal
Open your browser and navigate to:
**http://127.0.0.1:8000**

---

## 📡 API Endpoints

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/` | Web Chat UI for support agents |
| `POST` | `/api/chat` | Query RAG engine with strict grounding & citations |
| `POST` | `/api/ingest` | Re-read all workspace markdown files and refresh vector index |
| `GET` | `/api/documents` | List all indexed documents, sections, and chunk counts |
| `GET` | `/api/health` | Service health, ChromaDB stats, and API key status |
| `POST` | `/api/settings` | Dynamically update the Gemini API key from the frontend |

### Example Chat Request
```bash
curl -X POST "http://127.0.0.1:8000/api/chat" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is the cashback slab for an order of ₹3,50,000 using BIGFXSALE?"}'
```

---

## 🛡️ Guardrails & Compliance Details

Whenever a support agent queries a topic not documented in the knowledge base, the application strictly responds with:

```
"I cannot find this information in the official BookMyForex guidelines. Please check with the compliance desk."
```

No speculative answers, partial assumptions, or unverified claims are produced.
