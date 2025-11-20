# UL RAG Assistant

A production‑oriented Retrieval‑Augmented Generation (RAG) assistant for the **University of Limerick (UL)**, engineered by **Maryam Najafi**.

The system:

- Crawls or ingests **official UL web content**
- Builds a **hybrid search index** (BM25 + dense embeddings)
- Uses a **LangGraph** pipeline with safety, routing, retrieval and generation
- Exposes clean **chat interfaces** (CLI, Streamlit, FastAPI API)
- Can be integrated with a **Telegram bot** or other frontends

This README explains the architecture, setup, and how to extend the system as a research‑grade RAG project.

---

## 1. High‑Level Overview

### 1.1 Problem

Information about programmes, modules, regulations, research centres (e.g. Lero, SFI Research Centre for Software), accommodation, and campus life at the University of Limerick is spread across many pages and systems. Students and staff often:

- Struggle to find accurate, up‑to‑date answers
- Don’t know which page or system to search
- Want conversational, natural answers rather than raw web pages

### 1.2 Solution

The **UL RAG Assistant** is a **UL‑specialised QA assistant** that:

1. Uses UL websites as its **primary knowledge base**
2. Indexes content with a **hybrid retriever** (BM25 + embeddings)
3. Reranks candidate passages with a **cross‑encoder**
4. Uses a **LangGraph** pipeline to:
   - Ensure basic **safety**
   - **Route** the query (UL question vs greeting vs nonsense)
   - Retrieve relevant UL content
   - Generate **grounded answers** with citations

It is deliberately **modular**, **testable**, and designed as if it were to be deployed inside a real institution or company.

---

## 2. Repository Structure


## 2.1 High-Level Architecture

```text
User (CLI / Streamlit / API / Telegram)
            │
            ▼
      RAGChatSession
            │
            ▼
         run_ul_rag()
            │
            ▼
   +------------------------+
   | LangGraph UL RAG graph |
   +------------------------+
            │
   ┌────────┴────────┐
   ▼                 ▼
Safety           Router (LLM)
(esc / crisis)       │
   │                 ▼
   │           QueryPlan
   │                 │
   │         ┌───────┴─────────────────────────┐
   │         │                                 │
   ▼         ▼                                 ▼
[Escalate]  chitchat / nonsense           UL question
 (no RAG)     (no retrieval)             (RAG enabled)
                 │                             │
                 ▼                             ▼
           LLM small talk                Retriever (BM25 + dense)
           (no context)                       │
                                              ▼
                                       Reranker (CrossEncoder)
                                              │
                                              ▼
                                        Generator (LLM)
                                              │
                                              ▼
                                         Final answer
```
### 2.2 End-to-end flow
```text
ul-rag-assistant/
├─ README.md
├─ requirements.txt
├─ .env.example
├─ data/
│  └─ ul/
│     └─ ul_seeds.jsonl          # initial UL seed URLs for crawling
├─ storage/
│  └─ index/
│     └─ ul_index.pkl            # built index (BM25 + dense embeddings)
├─ scripts/
│  ├─ ingest_ul_web.py           # crawl UL seeds -> JSONL corpus
│  ├─ build_index.py             # JSONL corpus -> index pickle
│  ├─ chat_cli.py                # terminal chat interface
│  ├─ run_streamlit.py           # Streamlit chat UI
│  └─ run_api.py                 # FastAPI/Uvicorn runner
├─ api/
│  └─ main.py                    # FastAPI app (HTTP /chat endpoint)
├─ src/
│  └─ ul_rag/
│     ├─ __init__.py
│     ├─ config.py               # environment & settings loader
│     ├─ logging.py              # get_logger()
│     ├─ ingest/
│     │  ├─ __init__.py
│     │  ├─ web.py               # fetch UL pages into JSONL
│     │  └─ build_index.py       # build BM25 + embedding index
│     ├─ retrieval/
│     │  ├─ __init__.py
│     │  ├─ retriever.py         # hybrid retriever + RRF + reranker
│     │  └─ rerank.py            # CrossEncoder-based reranker
│     ├─ graph/
│     │  ├─ __init__.py
│     │  ├─ safety.py            # crisis / escalation checks
│     │  ├─ router.py            # query router -> QueryPlan
│     │  └─ graph.py             # LangGraph pipeline + run_ul_rag()
│     ├─ llm/
│     │  ├─ __init__.py
│     │  ├─ prompts.py           # system + user prompts
│     │  └─ generate.py          # Generator (RAG / chitchat / nonsense)
│     └─ interfaces/
│        ├─ __init__.py
│        └─ chat_session.py      # RAGChatSession + ChatTurn
└─ tests/
   ├─ test_router.py
   └─ test_graph.py
```

---

## 3. Installation & Configuration

### 3.1 Clone and create a virtual environment

```bash
git clone https://github.com/<your-username>/ul-rag-assistant.git
cd ul-rag-assistant

python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

### 3.2 Environment variables

Copy the example file and update it:

```bash
cp .env.example .env
```

`.env` contains:

```dotenv
OPENAI_API_KEY=sk-...

GEN_MODEL=gpt-4o-mini
EMBED_MODEL=sentence-transformers/all-MiniLM-L6-v2
RERANK_MODEL=cross-encoder/ms-marco-MiniLM-L-6-v2

INDEX_PATH=storage/index/ul_index.pkl
LOG_LEVEL=INFO
```

- `OPENAI_API_KEY` – OpenAI API key (required for routing + generation).
- `GEN_MODEL` – LLM used for router + RAG generation.
- `EMBED_MODEL` – sentence‑transformers model for dense retrieval.
- `RERANK_MODEL` – cross‑encoder reranker model.
- `INDEX_PATH` – location of the UL index pickle.
- `LOG_LEVEL` – `DEBUG`, `INFO`, `WARNING`, etc.

---

## 4. Data & Ingestion Pipeline

The knowledge base is built from UL web pages using a **two‑step pipeline**:

1. Crawl UL pages from seed URLs → **JSONL corpus**
2. Build hybrid index (BM25 + embeddings) → **`ul_index.pkl`**

### 4.1 Seed URLs

`data/ul/ul_seeds.jsonl` defines starting points for crawling:

```jsonl
{"url": "https://www.ul.ie"}
{"url": "https://www.ul.ie/courses"}
{"url": "https://www.ul.ie/study-at-ul"}
```

You can extend this with high‑value UL domains, such as:

- CSIS department pages
- Lero & SFI Research Centre for Software
- Campus accommodation and campus life
- Admissions, academic registry, exam timetables
- Staff and researcher profiles (e.g. pure.ul.ie)

### 4.2 Step 1 – Crawl UL pages → JSONL

```bash
python scripts/ingest_ul_web.py   --seeds data/ul/ul_seeds.jsonl   --out_jsonl data/ul/ul_docs.jsonl
```

`ingest_ul_web.py`:

- Reads each `{"url": ...}` line from `ul_seeds.jsonl`
- Fetches the page using `httpx` (with redirects)
- Cleans HTML with BeautifulSoup:
  - Drops `<script>`, `<style>`, `<header>`, `<footer>`, `<nav>` etc.
  - Extracts main visible text
- Emits one JSON‐object per page to `ul_docs.jsonl`:

```json
{
  "url": "https://www.ul.ie/some-page",
  "title": "Page Title",
  "text": "Cleaned plain-text content..."
}
```

This JSONL forms the **raw corpus** for indexing.

### 4.3 Step 2 – Build index (BM25 + embeddings)

```bash
python scripts/build_index.py   --input data/ul/ul_docs.jsonl   --index_path storage/index/ul_index.pkl
```

`build_index.py`:

1. Reads each line from `ul_docs.jsonl`
2. Applies a simple whitespace‑based chunker (e.g. ~200 tokens per chunk)
3. Computes dense embeddings with `SentenceTransformer(EMBED_MODEL)`
4. Builds a BM25 index (`BM25Okapi`) over tokenised chunks
5. Saves a pickle to `INDEX_PATH` with:

```python
{
  "texts": [...],         # chunk strings
  "metas": [...],         # {"source_url": ..., "title": ...}
  "embeddings": ndarray,  # dense embedding matrix
  "bm25": BM25Okapi,      # sparse index
  "embed_model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

> **If you see** `_pickle.UnpicklingError: pickle data was truncated`  
> The index file is corrupted (build interrupted).  
> Delete and rebuild:

```bash
rm storage/index/ul_index.pkl
python scripts/build_index.py --input data/ul/ul_docs.jsonl
```

---

## 5. Retrieval & Reranking

All retrieval logic is in `src/ul_rag/retrieval/`.

### 5.1 Hybrid Retriever

`ul_rag.retrieval.retriever.Retriever`:

- Loads:
  - `texts` (UL chunks)
  - `metas` (metadata such as `source_url`, `title`)
  - `embeddings` (dense embedding matrix)
  - `bm25` (BM25 index object)
- Implements:
  - `_dense_search(query, top_k)`:
    - Encode query via `SentenceTransformer(EMBED_MODEL)`
    - Compute cosine similarity vs each document embedding
  - `_sparse_search(query, top_k)`:
    - Compute BM25 relevance scores for query tokens

These two rankings are fused with **Reciprocal Rank Fusion (RRF)**:

- For rank `r` in dense:
  - `score_dense += 1 / (60 + r)`
- For rank `r` in sparse:
  - `score_sparse += 1 / (60 + r)`
- Combined score = `score_dense + score_sparse`

This yields a list of candidate passages.

### 5.2 Cross‑Encoder Reranker

`ul_rag.retrieval.rerank.Reranker`:

- Loads `CrossEncoder(RERANK_MODEL)` (e.g. `cross-encoder/ms-marco-MiniLM-L-6-v2`)
- Re‑scores `(query, passage)` pairs
- Sorts by cross‑encoder score
- Returns top `max_chunks` passages, each as:

```python
{
  "text": "chunk text...",
  "meta": {"source_url": "https://...", "title": "..."},
  "score": 0.93,
  "rank": 1
}
```

These passages are used as **context** for the generator (LLM).

---

## 6. LangGraph RAG Pipeline

The core RAG logic is implemented as a **LangGraph `StateGraph`** in `src/ul_rag/graph/graph.py`.

### 6.1 State Definition

```python
class ULState(TypedDict):
    question: str
    mode: Literal["student", "staff"]
    locale: str
    plan: Optional[Dict[str, Any]]
    docs: List[ULDoc]
    answer: Optional[str]
    citations: List[Dict[str, Any]]
    meta: Dict[str, Any]
```

Where `ULDoc` is:

```python
class ULDoc(TypedDict):
    text: str
    meta: Dict[str, Any]
```

### 6.2 Pipeline Flow

```text
safety -> route -> retrieve -> generate -> END
```

Nodes:

1. `safety` – basic safety / escalation
2. `route` – LLM router → `QueryPlan`
3. `retrieve` – hybrid retrieval + rerank
4. `generate` – LLM generation with/without context

### 6.3 Safety Node

`ul_rag.graph.safety.Safety`:

- Minimal, keyword‑based detection of crisis/self‑harm language.
- If triggered:
  - Sets a non‑harmful, escalation‑style message in `answer`
  - Attaches `meta["escalation"] = "crisis"`
  - Skips routing, retrieval, and generation

This is a placeholder for a more sophisticated institutional safety layer.

### 6.4 Router Node (QueryPlan)

`ul_rag.graph.router.Router` calls OpenAI to infer a structured `QueryPlan`:

```python
@dataclass
class QueryPlan:
    query_type: Literal[
        "who_is",
        "programme_or_module",
        "campus_directions",
        "admin_process",
        "research",
        "general",
        "chitchat",
        "nonsense",
    ]
    topic: str
    needs_multi_hop: bool
    retrieval_mode: Literal["hybrid", "dense_only", "sparse_only"]
    max_chunks: int
    domain_hint: Optional[str]
```

Key categories:

- **UL‑information queries**:
  - `who_is`, `programme_or_module`, `campus_directions`,
    `admin_process`, `research`, `general`
- **Non‑informational**:
  - `chitchat` – greeting / small talk (e.g. “hi”, “hello”, “thanks”)
  - `nonsense` – random characters / clearly not meaningful

The router’s LLM prompt:

- Explains all query types
- Instructs the model to output **pure JSON** only
- Encourages UL‑specific topic hints (e.g. `topic="lero"`)

### 6.5 Retrieve Node

`retrieve_node` reads the `plan`:

- If `query_type ∈ {"chitchat", "nonsense"}`:
  - Skips retrieval → `docs = []` (no context needed)
- Otherwise:
  - Calls `retriever.retrieve(question, max_chunks=plan.max_chunks)`
  - Stores results in `state["docs"]`

### 6.6 Generate Node

`generate_node` decides how to produce the final answer:

1. If `answer` already set by safety, return as‑is.

2. **Chitchat** (`query_type == "chitchat"`):
   - Calls `Generator.answer_chitchat(question, mode, locale)`
   - Returns a short, friendly response
   - **No citations, no context**

3. **Nonsense** (`query_type == "nonsense"`):
   - Calls `Generator.answer_nonsense(question, mode, locale)`
   - Explains it didn’t understand the message; invites a UL question
   - **No citations, no context**

4. **UL questions** (all other query types):
   - If `docs` is empty:
     - Returns a polite fallback: “I couldn’t find UL documents related to that question”
   - Else:
     - Calls `Generator.answer(question, docs, mode, locale)`
     - Uses retrieved UL snippets as context
     - Returns answer + citations

The top‑level function is:

```python
from ul_rag.graph.graph import run_ul_rag

resp = run_ul_rag("When do spring exams start?")
print(resp["answer"])
print(resp["citations"])
print(resp["plan"])
```

---

## 7. LLM Layer & Prompts

All LLM logic is under `src/ul_rag/llm/`.

### 7.1 System Prompts

`prompts.py` defines:

- `STUDENT_SYSTEM` – for student‑facing responses
  - Friendly, helpful, context‑grounded
  - Explicit guidelines:
    - Use UL context for real questions
    - Don’t hallucinate dates/emails if missing
    - Handle greetings/nonsense by **ignoring** context

- `STAFF_SYSTEM` – for staff‑oriented mode
  - More concise, professional, policy‑aware

- `USER_TEMPLATE` – shapes how question + context are passed to the LLM:
  - Includes the raw user message
  - Includes a formatted block of UL snippets
  - Instructs the model when to treat a message as greeting / nonsense vs real UL query

### 7.2 Generator

`generate.py` provides `Generator` with three key behaviours:

1. `answer(question, ctx, mode, locale)` – main RAG path
   - Formats retrieved documents:
     - Strips YAML frontmatter if present
     - Truncates to a reasonable length
     - Appends `(Source: <url>)` metadata
   - Calls OpenAI ChatCompletion with system + user prompts
   - Returns:
     - `"answer"` – natural language answer
     - `"citations"` – `{ "n": i, "source": url }`
     - `"meta"` – includes `model` used

2. `answer_chitchat(question, mode, locale)` – small talk
   - Uses a specialised system prompt:
     - 1–2 sentences
     - Friendly
     - **No UL details unless explicitly asked**
     - No citations / “Next steps”

3. `answer_nonsense(question, mode, locale)` – gibberish
   - Short explanation that message was unclear
   - Invites user to ask a UL question
   - No citations / context

If `OPENAI_API_KEY` is missing or fails:

- The generator logs a warning
- Falls back to a deterministic summarisation of retrieved context
- Still returns something informative (but clearly labelled as no‑LLM mode)

---

## 8. Interfaces & Usage

### 8.1 Library‑level: `RAGChatSession`

`src/ul_rag/interfaces/chat_session.py`:

```python
from ul_rag.interfaces.chat_session import RAGChatSession

session = RAGChatSession(mode="student", locale="IE")

turn = session.ask("Hi")
print(turn.content)

turn = session.ask("When do spring exams start?")
print(turn.content)
print(turn.citations)
```

- Maintains full chat history as `ChatTurn` objects.
- For each user message:
  - Calls `run_ul_rag(question, mode, locale)` internally.
  - Appends assistant turn to history.

### 8.2 CLI Chat

```bash
python scripts/chat_cli.py
```

- Starts a simple terminal REPL:

```text
UL RAG Assistant (mode=student, locale=IE)
Type your question (or 'quit' to exit).

You: hi
Bot: Hi! I'm the University of Limerick assistant. Ask me anything about UL whenever you're ready.
```

### 8.3 Streamlit Web UI

```bash
streamlit run scripts/run_streamlit.py
```

- Opens a web chat at `http://localhost:8501` (default)
- Uses `st.chat_message("user" / "assistant")` to render messages
- Session state holds a single `RAGChatSession` instance

### 8.4 FastAPI HTTP API

```bash
python scripts/run_api.py
# internally: uvicorn api.main:app --host 0.0.0.0 --port 8300
```

`api/main.py` exposes:

- `POST /chat`

Request body:

```json
{
  "question": "When do spring exams start?",
  "mode": "student",
  "locale": "IE"
}
```

Response:

```json
{
  "answer": "…",
  "citations": [
    {"n": 1, "source": "https://www.ul.ie/..."}
  ],
  "meta": {...}
}
```

This is ideal for:

- Telegram bot backends
- Web frontends
- Integrations with other systems

---

## 9. Development & Testing

### 9.1 Running Tests

Tests are under `tests/`.

```bash
pytest
```

Included tests:

- `test_router.py` – ensures the router returns a valid `QueryPlan` even without an API key (falls back to default plan).
- `test_graph.py` – verifies `run_ul_rag()` returns a dict with keys:
  - `answer`
  - `citations`
  - `meta`
  - `plan`

You can extend this test suite to cover:

- Router behaviour:
  - `"hi"` → `query_type == "chitchat"`
  - gibberish (`"dfghjkl;"`) → `query_type == "nonsense"`
  - `"Who is J.J. Collins?"` → `query_type == "who_is"`
- Retrieval quality:
  - For known UL queries, ensure non‑empty contexts.
- Generator fallback behaviour when `OPENAI_API_KEY` is absent.

### 9.2 Running in Dev

Useful commands:

```bash
# CLI
python scripts/chat_cli.py

# Streamlit
streamlit run scripts/run_streamlit.py

# API
python scripts/run_api.py
```

For debugging, set `LOG_LEVEL=DEBUG` in `.env`.

---

## 10. Extending the System

This project is intentionally modular and research‑friendly. Some natural extensions:

1. **Smarter chunking**
   - Replace `_simple_chunk` with:
     - Token‑aware splitting (e.g. sentence/paragraph level)
     - Overlapping windows for better context

2. **Richer routing**
   - Add more specific query types:
     - `accommodation`, `fees`, `modules_csis`, `jobs`, etc.
   - Adjust retrieval filters and prompts per type.

3. **Verification / Self‑checking**
   - Add a `verify` node that:
     - Takes the draft answer + context
     - Checks if claims are supported by retrieved passages
     - Refuses or revises ungrounded answers

4. **Conversation memory**
   - Add a summarisation node:
     - Summarise long histories into a compact “session memory”
     - Store memory in a separate vector store
     - Reuse memory for follow‑up questions

5. **RAG evaluation**
   - Build a UL‑specific QA dataset:
     - Questions + reference answers + gold documents
   - Evaluate:
     - Retrieval quality (Recall@k, MRR)
     - Answer quality (RAGAS or custom metrics)

6. **Deployment**
   - Dockerise the app
   - Add monitoring/logging for:
     - Most frequent queries
     - Retrieval hits/misses
     - Hallucination / low grounding events (via verifier)

---

## 11. Credits

- **Author**: **Maryam Najafi**  
  Lead architecture, implementation, and RAG design for the University of Limerick assistant.

- **Core stack**:
  - Python
  - `sentence-transformers`
  - `rank-bm25`
  - `langgraph`
  - `openai`
  - `fastapi`, `uvicorn`
  - `streamlit`

If you use this project, extend it, or adapt it for another institution, please consider linking back to the repository and crediting the original author.

---
