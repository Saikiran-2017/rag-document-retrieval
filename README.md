# Knowledge Assistant

**RAG-powered document Q&A** with honest routing: answers **cite your PDFs, Word docs, and text files** when retrieval is strong - and fall back to **general** replies (without fake citations) when the library is empty, sync fails, or the question is not document-shaped.

**Stack:** Python · OpenAI (embeddings + chat) · LangChain · **FAISS** · **Streamlit** · **FastAPI** · **Next.js** (TypeScript) · SQLite · Docker optional

---

## At a glance

| For… | Why this project |
|------|------------------|
| **Recruiters / hiring managers** | End-to-end **ingest → chunk → embed → retrieve → generate** with clear failure handling and a **production-style** API + web UI - not only a notebook. |
| **Engineers** | Shared **`app/`** domain layer; thin UIs (Streamlit, HTTP); explicit **routing**, **trust gates**, and **per-file health** in the index manifest. |
| **Demos** | Two runnable surfaces: **Streamlit** (fastest live link) or **Next.js + FastAPI** (streaming chat, document tools, SQLite history). |

---

## Feature summary

| Area | Details |
|------|---------|
| **Retrieval** | OpenAI embeddings, **FAISS** vector store on disk, hybrid-style retrieval hooks; **incremental re-index** when file hashes + chunk settings match saved state (reuse unchanged files). |
| **Answer modes** | **Grounded** (documents), **web**, **blended**, **general**; routing chooses based on library state, retrieval strength, and query shape; **no forced citations**. |
| **Trust & health** | Per-file manifest states (`uploaded` → `processing` → `ready` / `ready_limited` / `failed`); retrieval hits filtered for unsafe sources; grounded allowance respects library health. |
| **Task modes** | **Summarize**, **Extract**, **Compare** over retrieved context - **single LLM call** each (no agent loop). |
| **Streamlit UI** | Upload, sync, preferences, chat history, sources & excerpts expanders, optional debug panel (`KA_DEBUG=1`). |
| **Full stack** | **FastAPI** REST + **SSE streaming** chat; **Next.js** chat UI with sidebar (chats, upload, sync, file health); **SQLite** chat persistence. |
| **Ops** | `.env.example`, **Dockerfile** + **docker-compose** (API + web + `./data` volume), **[DEPLOYMENT.md](DEPLOYMENT.md)** for hosted options. |

---

## Architecture

### Full-stack view

```text
                    ┌──────────────────┐
                    │   Next.js (web)  │
                    │  SSE + REST API  │
                    └────────┬─────────┘
                             │ HTTP
                    ┌────────▼─────────┐
                    │ FastAPI (backend)│
                    │ /api/v1/*        │
                    └────────┬─────────┘
                             │ imports
┌────────────────┐   ┌───────▼────────────────────────────────────────┐
│ streamlit_app  │──►│  app/                                             │
│ (UI only)      │   │  services/  chat, index, upload, doc_task, …    │
└────────────────┘   │  llm/       generator (general, grounded, tasks)  │
                     │  retrieval/ FAISS load, retrieve, hybrid helpers │
                     │  persistence/ chat_store, document_manifest, …     │
                     │  ingestion/ chunker                              │
                     └──────────────────────────────────────────────────┘
                                         │
                              data/raw/   │   data/indexes/ (FAISS)
                              uploads     │   + manifest + library state
```

**Principle:** One **domain layer** (`app/`). **Streamlit** and **FastAPI** are hosts; **Next.js** talks only to HTTP. Retrieval, routing, validation, and trust logic stay in **`app/services`** and **`app/llm`** - not duplicated in the frontends.

### Request path (simplified)

1. **Upload** → safe save under `data/raw/` (typed, size limits, duplicate detection on API path).
2. **Sync** → fingerprint library → chunk changed files → **embed** (with disk cache) → rebuild or merge FAISS → **document manifest** updated from parse / index / probe.
3. **Chat** → optional web search branch → retrieve top-k → **usefulness + trust** gates → **one** completion (grounded with `[SOURCE n]` blocks, or general).

---

## Screenshots

Add PNGs under **`docs/images/`** and drop them into the table (or embed in this README).

### Streamlit

| # | File | Capture |
|---|------|---------|
| 1 | `01-streamlit-hero.png` | Empty state: hero, value prop, starter prompts. |
| 2 | `02-streamlit-general.png` | Short non-doc question → general answer, **no** sources. |
| 3 | `03-streamlit-library.png` | Sidebar: upload, file list, Sync, task mode. |
| 4 | `04-streamlit-grounded.png` | Grounded reply + **Sources** / **Supporting excerpts** expanded. |
| 5 | `05-streamlit-task.png` | Summarize or Compare with evidence. |

### Next.js + API

| # | File | Capture |
|---|------|---------|
| 6 | `06-web-empty.png` | Empty hero + suggested prompts. |
| 7 | `07-web-streaming.png` | Assistant streaming + mode badge. |
| 8 | `08-web-sources.png` | **Sources referenced** + optional web sources cards. |
| 9 | `09-web-mobile.png` | Narrow width: drawer menu + composer (optional). |

**Example embed (after you add files):**

```markdown
![Next.js chat](docs/images/07-web-streaming.png)
```

---

## Setup guide

**Prerequisites:** Python **3.11+**, Node **20+** (for `web/`), [OpenAI API key](https://platform.openai.com/).

```bash
git clone <your-repo-url>
cd rag-document-retrieval
python -m venv .venv
```

**Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`  
**macOS / Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
cp .env.example .env   # PowerShell: Copy-Item .env.example .env
# Set OPENAI_API_KEY in .env
```

**Optional tests:** `pip install -r requirements-dev.txt` then `pytest`.

### Run Streamlit

```bash
streamlit run streamlit_app.py
```

→ `http://localhost:8501` · Debug: `KA_DEBUG=1`

### Run FastAPI + Next.js

**Terminal 1 - API** (repo root, venv active):

```bash
# CMD
set PYTHONPATH=.
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

```powershell
# PowerShell
$env:PYTHONPATH="."
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 - Web:**

```bash
cd web
cp .env.example .env.local
npm install
npm run dev
```

→ `http://localhost:3000` · Set `NEXT_PUBLIC_API_URL` to match the API · CORS: `KA_CORS_ORIGINS` (see `.env.example`).

### Docker (all-in-one)

```bash
cp .env.example .env   # add OPENAI_API_KEY
docker compose up --build
```

UI `:3000`, API `:8000`, data persisted in `./data`. See **[DEPLOYMENT.md](DEPLOYMENT.md)** for cloud options.

### Environment variables

Summarized in **`.env.example`** (root) and **`web/.env.example`**. Highlights: `OPENAI_API_KEY`, `KA_CORS_ORIGINS`, `KA_ENV` (hide OpenAPI in prod), `NEXT_PUBLIC_API_URL`.

---

## Usage (quick)

1. **Upload** supported files → **Sync** to build/update the index.
2. Ask in **Auto** for normal Q&A; switch to **Summarize / Extract / Compare** for focused tasks.
3. Open **Sources** (Streamlit expanders or Next cards) when the answer is document-backed.
4. **New chat** clears thread history; the **library on disk** remains until you change files.

---

## Tradeoffs & limitations

| Topic | Reality |
|-------|---------|
| **Retrieval** | Embedding similarity is a **proxy** for relevance; distance gates reduce bogus grounding but do not guarantee correctness. |
| **Corpus size** | Answers use **top-k chunks**, not full multi-hundred-page reads in one shot, by design for latency and cost. |
| **Scale-out** | Default deployment assumes **one process** with **local FAISS + SQLite**; horizontal scale needs shared storage and a deliberate embedding/index strategy. |
| **Auth** | No built-in auth; treat as **personal / demo** unless you add a gateway. |
| **Cost** | Every sync and query uses **OpenAI** tokens; monitor usage on shared demos. |
| **Verification** | Users should **verify** high-stakes answers against the cited source text. |

---

## Future roadmap

- **Eval harness**: fixed Q/A set for retrieval hit rate and answer groundedness (great for interviews).
- **Managed vector DB**: optional swap from file-backed FAISS for multi-user demos.
- **Export**: chat or citations to Markdown/PDF.
- **Auth + tenancy**: API keys or OAuth for public deployments.
- **Observability**: structured logs / tracing around retrieve + generate.

---

## Appendix: portfolio & interview pack

*Copy the sections below into your resume site, LinkedIn, or interview prep doc.*

### Resume bullets (2)

- **Built a production-style RAG workspace** (Python, FAISS, OpenAI, LangChain) with **incremental indexing**, **per-document health**, and **trust-aware routing** so answers cite uploads only when retrieval is reliable, plus **FastAPI** + **Next.js** with **SSE streaming** and SQLite chat history.

- **Designed a shared domain layer** (`app/services`, `app/llm`) consumed by **Streamlit** and **HTTP APIs**, enabling **document + web + blended** answer modes and **summarize/extract/compare** tasks without duplicating retrieval or validation logic.

### Portfolio blurb (short)

> **Knowledge Assistant**. A document-grounded chat app that ingests PDFs and Office files into a **FAISS** index, retrieves with OpenAI embeddings, and answers with **honest routing**: grounded replies with **sources** when evidence is strong, general fallback when it is not. Includes **Streamlit** and a **FastAPI + Next.js** stack with **streaming** and Docker-ready deployment, built to demo **end-to-end ML engineering** and **API design** in interviews.

### Demo script (2–4 minutes)

1. **Hook (20s)**: “Small RAG workspace: chat plus a document library. It cites files when retrieval supports it, and it won’t invent sources when it doesn’t.”
2. **General path (30s)**: No uploads (or unrelated question): show a **clean general answer** and **no** source UI.
3. **Grounded path (60–90s)**: Upload a short PDF/TXT → **Sync** → ask something **only in the file** → show **Sources** (and excerpts in Streamlit or cards in Next).
4. **Trust / resilience (30s)**: Mention **manifest health**, fallback when sync or retrieval is weak, optional **`KA_DEBUG`** to show routing.
5. **Stretch (30s)**: **Summarize** or **Compare** task mode, or **Next.js streaming** line appearing token-by-token.
6. **Close (15s)**: “Single domain layer in `app/`, FAISS on disk, thin UIs. Easy to walk through **ingest → embed → retrieve → generate** in a system design round.”

### Interview explanation (60–90 seconds)

> “I built Knowledge Assistant to practice the full RAG loop in a maintainable way. Uploads land in a raw folder; sync chunks and embeds them into a **FAISS** index with **content-hash-based incremental rebuilds** so unchanged files aren’t re-embedded every time.  
> On a question, the **chat service** decides whether to retrieve, whether the hits are strong enough to **ground** the answer, and whether to blend in **web** results. There’s a **document manifest** for per-file health so we don’t treat broken or partial indexes as fully trustworthy.  
> The same logic powers **Streamlit** for quick demos and a **FastAPI** backend with **SSE streaming** for a **Next.js** UI, with chat history in **SQLite**. I kept retrieval and prompts in one place so the architecture stays easy to explain and extend.”

---

## Routing behavior (Auto mode): detail

- No files / failed sync → **general** answer (no citations).
- Short, non-document-ish queries → optional **fast path** without retrieval.
- Weak top hit (distance / usefulness) → **general** (no fake sources).
- Strong retrieval → **grounded** generation; LLM failure → safe general fallback + note.

Task modes add gates (e.g. **Compare** needs ≥2 files).

---

## Repository layout

```text
rag-document-retrieval/
├── streamlit_app.py
├── backend/app/           # FastAPI routes, schemas
├── web/                   # Next.js (App Router, Tailwind)
├── app/                   # Shared RAG domain
│   ├── services/
│   ├── llm/
│   ├── retrieval/
│   ├── persistence/
│   └── ingestion/
├── data/raw/              # uploads (gitignored except samples)
├── data/indexes/          # FAISS + manifest (gitignored)
├── docs/images/           # screenshots for README / portfolio
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── DEPLOYMENT.md
```

---

## License

MIT License. See [LICENSE](LICENSE).
