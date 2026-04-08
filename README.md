# Knowledge Assistant

**Production-style RAG document Q&A** with honest routing: answers **cite your PDFs, Word docs, and text files** when retrieval and file health are strong enough—and fall back to **general** replies (without fake citations) when the library is empty, sync fails, or evidence is weak.

**Stack:** Python · OpenAI (embeddings + chat) · LangChain · **FAISS** · **BM25 hybrid** · **Streamlit** · **FastAPI** · **Next.js** (TypeScript) · SQLite · Docker

**Docs:** [Architecture & design decisions](docs/ARCHITECTURE.md) · [Demo walkthrough](docs/DEMO.md) · [Screenshots checklist](docs/SCREENSHOTS.md) · [Interview notes](docs/INTERVIEW.md) · [Career assets](docs/CAREER.md) · [Deployment](DEPLOYMENT.md) · [Eval harness](eval/README.md)

---

## Why this project

| Audience | Value |
|----------|--------|
| **Recruiters / hiring managers** | End-to-end **ingest → chunk → embed → retrieve → generate** with explicit **routing**, **trust gates**, and **per-file health**—not a one-off notebook. |
| **Engineers** | One shared **`app/`** domain layer; thin hosts (**Streamlit**, **FastAPI**); **incremental indexing**, **hybrid retrieval**, **SQLite** chat (API path). |
| **Demos** | **Streamlit** for the fastest live demo, or **Next.js + FastAPI** for streaming chat and a premium chat-first UI. |

---

## How it works (high level)

1. **Upload** → files land under `data/raw/` (typed, size limits on API path).
2. **Sync** → parse → **structure-aware** normalize → **chunk** → **embed** (with disk cache) → **FAISS** on disk → **document manifest** + optional **retrieval self-probe**.
3. **Chat** → query rewrite (when needed) → **hybrid retrieve** (vector + BM25, RRF) → rerank → **trust filter** → **grounding gates** → one LLM call (**grounded** with `[SOURCE n]`, or **general** / **web** / **blended**).

Details, gate rationale, and file pointers: **[docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)**.

---

## Key features

| Area | Details |
|------|---------|
| **Retrieval** | FAISS + **BM25** fused with **RRF**; optional **in-process BM25 / FAISS load caches** for repeat queries (invalidated on index rebuild). |
| **Indexing** | **Content-hash** incremental rebuild: unchanged files can reuse chunks when settings match saved state. |
| **Answer modes** | **Grounded**, **general**, **web**, **blended**; routing from library state, retrieval strength, and query shape. |
| **Trust** | Per-file manifest (`ready` / `ready_limited` / `failed` / …); hits filtered; **calibrated gates** for broad vs narrow questions (eval-regression tested). |
| **Tasks** | **Summarize**, **Extract**, **Compare** over retrieved context (single LLM call per task). |
| **UIs** | **Streamlit**: upload, sync, expanders for sources & excerpts, optional debug. **Next.js**: sidebar (chats, library, sync), **Stop / New chat / Clear**, streaming, source snippets. |
| **Ingestion** | PDF (pdfplumber + **pypdf** fallback), DOCX (body + tables in order), TXT; **optional OCR** off by default (`RAG_ENABLE_PDF_OCR`, see `.env.example`). |
| **Quality** | Gold **document QA eval** (`scripts/run_document_qa_eval.py`): routing, anchors, refusals, forbidden tokens, citation surface checks. |

---

## Architecture (diagram)

```text
                    ┌──────────────────┐
                    │   Next.js (web)  │
                    │  SSE + REST      │
                    └────────┬─────────┘
                             │ HTTP
                    ┌────────▼─────────┐
                    │ FastAPI (backend)│
                    │ /api/v1/*        │
                    └────────┬─────────┘
                             │ imports
┌────────────────┐   ┌───────▼────────────────────────────────────────┐
│ streamlit_app  │──►│  app/                                             │
│ (UI only)      │   │  services/   chat, index, upload, doc_task, …   │
└────────────────┘   │  llm/        generator, validation, intent     │
                     │  retrieval/  FAISS, hybrid, context selection     │
                     │  persistence/ manifest, chat_store, library state │
                     │  ingestion/  loader (PDF, DOCX, TXT)            │
                     │  utils/      chunker                            │
                     └──────────────────────────────────────────────────┘
                                         │
                              data/raw/  │  data/indexes/ (FAISS + manifest)
```

**Principle:** One **domain layer** (`app/`). Frontends do not reimplement retrieval or gates.

---

## Screenshots

Add PNGs under **`docs/images/`** and embed them in this README (see **[docs/SCREENSHOTS.md](docs/SCREENSHOTS.md)** for a full checklist).

| Surface | Ideas |
|---------|--------|
| **Streamlit** | Hero, sidebar library + sync, grounded answer + sources/excerpts. |
| **Next.js** | Empty hero, streaming + mode badge, sources with snippets, mobile drawer. |

---

## Local setup

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
```

**Tests:** `pip install -r requirements-dev.txt` (if present) then `pytest`.

### Secrets (safe for GitHub)

1. **Never commit real keys.** `.env`, `.env.local`, and most `*.env` files are **gitignored**.
2. Copy **`.env.local.template`** → **`.env.local`** and paste your key there, **or** export `OPENAI_API_KEY` in the shell.
3. **`.env.example`** and **`.env.local.template`** contain placeholders only and are **not** used as runtime secrets by default.
4. **`OPENAI_API_KEY` resolution (first valid wins):** a **real** key in the **process environment** (CI/Docker) takes precedence; otherwise the loader uses the first **non-placeholder** value from **`.env.local`**, then **`.env`**. Sample/template keys never override a real key in `.env.local`.
5. Verify without printing the key: `set PYTHONPATH=.` then `python scripts/verify_openai_env.py` (optional `--ping-openai`).

### Run Streamlit

```bash
streamlit run streamlit_app.py
```

→ `http://localhost:8501` · Debug: `KA_DEBUG=1`

### Run FastAPI + Next.js

**Terminal 1 — API** (repo root, venv active):

```powershell
$env:PYTHONPATH="."
python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000
```

**Terminal 2 — Web:**

```bash
cd web
cp .env.example .env.local   # or copy on Windows
npm install
npm run dev
```

→ `http://localhost:3000` · Set `NEXT_PUBLIC_API_URL` if the API is not default.

### Docker

```bash
docker compose up --build
```

See **[DEPLOYMENT.md](DEPLOYMENT.md)** for hosted options. UI `:3000`, API `:8000`, `./data` volume.

### Environment variables

Summarized in **`.env.example`** (root) and **`web/.env.example`**. Highlights: `OPENAI_API_KEY`, `KA_CORS_ORIGINS`, `KA_ENV`, `NEXT_PUBLIC_API_URL`, optional `RAG_ENABLE_PDF_OCR`, performance toggles `KA_BM25_CACHE`, `KA_BM25_CACHE_MAX_DOCS` (see code comments in `app/retrieval/hybrid_retrieve.py`).

---

## Release validation (modest smoke)

With a **real** `OPENAI_API_KEY` (typically **`.env.local`**) and network access, these scripts were used for recent release readiness. Your machine must still satisfy dependencies and billing.

| Step | Command | What “good” looks like |
|------|---------|-------------------------|
| Env verify | `python scripts/verify_openai_env.py` | Exit **0**; `ready_for_local_dev: true`; `inferred_value_source` often **`.env.local`** for local dev |
| Real-doc pack | `python scripts/phase28_real_docs_pack.py` | Sync **`ok=True`**, **`vector_count` > 0**, grounded chat probes |
| Gold eval | `python scripts/run_document_qa_eval.py --json-report eval/_report_local.json` | **8/8** cases in the aggregate (exit **0** when all pass) |
| Brutal product | `python scripts/brutal_product_check.py` | Exit **0** after temp sync + two grounded queries |

**No API key:** `pytest` runs without OpenAI and should stay green. Eval JSON and phase workdirs are **gitignored** (see `.gitignore`).

---

## Document QA eval (regression)

With a real key and full `requirements.txt`:

```bash
set PYTHONPATH=.
python scripts/run_document_qa_eval.py --json-report eval/_report_local.json
```

JSON reports under `eval/_report*.json` are **gitignored** by default. See **[eval/README.md](eval/README.md)**.

---

## Observability / debug mode (safe)

This project is designed to be **inspectable** without exposing secrets or confusing normal users.

- **Structured retrieval logs**: set `KA_RETRIEVAL_DEBUG=1` to emit one JSON line per pipeline event (`turn_begin`, `retrieval_hybrid_done`, `routing_decision`, …).
- **Developer diagnostics on answers** (hidden by default):
  - Server: set `KA_DEBUG=1` (FastAPI or Streamlit process).
  - Web UI: set `localStorage.KA_DEBUG=1` and reload to show a “Developer diagnostics” panel under assistant messages.
  - Debug endpoint: `GET /api/v1/debug/last` returns the last-turn diagnostics snapshot (404 when debug is disabled).

Diagnostics include route selected, pool sizes, trust-filter counts, context chunks selected, grounding gate reason, and selected source names.

---

## Repository layout

```text
rag-document-retrieval/
├── streamlit_app.py
├── backend/app/           # FastAPI app
├── web/                   # Next.js (App Router, Tailwind)
├── app/                   # Shared RAG domain
├── eval/                  # Gold cases, harness, scoring
├── docs/
│   ├── ARCHITECTURE.md
│   ├── DEMO_AND_PORTFOLIO.md
│   └── images/            # screenshots (.gitkeep)
├── data/raw/              # uploads (gitignored except samples)
├── data/indexes/          # FAISS + manifest (gitignored)
├── scripts/               # eval, env verify, retrieval smoke
├── Dockerfile
├── docker-compose.yml
├── requirements.txt
├── requirements-optional-ocr.txt
├── .env.example
└── DEPLOYMENT.md
```

---

## Tradeoffs & limitations

| Topic | Reality |
|-------|---------|
| **Retrieval** | Embeddings approximate relevance; gates reduce bad grounding but do not guarantee correctness. |
| **Corpus size** | Answers use **top-k** chunks; very large libraries may need tuning, sharding, or a managed vector DB. |
| **Scale-out** | Default: **single process**, local FAISS + SQLite; multi-instance needs shared storage and an embedding strategy. |
| **Auth** | No built-in auth—treat as **personal / demo** unless you add a gateway. |
| **Cost** | OpenAI usage on sync and chat; monitor keys on shared demos. |

---

## Portfolio quick links

- **Architecture & design Q&A:** [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md)  
- **Demo script:** [docs/DEMO.md](docs/DEMO.md)  
- **Career assets (GitHub blurb, resume bullets, portfolio summary):** [docs/CAREER.md](docs/CAREER.md)

---

## License

MIT License. See [LICENSE](LICENSE).
