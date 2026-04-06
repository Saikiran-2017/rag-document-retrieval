# Architecture (interview-ready)

This document is a concise map of **Knowledge Assistant**: how data flows, where decisions happen, and why key technologies were chosen.

## System context

- **Users** upload documents, sync an index, and chat.
- **Surfaces**: **Streamlit** (single-file demo UI) and **FastAPI + Next.js** (API + streaming web client). Both call the same **domain layer** under `app/`.
- **Storage**: raw files under `data/raw/`, FAISS index + manifest under `data/indexes/`, optional SQLite chat history (API path).

## End-to-end flow

```text
Upload → raw library (data/raw/)
    ↓
Sync → ingest (PDF / DOCX / TXT) → normalize → chunk → embed (cached) → FAISS save
    ↓
Manifest + probe → per-file health (ready / ready_limited / failed / …)
    ↓
Chat → (optional rewrite) → hybrid retrieve (vector + BM25, RRF) → rerank → trust filter
    ↓
Grounding gates → context assembly → LLM (grounded / general / web / blended)
    ↓
Validation + UI (sources, mode badge, excerpts)
```

### Ingestion

- **Loader** (`app/ingestion/loader.py`): pdfplumber for PDF pages, python-docx for body + tables in order, UTF-8 TXT. Structure-preserving normalization keeps paragraphs and line breaks where possible. **pypdf** is a second-chance text path for empty plumber pages; **OCR** is opt-in via `RAG_ENABLE_PDF_OCR` and optional deps.

### Chunking

- **Chunker** (`app/utils/chunker.py`): LangChain `RecursiveCharacterTextSplitter` with separators that favor paragraphs and markdown headings before hard cuts.

### Indexing

- **Embeddings**: OpenAI `text-embedding-3-small` (configurable pattern) with **disk embedding cache** to avoid re-embedding unchanged text on rebuilds.
- **FAISS**: Vectors + docstore persisted as `{index}.faiss` + `{index}.pkl`. **Incremental rebuild**: unchanged files (content hash) can reuse existing chunk documents when settings match saved library state.

### Retrieval

- **Dense**: FAISS similarity search (L2 distance in current setup).
- **Sparse**: BM25 over the same corpus (rank-bm25), fused with dense via **RRF** (reciprocal rank fusion).
- **Performance**: In-process **BM25 cache** and **FAISS load cache** (mtime-invalidated) reduce repeated work per query without changing ranked results for a fixed index.

### Grounding and trust

- **Hybrid usefulness gate**: Top hit must meet combined vector + RRF criteria (with relaxed thresholds for broad doc-QA when classified by query intent).
- **Manifest**: Files can be `ready`, `ready_limited` (index/probe mismatch or weak probe), etc. Stricter **limited-corpora** rules apply when grounding from partially trusted files; coherence fallbacks avoid false refusals when multiple hits agree on one document.
- **Outcome**: If gates fail, the app answers **general** or abstains with a fixed unknown phrase—**no invented `[SOURCE n]`** for that turn.

### Generation

- **Grounded** prompts use numbered `[SOURCE n]` blocks; the model is instructed to cite only those numbers and to use the unknown phrase when evidence is insufficient.
- **Post-checks**: Citation overlap / validation warnings can surface in the UI without changing core retrieval.

## Design decisions (short answers for interviews)

| Choice | Why |
|--------|-----|
| **FAISS** | Simple, fast local similarity search; no hosted vector DB required for demos and portfolio; easy to ship in Docker with a volume. |
| **Hybrid retrieval** | Pure embedding search misses exact tokens (SKUs, codes, rare strings); BM25 + RRF improves recall and gives explainable fusion scores in debug logs. |
| **Grounding gates** | Prevents “confident wrong” answers from weak retrieval; encodes product trust: cite only when evidence is strong enough and files are healthy enough. |
| **Optional OCR** | Scanned PDFs break text extractors; OCR is expensive and environment-specific, so it stays **off by default** and isolated behind env + optional requirements. |
| **Caching** | Embedding cache cuts re-index cost; BM25 / FAISS load caches cut **per-query** latency for repeat traffic on a stable index—correctness preserved via invalidation (rebuild changes files on disk). |

## Where to read code

| Concern | Primary locations |
|---------|-------------------|
| Chat routing | `app/services/chat_service.py` |
| Index build / sync | `app/services/index_service.py` |
| Hybrid search | `app/retrieval/hybrid_retrieve.py` |
| Gates + thresholds | `app/services/document_health.py`, `app/llm/generator.py` |
| Prompts + citations | `app/llm/generator.py`, `app/llm/answer_validation.py` |
| Eval harness | `eval/harness.py`, `eval/scoring.py`, `scripts/run_document_qa_eval.py` |
