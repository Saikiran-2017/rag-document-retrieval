# Demo, screenshots, and portfolio copy

Use this page when preparing a **live demo**, a **GitHub About** blurb, or a **resume / portfolio** entry.

## Recruiter-friendly one-liner (GitHub tagline)

**Document-grounded chat with honest routing:** ingest PDFs and Office files into a local FAISS index, retrieve with hybrid search, and answer with citations only when retrieval and file health say it is safe—plus Streamlit and FastAPI + Next.js UIs.

## Short paragraph (portfolio / LinkedIn)

Knowledge Assistant is an end-to-end RAG workspace: uploads land in a raw library, sync builds or incrementally updates a **FAISS** vector index with **OpenAI embeddings**, and chat runs **hybrid retrieval** (dense + BM25, RRF fusion) through explicit **grounding and trust gates**. The same Python domain layer powers **Streamlit** for quick demos and **FastAPI + Next.js** with **SSE streaming** and SQLite chat history for a product-style experience. A small **gold eval harness** checks routing, retrieval anchors, refusals, and citation form.

## Screenshot checklist (add files under `docs/images/`)

| # | Suggested filename | What to capture |
|---|-------------------|-----------------|
| 1 | `01-hero-empty.png` | Empty state: value prop + starter prompts (web or Streamlit). |
| 2 | `02-library-sidebar.png` | Document list with health (Ready / Limited) + Upload + Sync. |
| 3 | `03-grounded-answer.png` | Answer with `[SOURCE n]` in body + sources panel with snippets. |
| 4 | `04-general-no-sources.png` | Non-document or empty library: general mode, no fake sources. |
| 5 | `05-streaming.png` | Token streaming + Stop (web). |
| 6 | `06-negative-refusal.png` | Question not in docs: refusal without forbidden content (optional). |
| 7 | `07-mobile-drawer.png` | Narrow viewport: sidebar / composer (optional). |

Embed in the root **README** once files exist.

## Demo flow (3–4 minutes)

1. **Positioning (30 s)**  
   “RAG app with two UIs, one domain layer. It cites files only when retrieval and health checks pass.”

2. **Upload + sync (45 s)**  
   Drop a short PDF or TXT → Sync → point at manifest / “Ready” state.

3. **Grounded Q&A (60 s)**  
   Ask something only in the document → show answer + sources (and excerpts in Streamlit).

4. **Honest fallback (30 s)**  
   Ask off-topic or empty library → general answer, no sources.

5. **Depth option (45 s)**  
   Mention hybrid retrieval + gates, or run `scripts/run_document_qa_eval.py` / `pytest` if the audience is technical.

6. **Close (15 s)**  
   “FAISS on disk, incremental index, eval harness for regression—easy to extend to a managed vector DB later.”

## Talking points (system design)

- **Single domain layer** (`app/`) shared by Streamlit and HTTP—no duplicated retrieval logic in the frontends.
- **Incremental indexing** via content hashes; unchanged files skip re-embedding when settings match.
- **Safety vs UX**: strict gates for narrow facts; calibrated relaxation for broad summaries and vague performance questions, with negative-case tests.
- **Observability**: structured retrieval debug events when `KA_DEBUG` / `KA_RETRIEVAL_DEBUG` is set.

## Quality signal (for interviews)

- Run **`pytest`** for fast checks; run **`scripts/run_document_qa_eval.py`** with a real key for the full 8-case gold suite (routing, anchors, refusals, citations). Reports under `eval/_report*.json` are gitignored by default.
