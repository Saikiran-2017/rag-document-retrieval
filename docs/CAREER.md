## Recruiter-friendly GitHub “About” text

**Document-grounded chat with honest routing:** ingest PDFs/DOCX/TXT into a local FAISS index, hybrid retrieve (BM25 + embeddings, RRF), and answer with citations only when retrieval and per-file health justify it—plus Streamlit and FastAPI + Next.js UIs, gold eval harness, and safe local secret handling (`.env.local` overrides sample keys).

## Resume bullets (pick 3–5)

- Built a production-style RAG document Q&A app with **hybrid retrieval** (FAISS + BM25, RRF fusion), **grounding gates**, and **per-file health** so citations appear only when evidence is strong enough.
- Implemented end-to-end ingestion for **PDF/DOCX/TXT** (table-aware DOCX, PDF fallbacks, optional OCR hooks) with **incremental indexing**, embedding disk cache, and FAISS on disk.
- Shipped **Streamlit** and **FastAPI + Next.js** (SSE streaming, library readiness UX) on one shared **`app/`** domain layer—no duplicated retrieval logic in frontends.
- Added **deterministic `OPENAI_API_KEY` resolution** for local dev (valid process key → `.env.local` → `.env`; placeholders never override real keys) plus **`verify_openai_env`** for masked checks.
- Delivered regression tooling: **`pytest`**, gold **document QA eval** (8 cases: routing, anchors, refusals, forbidden tokens), **phase28 real-doc pack**, and **`brutal_product_check`** sync→chat smoke; debug diagnostics stay off by default.

## 60–90 second interview explanation

“This is a trust-first RAG assistant. Uploads go to a raw library; Sync parses, chunks, embeds with OpenAI, and writes FAISS plus a manifest with per-file readiness. Chat does hybrid retrieval—dense plus BM25 fused with RRF—then reranking, trust filtering, and grounding gates so we only cite documents when retrieval and file health support it; otherwise we answer in general mode instead of faking sources. The same `app/` package powers Streamlit and a Next.js client over FastAPI. I also hardened local secrets so a real key in `.env.local` wins over template lines, and I ship scripts for verify, gold eval, and a brutal sync-plus-chat check so regressions are easy to prove.”

## Portfolio summary paragraph

Knowledge Assistant is an end-to-end RAG workspace: upload→sync→ask with grounded answers and `[SOURCE n]` citations when evidence is strong, and deliberate general/web/blended fallbacks when it is not. Hybrid retrieval, trust gates, incremental indexing, and SQLite chat history support a credible demo and a credible engineering story—backed by pytest, a gold eval harness, and optional real-doc packs for release checks.
