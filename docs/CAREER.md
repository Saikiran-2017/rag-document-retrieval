## Recruiter-friendly GitHub “About” text

**Document-grounded chat with honest routing:** upload PDFs/DOCX/TXT, sync to a local FAISS index, retrieve with hybrid search (BM25 + embeddings, RRF), and answer with citations only when retrieval strength and file health make it safe. Includes Streamlit + FastAPI + Next.js UIs, eval harness, and debug diagnostics.

## Resume bullets (pick 3–5)

- Built a production-style RAG document Q&A app with **hybrid retrieval** (FAISS + BM25, RRF fusion), **grounding gates**, and **per-file health** to prevent weak/unsafe citations on real corpora.
- Implemented an end-to-end ingestion pipeline for **PDF/DOCX/TXT** with table-aware DOCX extraction, safer PDF fallbacks, optional OCR hooks, and heading-aware chunking for better section navigation.
- Added **incremental indexing** with embedding cache + index state tracking to reduce rebuild time and cost while keeping sync correctness.
- Shipped two frontends (Streamlit + **Next.js** chat UI) on a shared Python domain layer with **SSE streaming**, library readiness UX, and reliable upload→sync→ask workflow.
- Created a regression-friendly eval harness and real-doc pack validating routing, retrieval anchors, refusals, and citations (benchmark **8/8 pass**), plus debug-only diagnostics for production triage.

## 60–90 second interview explanation

“This project is a trust-focused RAG assistant. Uploads land in a raw library, Sync builds a FAISS index using OpenAI embeddings with a disk cache, and queries run hybrid retrieval (dense + BM25) fused with RRF. Before grounding, results go through rerank + trust filtering and a gating layer that decides whether we should cite documents or fall back to a general answer. The same shared `app/` layer powers Streamlit and a FastAPI + Next.js product UI. I also built an eval harness and debug-only diagnostics so regressions are easy to reproduce and inspect.”

## Portfolio summary paragraph

Knowledge Assistant is an end-to-end RAG system designed for real documents and real UX: it provides a clean upload→sync→ask workflow, grounded answers with citations when evidence is strong, and deliberate fallbacks when it is not. The system combines FAISS vector search with BM25 keyword retrieval (RRF fusion), uses explicit trust gates to avoid misleading citations, and includes regression evaluation plus debug-mode diagnostics for production-style observability.

