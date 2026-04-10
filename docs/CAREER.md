## Recruiter-friendly GitHub “About” text

Paste into the repo **About → Description** (plain text; fits GitHub’s short description field):

Document RAG with hybrid retrieval (FAISS + BM25), per-file readiness, and honest grounded vs general routing. Streamlit + FastAPI + Next.js share one Python domain layer. Validated: pytest, gold doc QA eval, transcript gate, Docker API replay, Docker UI smoke.

## Resume bullets (pick 3–5)

- Built a production-style **RAG document Q&A** stack with **hybrid retrieval** (FAISS + BM25, RRF), **grounding gates**, and **per-file library health** so citations appear only when evidence is strong enough.
- Implemented end-to-end **ingestion** for **PDF/DOCX/TXT** (table-aware DOCX, PDF fallbacks, optional OCR hooks) with **content-hash incremental indexing**, embedding disk cache, and **FAISS + manifest** on disk.
- Shipped **Streamlit** and **FastAPI + Next.js** (chat-first UI, library/sync UX, SQLite-backed chat on the API path) while keeping **all retrieval and routing in shared `app/`**—no duplicated RAG logic in frontends.
- Hardened **local and Docker secrets** (real process key and **`.env.local`** win over placeholders; optional compose merge) plus **`verify_openai_env`** for masked checks.
- Proved regressions with **`pytest`**, gold **document QA eval** (8 cases), **`brutal_product_check`**, **`transcript_product_gate`**, **Docker API HTTP replay**, and **Docker web Playwright smoke** (upload → sync → grounded Q&A with citations).

## 60–90 second interview explanation

“This is a **trust-first** RAG assistant over a personal document library. Uploads land under a raw folder; **Sync** parses, chunks, embeds with OpenAI, and writes **FAISS** plus a **manifest** with per-file readiness. **Chat** runs **hybrid retrieval**—dense search plus **BM25**, fused with **RRF**—then reranking, trust filtering, and **grounding gates** so we only cite documents when retrieval and file health justify it; otherwise we answer in **general** mode instead of inventing sources. The same **`app/`** package backs **Streamlit** and a **Next.js** client on **FastAPI**. For credibility I ship **pytest**, a small **gold eval** harness, a **transcript gate** for multi-turn replay, and **Docker** checks: an **in-container HTTP replay** against the API plus a **UI smoke** that does upload, sync, and grounded questions with **`[SOURCE n]`** visible.”

## Portfolio summary paragraph

**Knowledge Assistant** is an end-to-end RAG workspace: **upload → sync → ask** with grounded answers and **`[SOURCE n]`** citations when evidence is strong, and deliberate **general / web / blended** behavior when it is not. **Hybrid retrieval**, trust gates, incremental indexing, and SQLite chat history support a demo that matches the engineering story—backed by **pytest**, a gold **document QA eval**, transcript replay, and **Docker** API + web smoke checks you can re-run before interviews.
