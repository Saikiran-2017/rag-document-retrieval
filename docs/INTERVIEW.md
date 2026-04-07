## Interview-ready explanation (short)

### What it is

Knowledge Assistant is a production-style RAG app for document Q&A that prioritizes **trust**: it answers with citations only when retrieval and file health justify grounding, and it intentionally falls back to **general** replies when evidence is weak.

### How it works (end-to-end)

1. **Upload** files (PDF/DOCX/TXT) → stored in `data/raw/`.
2. **Sync** builds the index:
   - parse + normalize text (PDF/DOCX/TXT; optional OCR off by default)
   - chunk with heading-aware separators
   - embed with a disk cache
   - store vectors in FAISS + write manifest/library state
3. **Chat**:
   - optional query rewrite
   - hybrid retrieve (FAISS vector + BM25) fused via RRF
   - rerank + trust filter
   - grounding gates decide whether to supply context
   - answer mode: grounded / general / web / blended

### Why hybrid retrieval (vector + BM25)

- Vector retrieval handles paraphrase and semantic match.
- BM25 is strong for **exact identifiers**, names, and sparse lines.
- RRF fusion stabilizes relevance across messy corpora and reduces single-method failure modes.

### Why grounding gates

- “Always ground” systems produce confident but wrong citations on weak matches.
- Gates use **retrieval strength + file health** to decide whether citations are appropriate.

### Why optional OCR

- OCR is environment-heavy and can slow ingestion.
- The default path favors fast text-based extraction; OCR is an opt-in fallback for scanned PDFs.

### Why caching

- Embedding cache reduces cost and makes re-syncs faster.
- FAISS load cache avoids repeated disk loads per chat turn (invalidated on rebuild).

### Observability

- Structured retrieval logs (`KA_RETRIEVAL_DEBUG=1`).
- Debug-only diagnostics on responses (`KA_DEBUG=1` + `localStorage.KA_DEBUG=1` in web UI).

