## Demo walkthrough (5–7 minutes)

This script is optimized for live demos, recruiter screenshares, and interview walkthroughs.

### 0) Prerequisites (30 seconds before you share screen)

- **Real `OPENAI_API_KEY`:** prefer **`.env.local`** (gitignored). If **`.env`** still contains a template key, that is fine—the loader uses the first **valid** key (see root **README**).
- Quick check (prints **masked** key only): `set PYTHONPATH=.` then `python scripts/verify_openai_env.py` → exit **0**, `ready_for_local_dev: true`.
- Start **Next.js + API** or **Streamlit** as in the root README.

### 1) Show the empty state

- Open the web UI (`http://localhost:3000`).
- Call out: “Chat-first UX. Upload → Sync → Ask.”

### 2) Upload documents

- Upload a small set (PDF + DOCX + TXT if available).
- Say: “The app does not pretend it indexed anything yet; it tracks readiness.”

### 3) Sync and explain what happens

- Click **Sync documents**.
- Explain: parse → normalize → chunk → embed (cached) → FAISS → manifest health.

### 4) Ask a broad overview question (grounded)

- Example: “What is this document about in plain language?”
- Show: citations + Sources panel.
- Say: “If grounding isn’t justified, it falls back to general instead of hallucinating citations.”

### 5) Ask a narrow factual question (entity lookup)

- Example: “Who is named as CFO or finance lead?”
- Show: short factual answer + citation.

### 6) Ask a section navigation question

- Example: “What does section 7 say about disaster recovery?”
- Show: section anchor / heading behavior.

### 7) Observability (debug mode)

- Backend: run with `KA_DEBUG=1`.
- Web: set `localStorage.KA_DEBUG=1` and reload.
- Open **Developer diagnostics** under an assistant message.
- Call out: route, gate reason, pool sizes, selected sources.

### 8) Close with limitations (honest)

- Scanned PDFs need optional OCR (off by default).
- Very large libraries may require tuning / sharding / managed vector DB.

