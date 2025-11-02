# Knowledge Assistant

**Document-grounded chat** in a simple Streamlit app: ask everyday questions, upload PDFs / Word / text, sync your library, and get answers that can cite **sources** and **supporting excerpts** when your files are relevant—without inventing citations when they are not.

---

## Overview

Knowledge Assistant is a **local-first RAG (retrieval-augmented generation)** workspace. You chat in the main area; documents live in the sidebar. The app embeds your text into a **FAISS** index on disk, retrieves the closest chunks per question, and either answers **from those passages with citations** or falls back to a **general** assistant reply when the library is empty, sync fails, retrieval is weak, or you clearly are not asking about documents.

It is built for **demos, portfolios, and interviews**: the code is modular, failures are handled explicitly, and optional **developer debug** (`KA_DEBUG=1`) exposes routing metadata without leaking secrets.

---

## Why this app exists

General-purpose chat tools are broad; this app is **narrow on purpose**: it shines when you have a **known set of files** (notes, policies, specs, study material) and want answers **anchored to that corpus**, with **inspectable evidence**. It does not replace every AI product—it **complements** them for document-centric Q&A and light tasks (summarize, extract, compare) over your own uploads.

---

## Key capabilities

| Area | What you get |
|------|----------------|
| **Chat & Q&A** | Default **Auto** mode: general chat, optional **no-retrieval fast path** for short non-document questions, and **grounded** answers when retrieval is strong. |
| **Library** | Sidebar **upload** → **Sync**; files under `data/raw/`, index under `data/indexes/`. |
| **Sources & excerpts** | Collapsible **Sources referenced** and **Supporting excerpts** on grounded turns. |
| **Task modes** | **Summarize**, **Extract**, **Compare** (optional scope for summarize): one LLM call each, same evidence rules—no agents. |
| **Resilience** | Upload/index/retrieval/generation failures route to **safe general answers**; session sync state only advances when rebuild succeeds. |
| **Debug** | `KA_DEBUG=1` or code flag: sidebar JSON of routing and errors (no API keys in UI). |

---

## Product highlights

- **Chat-first UX** with a polished empty state, starter prompts, and calm positioning copy (“Why this workspace”, comparison table).
- **Honest routing**: no fake sources; weak retrieval → general reply with a short status line when useful.
- **Interview-friendly layout**: UI in `streamlit_app.py`; policy and I/O in `app/services/` and `app/llm/`.

---

## Architecture

### Layered design

```text
┌─────────────────────────────────────────────────────────────┐
│  streamlit_app.py   UI, session, ingest wiring, CSS         │
└───────────────┬─────────────────────────────────────────────┘
                │
    ┌───────────┼───────────┬──────────────┬────────────────┐
    ▼           ▼           ▼              ▼                ▼
 upload    index       chat          doc_task         debug / message
 service   service     service       service          service
    │           │           │              │                │
    └───────────┴───────────┴──────┬───────┴────────────────┘
                                   ▼
                    app/llm/generator.py  (general + grounded + document tasks)
                                   │
                    app/retrieval/vector_store.py  (FAISS)
                                   │
                    app/ingestion + app/utils/chunker
```

### Module roles

| Layer | Responsibility |
|--------|----------------|
| **UI (`streamlit_app.py`)** | Page layout, sidebar (documents, preferences, task mode, positioning), chat rendering, `ingest_composer_attachments` + manual sync orchestration (same session keys as before), global CSS, toasts. |
| **Upload service** | Save uploaded bytes to `data/raw/`; supported extensions; returns counts and filenames. |
| **Index service** | List library files, **fingerprint** for sync, **rebuild** FAISS from chunks, **load** store, **`ensure_index_matches_library`** (rebuild + update `kb_sync_fingerprint` on success). Cached OpenAI embeddings via Streamlit `cache_resource`. |
| **Chat service** | **`answer_user_query`**: empty query handling; **`auto`** → legacy path (library check, sync, fast path, retrieve, `retrieval_is_useful`, grounded vs general); **`summarize` / `extract` / `compare`** → delegates to doc task service. **`AssistantTurn`** shaping and `append_assistant_turn`. |
| **Doc task service** | Document-only tasks: retrieval width, optional file filter for summarize, gates (e.g. compare needs ≥2 files, extract/compare use L2 usefulness + relaxed retry), calls **`generate_document_task_answer`** in the generator. |
| **Debug service** | `KA_DEBUG` / flag, per-turn dict, `merge`, `short_exc`, sidebar JSON panel. |
| **Message service** | User-facing strings, `merge_notes`, preview/path/status HTML helpers. |
| **Generator layer** | OpenAI chat via LangChain: **general** system prompt, **grounded** Q&A with `[SOURCE n]` blocks, **summarize / extract / compare** system prompts—each a **single** completion, temperature 0 for grounded/tasks. |

---

## How it works end to end

1. **Upload** (sidebar) saves files to `data/raw/`.
2. **Sync** (or send with new files) runs **ingest**: save → **chunk** → **embed** → **write FAISS** → set **library fingerprint** in session when successful.
3. **Ask** (chat input): **`answer_user_query`** runs. In **Auto**, the app may skip retrieval for very short non-document questions; otherwise it **ensures the index matches** the fingerprint, **loads FAISS**, **retrieves top-k**, and if the best hit is strong enough, calls **grounded generation** with numbered context; else **general generation** (and notes when the library was unavailable or retrieval weak).
4. **Task modes** bypass the “chatty” fast path and run **task-specific** prompts over retrieved chunks, with explicit failure messages when prerequisites are not met.
5. **UI** renders markdown, optional status lines, and expanders for sources and excerpts on grounded-shaped messages.

---

## Routing behavior (Auto mode)

- **No files on disk** → general answer (no citations).
- **Sync / rebuild failed** → general answer + library-unavailable note.
- **Short query with no document keywords** → general answer (skip FAISS) for latency and relevance.
- **FAISS load or retrieve error** → general answer (debug keys if enabled).
- **Weak best-hit distance** → general answer (no fake sources).
- **Strong retrieval** → grounded answer; **grounded LLM error** → general fallback + note.

Task modes add their own gates (e.g. compare needs two files; extract/compare use retrieval quality).

---

## Failure resilience

- **Ingest** errors do not block the assistant turn: user still gets a reply; warnings merge into the status line when relevant.
- **`safe_general_answer`** wraps general generation so the UI never hard-crashes on model errors.
- **Session integrity**: `kb_sync_fingerprint` updates only after a successful rebuild; composer widget reset aligns with successful ingest.
- **Debug** records routing and exception summaries without printing stack traces in the main UI.

---

## Screenshots

Place PNGs under **`docs/images/`** and reference them below (uncomment or add links after you capture).

| # | Suggested file | What to show |
|---|----------------|--------------|
| 1 | `01-hero-empty.png` | Empty state: hero panel, value prop, starter prompts. |
| 2 | `02-chat-general.png` | General question; clean assistant reply; **no** source expanders. |
| 3 | `03-sidebar-library.png` | Sidebar open: upload, library list, Sync, Preferences (task mode visible). |
| 4 | `04-grounded-answer.png` | Document-grounded reply; **Sources referenced** + **Supporting excerpts** (one expanded). |
| 5 | `05-task-summarize.png` | Task mode **Summarize** + composer hint above input; summary with sources. |
| 6 | `06-compare-or-extract.png` | **Compare** or **Extract** with a grounded-style answer. |
| 7 | `07-dark-or-positioning.png` | Optional: dark theme **or** “Why this workspace” expander open. |

```markdown
<!-- Example once files exist:
![Empty state](docs/images/01-hero-empty.png)
-->
```

---

## Local setup

**Prerequisites:** Python **3.11+**, [OpenAI API key](https://platform.openai.com/).

```bash
git clone <your-repo-url>
cd rag-document-retrieval
python -m venv .venv
```

**Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`  
**macOS / Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
cp .env.example .env    # Windows: Copy-Item .env.example .env
```

`.env`:

```env
OPENAI_API_KEY=sk-...
```

**Developer debug (optional):**

```bash
set KA_DEBUG=1          # Windows CMD
$env:KA_DEBUG="1"      # PowerShell
export KA_DEBUG=1      # macOS / Linux
```

```bash
streamlit run streamlit_app.py
```

Open `http://localhost:8501` (or the URL Streamlit prints).

**Deploy:** See [DEPLOYMENT.md](DEPLOYMENT.md) for a Streamlit-friendly host (e.g. Render).

---

## Usage (quick)

1. Open **Preferences** → leave **Task mode** on **Auto** for normal chat.
2. **Upload** documents → **Sync documents**.
3. Ask questions; open **Sources referenced** / **Supporting excerpts** when the answer is document-backed.
4. Switch task mode to **Summarize**, **Extract**, or **Compare** for focused prompts (still one-shot completions over retrieved text).
5. **New chat** clears the thread; the library on disk stays until you change files.

---

## Limitations and tradeoffs

- **Retrieval is approximate**: embedding similarity ≠ perfect relevance; the L2 gate reduces bogus grounding but does not guarantee correctness.
- **Summaries / tasks over large libraries** use **top-k excerpts**, not full-document reads in one shot—appropriate for this stack, not a replacement for dedicated summarization pipelines at huge scale.
- **Single-user / local session**: Streamlit `session_state`; not a multi-tenant production API.
- **Costs**: OpenAI usage for embeddings + chat per query/sync.
- **Trust**: always verify critical answers against source text.

---

## Future improvements

- Optional **vector store** swap (e.g. managed DB) for multi-user demos.
- **Evaluation** set (retrieval hit rate, grounded answer faithfulness) for interviews.
- **Export** chat or cited snippets to Markdown/PDF.
- **Auth** and deployed secrets pattern for a public demo host.

---

## Repository layout

```text
rag-document-retrieval/
├── streamlit_app.py          # UI, layout, CSS, event wiring
├── requirements.txt
├── .env.example
├── LICENSE
├── app/
│   ├── services/
│   │   ├── upload_service.py
│   │   ├── index_service.py
│   │   ├── chat_service.py
│   │   ├── doc_task_service.py
│   │   ├── debug_service.py
│   │   └── message_service.py
│   ├── llm/generator.py      # General, grounded, document-task completions
│   ├── retrieval/vector_store.py
│   ├── ingestion/
│   └── utils/chunker.py
├── data/raw/                 # documents (often gitignored)
├── data/indexes/             # FAISS (often gitignored)
└── docs/images/              # README / portfolio screenshots
```

---

## Demo script (2–4 minutes)

1. **Hook (20s)** — “This is a small RAG app: chat in the center, documents in the sidebar. It answers from your files when it should—and it won’t fake citations.”
2. **General chat (30s)** — Auto mode, short question with no uploads → fast, plain answer; point out **no** source panels.
3. **Library + grounded (60–90s)** — Upload a short PDF/TXT → **Sync** → ask something only the file contains → expand **Sources referenced** and one **Supporting excerpt**.
4. **Resilience (30s)** — Mention: sync failure doesn’t fake “updated”; weak retrieval falls back to general; optional `KA_DEBUG=1` for routing.
5. **Task mode (30–60s)** — **Summarize** or **Extract** one request → same evidence model, one LLM call, no agents.
6. **Close (15s)** — “Modular services under `app/services/`, FAISS on disk, OpenAI for embed + chat—good for explaining ingest → retrieve → generate in interviews.”

---

## License

MIT License — see [LICENSE](LICENSE).
