# Knowledge Assistant

A chat-style assistant that answers **general questions** and, when you add your own **PDF, Word, or text files**, can ground replies in that material with **sources** and short **excerpts** so you can see what text was used.

---

## Key features

- **Chat-first interface** (Streamlit): conversation, optional file attachments above the input, and a simple empty state with suggested prompts.
- **Smart routing**: with no documents, or when retrieval does not match well, you get a normal assistant reply (no fake citations). When retrieval is strong, answers follow your files and show source references.
- **Local document pipeline**: files are chunked, embedded with OpenAI, and stored in a **FAISS** index on your machine under `data/indexes/`.
- **Transparent grounded answers**: optional collapsible **Sources** and **Supporting excerpts** for document-backed replies.
- **Configurable behavior** (sidebar): sources per answer, context length, and overlap, with sensible defaults out of the box.

---

## How it works (simple)

1. You type a message (and optionally attach files). New files are saved and indexed for your library.
2. Your question is turned into an embedding and compared to chunks from your files.
3. If the best matches are strong enough, the model answers using only those passages and cites them. Otherwise you get a general answer, the same as when you have no files or the question is not about your documents.

No web search: answers come from the model and, when grounded, from the text you uploaded.

---

## Architecture (high level)

| Stage | What happens |
|--------|----------------|
| **Ingestion** | PDFs, DOCX, and TXT are read from `data/raw/` and normalized with metadata (e.g. page hints for PDFs). |
| **Chunking** | Text is split into overlapping segments with stable IDs for traceability. |
| **Indexing** | Chunks are embedded and stored in a FAISS vector index on disk. |
| **Query** | The user message is embedded; nearest chunks are retrieved and scored. |
| **Generation** | Either a general chat completion or a **grounded** completion that only uses labeled context blocks for citations. |

The UI and routing live in `streamlit_app.py`. Ingestion, chunking, retrieval, and LLM helpers are organized under `app/`.

---

## Tech stack

| Layer | Choices |
|--------|---------|
| Language | Python 3.11+ |
| UI | Streamlit |
| Models | OpenAI (chat + embeddings) via LangChain |
| Vectors | FAISS (`faiss-cpu`), local files |
| Documents | pdfplumber, python-docx |
| Config | `python-dotenv` |

---

## Run locally

**Prerequisites:** Python 3.11+, an [OpenAI API key](https://platform.openai.com/).

```bash
git clone <your-repo-url>
cd rag-document-retrieval
python -m venv .venv
```

Activate the environment:

- **Windows (PowerShell):** `.\.venv\Scripts\Activate.ps1`
- **macOS / Linux:** `source .venv/bin/activate`

```bash
pip install -r requirements.txt
cp .env.example .env          # Windows: copy .env.example .env
```

Put your key in `.env`:

```env
OPENAI_API_KEY=sk-...
```

Start the app from the project root:

```bash
streamlit run streamlit_app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

---

## Screenshots

Add images under `docs/images/` and drop them into the table when you publish the repo.

| | File (suggested) | Caption |
|---|------------------|---------|
| 1 | `docs/images/01-empty-state.png` | Empty state with title, short value line, and starter prompts. |
| 2 | `docs/images/02-chat-general.png` | General question: plain reply, no source panels. |
| 3 | `docs/images/03-upload-and-ask.png` | Files attached, user message, assistant reply. |
| 4 | `docs/images/04-grounded-answer.png` | Grounded reply with Sources / Supporting excerpts (collapsed). |
| 5 | `docs/images/05-sidebar.png` | Sidebar: library list, New chat, options, About. |

*Tip: one shot with the sidebar closed (chat-focused) and one with it open (library visible) works well for README readers.*

---

## Why this project is useful

- **Learning and interviews:** A small, readable RAG pipeline (ingest → chunk → embed → retrieve → generate) that you can explain end-to-end without a separate vector database.
- **Portfolio:** Demonstrates practical LLM usage: grounded prompts, citations, and honest fallbacks when retrieval is weak.
- **Real workflows:** Fits note-taking, policies, study material, or internal docs where **your** text is the source of truth, not the open web.

---

## Repository layout

```text
rag-document-retrieval/
├── streamlit_app.py       # UI and query routing
├── requirements.txt
├── .env.example
├── app/                   # ingestion, chunking, retrieval, LLM
├── data/raw/              # uploaded documents (often gitignored)
├── data/indexes/          # FAISS index files (often gitignored)
└── docs/images/           # screenshots (optional)
```

---

## License

MIT License - see [LICENSE](LICENSE) for details.

Copyright (c) 2025 Sai Kiran. Free to use, modify, and distribute under the terms of the MIT License.
