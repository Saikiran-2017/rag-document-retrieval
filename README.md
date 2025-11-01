# RAG-Based Document Retrieval System

End-to-end **retrieval-augmented generation** over your own documents: upload PDFs, Word files, or plain text, build a local vector index, and ask questions in natural language. Answers are generated with **grounding instructions** so the model stays close to retrieved text, with **[SOURCE N]** citations and file metadata for transparency.

---

## Overview

This repository implements a full RAG pipeline in Python. Documents are loaded from disk, split into overlapping chunks with stable metadata, embedded with OpenAI, and stored in a **FAISS** index on your machine. At query time, your question is embedded with the **same model**, the index returns the most similar chunks, and a chat model answers using **only** that context. A **Streamlit** app ties the steps together so you can demo the flow without touching the command line.

The codebase is split into small modules (ingestion, chunking, vector store, LLM layer, UI) so you can explain each stage clearly in interviews or extend one part without rewriting the rest.

---

## Key Features

- **Ingestion** for **PDF** (pdfplumber), **DOCX** (python-docx), and **TXT**, with metadata such as source file, path, file type, and page number for PDFs where available  
- **Chunking** via LangChain’s `RecursiveCharacterTextSplitter`, with configurable size and overlap  
- **Embeddings** through OpenAI (`text-embedding-3-small` by default) and **langchain-openai**  
- **Local FAISS index** saved under `data/indexes/`, reloadable across sessions  
- **Top-*k* retrieval** with **L2 distance** scores for debugging and ordering  
- **Grounded generation**: strict prompts, low temperature, and explicit handling when retrieval is empty or the answer is not in context  
- **Streamlit UI**: save uploads to `data/raw/`, build or rebuild the index, ask questions, and inspect answers, sources, and chunk previews  

---

## Architecture / Workflow

The pipeline runs in this order:

1. **Ingestion** - Files in `data/raw/` are read; text is extracted and normalized into structured segments (e.g. one record per non-empty PDF page or one per TXT/DOCX file).  
2. **Chunking** - Each segment is split into smaller overlapping pieces. Every chunk keeps ingestion metadata plus `chunk_id`, `chunk_index`, and `total_chunks` for that segment.  
3. **Embedding** - Chunk text is sent to the OpenAI embeddings API; vectors are built with the same settings used later for queries.  
4. **FAISS** - Vectors and LangChain `Document` objects (text + metadata) are stored in a FAISS index and written to disk (`*.faiss` + `*.pkl`).  
5. **Retrieval** - The user question is embedded; FAISS returns the top-*k* nearest chunks and their metadata and distances.  
6. **LLM** - Retrieved chunks are formatted as numbered **[SOURCE 1], [SOURCE 2], …** blocks; the chat model answers using only that text and is instructed to cite sources or say it cannot answer from the documents.  
7. **UI** - Streamlit orchestrates upload, index build, question submission, and display of the answer, source list, and optional chunk previews.  

---

## Tech Stack

| Area | Technology |
|------|------------|
| Language | Python 3.11+ (3.12 recommended) |
| UI | Streamlit |
| LLM & embeddings | OpenAI API, `langchain-openai` |
| Orchestration | LangChain (documents, text splitters, vector store integration) |
| Vector store | FAISS (`faiss-cpu`), `langchain-community` |
| PDF / Word | pdfplumber, python-docx |
| Configuration | python-dotenv |

---

## Project Structure

```text
rag-document-retrieval/
├── streamlit_app.py              # Full RAG UI
├── requirements.txt
├── .env.example                  # Template for OPENAI_API_KEY
├── README.md
├── app/
│   ├── config.py                 # Loads API key from environment
│   ├── ingestion/
│   │   ├── loader.py             # PDF, DOCX, TXT loading
│   │   └── __init__.py
│   ├── utils/
│   │   ├── chunker.py            # TextChunk + chunk_ingested_documents
│   │   └── __init__.py
│   ├── retrieval/
│   │   ├── vector_store.py       # Embeddings, FAISS, retrieve_top_k, RetrievedChunk
│   │   └── __init__.py
│   └── llm/
│       ├── generator.py          # GroundedAnswer, generate_grounded_answer
│       └── __init__.py
├── data/
│   ├── raw/                      # Input documents (typically gitignored)
│   └── indexes/                  # Saved FAISS store (typically gitignored)
└── tests/
```

---

## Setup Instructions

1. Clone the repository and open a terminal at the project root (`rag-document-retrieval`).

2. Create and activate a virtual environment:

   ```bash
   python -m venv .venv
   ```

   **Windows (PowerShell):**

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   **macOS / Linux:**

   ```bash
   source .venv/bin/activate
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Copy `.env.example` to `.env` and set your OpenAI key:

   ```env
   OPENAI_API_KEY=sk-...
   ```

5. Ensure `data/raw/` and `data/indexes/` exist (the app creates them as needed).

---

## How to Run

Always run commands from the **project root** so imports resolve.

**Start the app:**

```bash
streamlit run streamlit_app.py
```

Streamlit prints a local URL; open it in your browser.

**Optional module checks (same root):**

```bash
python -m app.ingestion.loader
python -m app.utils.chunker
python -m app.retrieval.vector_store
python -m app.llm.generator
```

---

## How to Use

1. In the browser, use **Choose files** to pick PDF, DOCX, or TXT files, then click **Save uploaded files to data/raw**.  
2. In the sidebar, set **chunk size**, **chunk overlap**, and **top *k*** if you want values other than the defaults.  
3. Click **Build / Rebuild Index**. This ingests everything in `data/raw/`, chunks, calls the embeddings API, and saves FAISS under `data/indexes/`.  
4. Type a question and click **Ask Question**. The app loads the index, retrieves top-*k* chunks, runs grounded generation, and shows the answer.  
5. Expand **[SOURCE *N*]** rows to see `chunk_id`, file name, page, and path; scroll to **retrieved chunk previews** to see what text was retrieved and its distance score.  

Rebuild the index after you add or replace documents so vectors match your library.

---

## Example Workflow

You save a two-page internal FAQ as a PDF into `data/raw/`, build the index, then ask: *“What is the refund policy?”* The app retrieves the chunks whose embeddings are closest to that question, passes them to the model as **[SOURCE 1]** and **[SOURCE 2]**, and returns a short answer that cites those labels. You confirm the cited passages actually appear in the FAQ PDF. If nothing relevant is in the index, the model is instructed to say it cannot answer from the provided documents.

---

## Grounding and Citations

Retrieved chunks are injected into the prompt as labeled blocks: **[SOURCE 1]**, **[SOURCE 2]**, and so on. Each block includes metadata lines (chunk id, file name, path, page) and a **Text:** section that is the only part the model is told to treat as evidence.

The UI lists the same chunks as structured **sources** (aligned with those numbers). The model is asked to cite **[SOURCE *N*]** when it uses a passage and to give a fixed *cannot answer* response if the text does not support an answer. That reduces hallucinations but is not mathematically guaranteed; production systems still need review, evaluation, and monitoring.

---

## Screenshots

Replace these with your own images after you capture the UI:

| Screen | Placeholder |
|--------|-------------|
| Upload & save to `data/raw/` | `docs/images/01-upload.png` |
| Build / Rebuild Index (spinner + success) | `docs/images/02-index-build.png` |
| Answer, sources, and chunk previews | `docs/images/03-answer-sources.png` |

---

## Future Improvements

- Hybrid retrieval (keyword + dense) and a reranker on top of top-*k* results  
- Automated tests and a small golden set for retrieval and answer faithfulness  
- Docker image and Streamlit Cloud deployment with secrets management  
- OCR or a dedicated pipeline for scanned PDFs  
- Optional async embedding and clearer progress for large folders  

---

## Interview Talking Points

- **What problem RAG solves:** The model’s weights are static; RAG grounds answers in **your** documents that were not in training.  
- **Why chunk:** Embeddings and context windows work on bounded spans; overlap helps when an answer sits across a split.  
- **Why FAISS:** Fast similarity search locally without running a separate vector database for an MVP.  
- **Why the same embedding model for index and query:** Vector dimension and geometry must match; mixing models breaks retrieval.  
- **What top-*k* trades off:** Higher *k* adds context but also noise and cost; lower *k* can miss relevant passages.  
- **How you explain grounding:** The LLM only sees retrieved text in numbered blocks and is instructed not to use outside knowledge; citations tie claims back to chunks you can audit.  
- **Honest limitation:** Prompting improves faithfulness but is not a formal guarantee; evals and human review still matter.  

---

## License

Specify a license (for example MIT) when you publish the repository, or retain default copyright until you add a `LICENSE` file.
