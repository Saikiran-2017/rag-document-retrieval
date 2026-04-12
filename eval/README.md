# Document QA evaluation harness

Repeatable checks for routing, retrieval anchors, refusal calibration, and citation surface form.

## Run (full stack, costs tokens)

Use the **same Python environment** as the app (`.venv` with `pip install -r requirements.txt`). The global `python` may miss `rank_bm25` or `langchain_openai`.

Provide a real `OPENAI_API_KEY` via **environment variable** (safest for public clones), or a **gitignored** `.env.local` / `.env`. Never commit secrets. The script validates the key before building the index.

From the repository root:

```bash
# Windows CMD: env var first (recommended)
set OPENAI_API_KEY=sk-...your-real-key...
set PYTHONPATH=.
.venv\Scripts\python.exe scripts/run_document_qa_eval.py --json-report eval/_report_local.json
```

Optional JSON output is gitignored by default (`eval/_report*.json`).

```bash
# Same without inline secret: put OPENAI_API_KEY in .env.local (gitignored), then:
set PYTHONPATH=.
.venv\Scripts\python.exe scripts/run_document_qa_eval.py --json-report eval/_report_local.json
```

The runner disables streaming and web search for stable comparisons. If the key is invalid, exit code is `2` and the JSON report records `eval_status: "blocked"`.

**Troubleshooting:** run **`scripts/verify_openai_env.py`** for a masked key check; or run this script with `--verbose-key` (or `KA_EVAL_KEY_DIAG=1`). The app picks the first **valid** (non-placeholder) key: real **process** env wins in CI; locally, a real key in **`.env.local`** overrides sample/template lines in **`.env`** or a bad value pre-set in the shell.

**Retrieval pipeline (stderr JSON lines):** set `KA_RETRIEVAL_DEBUG=1` (or `KA_DEBUG=1`) while running Streamlit, the API, or this eval. Each chat turn emits structured events: `turn_begin`, `index_loaded`, `retrieval_hybrid_done` (pool sizes, top hit L2/RRF previews, grounding gate reason), and `routing_decision` (whether the LLM gets a grounded prompt).

**Manual hybrid search:** `scripts/debug_retrieval_smoke.py` runs the **same pipeline as chat** (hybrid pool → rerank → trust filter → grounding gate → `select_generation_context`) for CEO / revenue / playbook / latency queries. With a real key it syncs `data/raw` + `data/indexes/faiss_store`. With **`--skip-sync`** it loads an existing FAISS folder only (no API calls); use **`--work-dir eval/_work`** after a successful eval if those artifacts exist. Exit **3** = placeholder key and no index to load.

## Artifacts

| Path | Role |
|------|------|
| `eval/gold_cases.json` | Queries + expected routing / keywords / anchors |
| `eval/fixtures/` | Short corpus copied into a temp `data/raw`-style folder |
| `eval/scoring.py` | Dimension checks (no extra frameworks) |
| `eval/harness.py` | Index build + `chat_service.answer_user_query` loop |

## Extend the dataset

1. Add or edit a UTF-8 text file under `eval/fixtures/`.
2. List it in `corpus_files` inside `gold_cases.json`.
3. Append a case with `category`, `query`, and `expected` fields.

## CI

- `pytest tests/test_eval_scoring.py` exercises scoring without API calls.
- Full eval is opt-in (requires keys + spend).
