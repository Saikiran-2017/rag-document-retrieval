# Document QA evaluation harness

Repeatable checks for routing, retrieval anchors, refusal calibration, and citation surface form.

## Run (full stack, costs tokens)

Use the **same Python environment** as the app (`.venv` with `pip install -r requirements.txt`). The global `python` may miss `rank_bm25` or `langchain_openai`.

Provide a real `OPENAI_API_KEY` via **environment variable** (safest for public clones), or a **gitignored** `.env.local` / `.env`. Never commit secrets. The script validates the key before building the index.

From the repository root:

```bash
# Windows CMD — env var first (recommended)
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
