## Screenshot checklist (recommended)

Add images under `docs/images/` and link them from `README.md`.

### Portfolio order (working product)

For a **release / showcase** repo, capture the **happy path first** so reviewers see grounded behavior immediately:

1. **`web_library_ready.png`** — documents listed with healthy readiness (proves sync + manifest).
2. **`web_grounded_sources.png`** — answer with `[SOURCE n]` and Sources panel (proves RAG).
3. **`web_empty.png`** — clean hero (optional opener).
4. **`web_general.png`** — honest **general** mode when the library is empty or the question is off-doc (proves routing).

Then add upload/sync/diagnostics as needed (see below).

### Must-have captures

- **Empty state (web)**: `web` app, no messages, sidebar visible.
  - Filename: `docs/images/web_empty.png`
- **Upload flow (web)**: upload picker + post-upload hint.
  - Filename: `docs/images/web_upload.png`
- **Sync in progress (web)**: library “Sync documents” busy state + hint line.
  - Filename: `docs/images/web_syncing.png`
- **Library ready state (web)**: document list with health pills.
  - Filename: `docs/images/web_library_ready.png`
- **Grounded answer + sources (web)**: answer with `[SOURCE n]` and Sources panel open.
  - Filename: `docs/images/web_grounded_sources.png`
- **General answer (web)**: a question answered in `general` mode (no sources).
  - Filename: `docs/images/web_general.png`
- **Diagnostics (web, debug)**: set `localStorage.KA_DEBUG=1` and show “Developer diagnostics”.
  - Filename: `docs/images/web_diagnostics.png`

### Optional captures (nice to have)

- **Streamlit demo**: grounded answer + sources/excerpts expanders.
  - Filename: `docs/images/streamlit_grounded.png`
- **Readiness banner**: “library needs sync” banner after changing files.
  - Filename: `docs/images/web_needs_sync_banner.png`

### How to enable web diagnostics (safe)

1. Run the app normally.
2. In the browser console:

```js
localStorage.setItem("KA_DEBUG", "1");
location.reload();
```

3. Server-side diagnostics require `KA_DEBUG=1` on the backend process.

