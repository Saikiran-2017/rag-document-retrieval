"""User-facing copy, merged status notes, and small display-formatting helpers."""

from __future__ import annotations

import html
from pathlib import Path

APP_NAME = "Knowledge Assistant"
# First-screen value prop (also used where TAGLINE was referenced).
EMPTY_STATE_VALUE_PROP = "Chat normally—when your library applies, answers cite your files."
TAGLINE = EMPTY_STATE_VALUE_PROP
SIDEBAR_CAPTION = "Upload, sync, ask."

# Empty-state: short trust/context line (HTML escaped where injected).
HERO_BEST_FOR = (
    "Best for a focused set of files. Works without uploads. Complements your other AI tools."
)

# Sidebar positioning (modest, factual; no product names as endorsements—examples only where needed).
WHY_THIS_WORKSPACE_MD = """
- When retrieval applies, answers use **your uploaded documents**, not only the model's general knowledge.
- Open **Sources** and **Supporting excerpts** under a reply to inspect what was used.
- Fits a **defined file set** you already care about.
- **Complements** tools like ChatGPT, Claude, or Copilot—not a substitute for every workflow.
"""

COMPARISON_MD = """
| | General-purpose assistants | This workspace |
|:---|:---|:---|
| **Primary focus** | Broad knowledge across many topics | Your **uploaded documents** |
| **Typical answers** | General responses | **Source-aware** replies when your library applies |
| **Retrieval** | Flexible chat for varied tasks | **Corpus-focused** search over your synced files |
| **Natural fit** | Open-ended use | **Document workflow**: upload → sync → ask |

Neither column is "better" overall—they suit different jobs. Use this app when your own files should anchor the answer.
"""
PREVIEW_CHARS = 300
# Sidebar UI: max value for "Sources per answer" (retrieval top_k).
SIDEBAR_TOP_K_MAX = 12
# Default retrieval width (lower = faster retrieval + smaller prompts).
DEFAULT_TOP_K = 3

STARTER_QUESTIONS = [
    "Summarize the key points I should remember.",
    "What themes appear in my documents?",
    "Give me three concise writing tips.",
]

# User-facing copy only (no indexing / embedding / vector jargon).
MSG_LIBRARY_UPDATED = "Library ready."
MSG_PREPARE_DOCS_FAILED = "Couldn't prepare documents. Please try again."
MSG_PREPARE_SETTINGS_HINT = "Couldn't prepare documents. Adjust Preferences, then try again."
MSG_READ_FILES_FAILED = "Couldn't read text from those files. Try another file or check they aren't empty."
MSG_NO_DOCS = "Add documents in the sidebar, sync, then ask again."
MSG_EMPTY_MESSAGE = "Please enter a message."
# Resilience: short, calm copy (no API/technical jargon for end users).
MSG_UPLOAD_FAILED = "We couldn't save those files. Your question is answered below."
MSG_DOCS_PREP_FAILED = "We couldn't finish preparing your documents. Your question is answered below without using those files."
MSG_LIBRARY_UNAVAILABLE = "Your library isn't available for this reply, so this answer is general."
MSG_GROUNDED_FALLBACK_NOTE = "Your documents couldn't be used for this reply, so this answer is general."
MSG_WEB_RESULTS_THIN = (
    "Web search didn't return solid enough snippets to cite safely, so this answer is general."
)
MSG_SERVICE_UNAVAILABLE = "Something went wrong on our side. Please try again in a moment."

# Document library health (plain language; avoid embedding / vector jargon in UI).
DOC_HEALTH_COULD_NOT_READ_FILE = (
    "We couldn't read usable text from this file. Try PDF, Word, or plain text, or re-export the file."
)
DOC_HEALTH_NO_TEXT_EXTRACTED = (
    "No text could be extracted. The file may be image-only or empty—try a text-based export if you can."
)
DOC_HEALTH_INDEX_SECTIONS_MISMATCH = (
    "This file is saved, but search didn't match the text we extracted. Try Sync again."
)
DOC_HEALTH_SEARCH_PARTIALLY_RELIABLE = (
    "Search is only partially reliable for your library right now, so answers may miss some documents."
)
DOC_HEALTH_INDEX_INTERRUPTED = "We couldn't finish indexing your files. Try Sync again."
DOC_HEALTH_GENERIC_FAILURE = "Something went wrong while preparing this file. Try Sync again or use another file."

def merge_notes(*parts: str | None) -> str | None:
    bits = [p.strip() for p in parts if p and str(p).strip()]
    if not bits:
        return None
    return "\n".join(bits)


def preview_text(text: str, max_len: int = PREVIEW_CHARS) -> str:
    t = (text or "").strip()
    if len(t) <= max_len:
        return t
    return t[: max_len - 1].rstrip() + "..."


def display_path_hint(path_str: str | None) -> str | None:
    if not path_str:
        return None
    try:
        p = Path(path_str)
        return p.name if p.name else path_str
    except (TypeError, ValueError):
        return path_str


def render_status_note_html(note: str) -> str:
    """Escape user-visible status lines; preserve intentional newlines from merged notes."""
    return html.escape(note).replace("\n", "<br>")
