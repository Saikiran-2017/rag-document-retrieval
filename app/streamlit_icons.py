"""Emoji constants for Streamlit APIs that validate ``icon`` (e.g. ``st.toast``).

Streamlit rejects plain symbols such as U+2713 CHECK MARK; use pictographic emoji only.
See https://docs.streamlit.io/ — keep values single-codepoint emoji where possible.
"""

# Success / failure / alerts
TOAST_SUCCESS = "✅"
TOAST_ERROR = "❌"
TOAST_WARNING = "⚠️"
TOAST_INFO = "ℹ️"

# Actions (use consistently for the same UX path)
TOAST_LIBRARY = "📚"
TOAST_SYNC = "🔄"
TOAST_DELETE = "🗑️"
