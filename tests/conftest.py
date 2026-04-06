"""
Pytest hooks: stub optional heavy imports so routing tests collect without a full venv.

Full integration still requires ``pip install -r requirements.txt``.
"""

from __future__ import annotations

import sys
from unittest.mock import MagicMock

_st = MagicMock()
_st.session_state = {}
sys.modules.setdefault("streamlit", _st)

_stubs = (
    "pdfplumber",
    "pypdf",
    "langchain_openai",
    "faiss",
    "faiss_cpu",
)
for _name in _stubs:
    sys.modules.setdefault(_name, MagicMock())

_m = sys.modules["langchain_openai"]
_m.OpenAIEmbeddings = MagicMock

_r = MagicMock()
_r.BM25Okapi = MagicMock
sys.modules.setdefault("rank_bm25", _r)
