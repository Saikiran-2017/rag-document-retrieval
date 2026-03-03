"""
FastAPI entrypoint.

**Local dev** (repository root)::

    set PYTHONPATH=.
    python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

**Production (example)**::

    export KA_ENV=production
    uvicorn backend.app.main:app --host 0.0.0.0 --port 8000 --workers 1

**Docker:** see repository ``Dockerfile`` and ``docker-compose.yml``.

Reliability logging: ``KA_RELIABILITY_LOG=1`` (same as Streamlit).
"""

from __future__ import annotations

import os
from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

import app.config  # noqa: F401; load_repo_dotenv() before services read OPENAI_API_KEY

from app.persistence import chat_store

from backend.app.core.config import get_cors_allow_origins
from backend.app.routes import chat, chats, documents, health, sync, upload


def _is_production_env() -> bool:
    return os.environ.get("KA_ENV", "").strip().lower() in ("production", "prod", "1")


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_store.init_db()
    yield


_prod = _is_production_env()
app = FastAPI(
    title="Knowledge Assistant API",
    version="0.1.0",
    lifespan=lifespan,
    docs_url=None if _prod else "/docs",
    redoc_url=None if _prod else "/redoc",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=get_cors_allow_origins(),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["Content-Type"],
)

app.include_router(health.router)

v1 = APIRouter(prefix="/api/v1")
v1.include_router(upload.router)
v1.include_router(sync.router)
v1.include_router(documents.router)
v1.include_router(chat.router)
v1.include_router(chats.router)
app.include_router(v1)
