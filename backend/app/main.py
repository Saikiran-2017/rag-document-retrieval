"""
FastAPI entrypoint (Phase 18A).

Run from **repository root** so ``app`` and ``backend`` resolve::

    set PYTHONPATH=.
    python -m uvicorn backend.app.main:app --reload --host 127.0.0.1 --port 8000

Reliability logging: set ``KA_RELIABILITY_LOG=1`` (same as Streamlit).
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import APIRouter, FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.persistence import chat_store

from backend.app.routes import chat, chats, documents, health, sync, upload


@asynccontextmanager
async def lifespan(app: FastAPI):
    chat_store.init_db()
    yield


app = FastAPI(
    title="Knowledge Assistant API",
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(health.router)

v1 = APIRouter(prefix="/api/v1")
v1.include_router(upload.router)
v1.include_router(sync.router)
v1.include_router(documents.router)
v1.include_router(chat.router)
v1.include_router(chats.router)
app.include_router(v1)
