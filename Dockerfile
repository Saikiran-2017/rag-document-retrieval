# Knowledge Assistant: FastAPI API + shared Python RAG stack
# Build from repository root. Persist ./data as a volume for uploads, FAISS, and SQLite.

FROM python:3.12-slim-bookworm

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Production: hide OpenAPI docs; set KA_ENV=production at runtime
EXPOSE 8000

# Single worker: in-process FAISS + embedding cache; scale horizontally only if you isolate state
CMD ["uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
