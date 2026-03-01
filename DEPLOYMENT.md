# Deployment guide

Practical options for hosting **Knowledge Assistant** in a portfolio-friendly way. The same RAG stack powers **Streamlit** and **FastAPI**; pick one primary surface for a free-tier demo, or run both.

---

## 1. Streamlit only (simplest live demo)

Best when you want **one** deployable artifact and the original sidebar UX.

**Platform example: Render**

- **Build:** `pip install -r requirements.txt`
- **Start:** `streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0 --server.enableCORS=false`
- **Env:** `OPENAI_API_KEY`, `PYTHONUNBUFFERED=1`

This repo includes **`render.yaml`** as a starting point; adjust the start command to use Render’s `PORT` if you do not hard-code `10000`.

**Persistence:** Mount or use a **persistent disk** for `data/raw` and `data/indexes` (and optionally `data/chat_history.db` if you add shared DB paths later), or uploads re-index will reset on ephemeral disks.

---

## 2. FastAPI API (container or PaaS)

**Docker (recommended pattern)**

```bash
docker build -t ka-api .
docker run --rm -p 8000:8000 --env-file .env -v "$(pwd)/data:/app/data" ka-api
```

- Set `OPENAI_API_KEY` and, for public browsers, **`KA_CORS_ORIGINS`** (your Next.js site URL, comma-separated).
- Set **`KA_ENV=production`** to disable `/docs` and `/redoc` in public demos.
- Use **one uvicorn worker** unless you externalize FAISS + SQLite to shared storage (current design is single-node).

**PaaS (Render / Fly.io / Railway / similar)**

- **Build:** same as Dockerfile (`pip install …` or `docker build`).
- **Start:** `uvicorn backend.app.main:app --host 0.0.0.0 --port $PORT --workers 1`
- Attach a **volume** for `/app/data` if the platform supports it.

**Health checks:** `GET /health`

---

## 3. Next.js frontend (static + server)

The UI reads **`NEXT_PUBLIC_API_URL`** at **build time** (it is inlined into the client bundle).

**Vercel / Netlify (typical)**

1. Create a project from `web/`.
2. Set **environment variable** `NEXT_PUBLIC_API_URL` to your **public API origin** (e.g. `https://api.yourdomain.com`), no trailing slash.
3. Build command: `npm run build` (from `web/`). Output: Next default.

**Backend CORS:** set `KA_CORS_ORIGINS` on the API to your frontend origin (e.g. `https://your-app.vercel.app`).

---

## 4. Docker Compose (API + web together)

From the **repository root**:

```bash
cp .env.example .env
# Edit .env (at minimum OPENAI_API_KEY)

docker compose up --build
```

- **Web:** http://localhost:3000  
- **API:** http://localhost:8000  

`./data` on the host is mounted into the API container for uploads, FAISS, and SQLite chat DB.

To change the API URL baked into the web image, edit **`docker-compose.yml`** `args.NEXT_PUBLIC_API_URL` and rebuild.

---

## 5. Checklist before you share a public URL

- [ ] `OPENAI_API_KEY` set on the API host (never commit real keys).
- [ ] `KA_CORS_ORIGINS` matches the **exact** browser origin of the Next app (scheme + host + port).
- [ ] `NEXT_PUBLIC_API_URL` points to the **public** API URL users’ browsers can reach.
- [ ] `KA_ENV=production` on the API if you do not want public OpenAPI docs.
- [ ] Persistent disk (or acceptance of reset) for `data/` if demos should keep uploads across restarts.

---

## 6. Cost and limits

- **OpenAI:** embedding + chat usage per query and sync.
- **Free tiers:** cold starts and sleep (e.g. Render free) affect first request latency.
- **Portfolio:** a cold start is acceptable; document it in README or demo notes.

---

## References

- [Render docs](https://render.com/docs)
- [Streamlit deployment](https://docs.streamlit.io/deploy)
- [Next.js deployment](https://nextjs.org/docs/app/building-your-application/deploying)
- [FastAPI behind a proxy](https://fastapi.tiangolo.com/deployment/)
