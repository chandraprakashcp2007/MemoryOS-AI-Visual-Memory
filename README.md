# MemoryOS

MemoryOS is a visual memory engine: it combines Gemini multimodal analysis,
local OCR, semantic embeddings, FAISS retrieval, multimodal ranking, and
feedback-driven retrieval. Gemini is used for analysis; it is not trained by
this application.

## Local development

Create a virtual environment, install the project requirements, and set the
server-side `GEMINI_API_KEY` in `.env` (never in the frontend):

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
cd frontend
npm run dev
```

The frontend runs on `http://localhost:5173`; the API runs on port 8000.
Use `GET /health` for liveness and `GET /health/ready` for dependency status.

## Production and HTTPS

Run the API behind an HTTPS reverse proxy (for example Nginx or Caddy), serve
the built frontend over HTTPS, and configure `MEMORYOS_CORS_ORIGINS` to the
single public frontend origin. Do not enable HTTPS directly for localhost or
commit certificates/keys. Enable the in-process safeguard with
`MEMORYOS_RATE_LIMIT_ENABLED=true`; use gateway rate limiting for multi-worker
deployments. Set `MEMORYOS_ENV=production` to enable rate limiting by default.

Build the frontend with `npm run build`. Run the API with a production ASGI
server configuration appropriate for the host. Ensure the proxy forwards the
original HTTPS scheme and does not create mixed-content API URLs.

## Data, limits, and maintenance

Uploads accept JPEG, PNG, and WEBP up to 10 MB, with safe dimension/pixel
limits. Originals and thumbnails live under `data/`; memory metadata is in
`data/memory_index`, vectors in `data/faiss`, and relevance feedback in
`data/search_feedback.json`.

Backfill existing memories without changing IDs or original files:

```powershell
.\.venv\Scripts\python.exe scripts\backfill_visual_index.py
```

Gemini failures are recorded as degraded and can be retried by running
backfill again. OCR and basic search remain available when visual analysis is
unavailable.

## Backup and restore

Stop writes or take a filesystem-consistent snapshot, then back up `data/`
(originals, thumbnails, metadata, index, FAISS data, and feedback) plus the
non-secret configuration. Keep `.env` and certificates in a separate secret
manager/secure backup. Restore the same directories before starting the API;
do not delete originals or rewrite memory IDs during restore.
