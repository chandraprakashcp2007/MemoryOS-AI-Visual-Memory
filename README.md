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

## Production status: cloud migration required

This checkout includes a cloud route layer but is **not a deployed cloud
environment**. Local mode still uses `data/` (originals, thumbnails, JSON
metadata, SQLite and FAISS); cloud mode replaces those routes with private
object storage and user-scoped PostgreSQL/pgvector records. A Vercel visitor
must be configured to call the deployed API, never their own machine.

The frontend now reads `VITE_API_BASE_URL`; set it in Vercel to the HTTPS URL
of a separately deployed API. It deliberately has no production localhost
fallback. The API refuses `MEMORYOS_ENV=production` with SQLite or without an
object-store configuration, preventing ephemeral server files from being
misrepresented as persistent data.

To complete a secure production migration, create managed Postgres/pgvector,
private object storage and Auth (Supabase is a suitable single provider), then
deploy FastAPI to a persistent API host. Configure the backend with a managed
`DATABASE_URL`, `MEMORYOS_STORAGE_BACKEND=supabase`, `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET`, `GEMINI_API_KEY`, and
the exact `MEMORYOS_CORS_ORIGINS`. Set only `VITE_API_BASE_URL` in Vercel;
never expose Gemini or the Supabase service-role secret in `VITE_*`.

The cloud source of truth has stable `memory_id`, server-derived `user_id`,
content hash, original/thumbnail object keys, metadata/analysis, processing
state, and pgvector embedding. Every cloud upload, list, search, image and
retry query enforces the verified token's user ID. Object keys use
`user_id/memories/memory_id/original` and `.../thumbnail.webp`, returning an
authorized stream or short-lived signed URL—not a filesystem path.

## Data, limits, and maintenance

## Cloud mode deployment

`MEMORYOS_CLOUD_MODE=true` switches the API from legacy filesystem routes to
the cloud route layer. It uses PostgreSQL tables (`cloud_users`,
`cloud_memories`, `cloud_memory_jobs`) as durable metadata/job state and a
private Supabase bucket for exact originals and thumbnails. The API verifies a
server-signed session on every memory/search/image request and streams an
object only when its `user_id` matches the database owner.

Deploy FastAPI to a persistent backend host and configure
`DATABASE_URL`, `MEMORYOS_STORAGE_BACKEND=supabase`, `SUPABASE_URL`,
`SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET`,
`MEMORYOS_AUTH_SECRET`, `GEMINI_API_KEY`, and the exact
`MEMORYOS_CORS_ORIGINS`. In Vercel configure only `VITE_API_BASE_URL` and
`VITE_CLOUD_AUTH=true`. Never expose service-role, Gemini, database, or auth
secrets as `VITE_*` values. Run `python scripts/process_cloud_jobs.py` as the
persistent worker; it resumes queued jobs from PostgreSQL after a browser or
worker restart.

Run `python scripts/migrate_cloud.py` once against the production database to
create the cloud schema, enable pgvector, and create its HNSW cosine index.
Cloud search merges owner-scoped pgvector candidates with OCR, concepts, and
filenames. Legacy FAISS remains local-only for development.

Uploads accept JPEG, PNG, and WEBP up to 10 MB, with safe dimension/pixel
limits. In the current local mode originals and thumbnails live under `data/`;
memory metadata is in `data/memory_index`, vectors in `data/faiss`, and
relevance feedback in `data/search_feedback.json`. These are local development
state only and must never be the production source of truth.

Backfill existing memories without changing IDs or original files:

```powershell
.\.venv\Scripts\python.exe scripts\backfill_visual_index.py
```

Gemini failures are recorded as degraded and can be retried by running
backfill again. OCR and basic search remain available when visual analysis is
unavailable.

## Production configuration reference

### 1. Architecture

Cloud mode keeps the local architecture intact and uses PostgreSQL for user-scoped metadata/jobs, private Supabase Storage for originals and thumbnails, pgvector for cloud embeddings, and a separate worker. Local mode continues using SQLite/filesystem/FAISS.

### 2. Local development

```powershell
.\.venv\Scripts\python.exe -m uvicorn backend.main:app --host 127.0.0.1 --port 8000 --reload
cd frontend
npm run dev
```

### 3. Cloud development

Set `MEMORYOS_CLOUD_MODE=true` with the server variables below. Production startup refuses SQLite, local object persistence, wildcard CORS, and a missing auth secret.

### 4. Supabase setup

Provision PostgreSQL with the `vector` extension and a private bucket called `memoryos-private`. The service-role key belongs only in the backend secret manager.

### 5. pgvector setup

```powershell
.\.venv\Scripts\python.exe scripts\migrate_cloud.py
```

This idempotently creates cloud tables, the embedding table, owner index, and HNSW cosine index where supported. Do not report it as passed until it is run against PostgreSQL.

### 6. Storage bucket setup

Objects use `user_id/memories/memory_id/original.ext` and `thumbnail.webp`. The API checks the authenticated owner before streaming either object. The frontend retrieves private images via its Authorization header and uses an in-memory object URL, never a tokenized image URL.

### 7. Environment variables

Backend only: `DATABASE_URL`, `MEMORYOS_CLOUD_MODE=true`, `MEMORYOS_STORAGE_BACKEND=supabase`, `SUPABASE_URL`, `SUPABASE_SERVICE_ROLE_KEY`, `SUPABASE_STORAGE_BUCKET`, `GEMINI_API_KEY`, `MEMORYOS_AUTH_SECRET`, `MEMORYOS_VECTOR_BACKEND=pgvector`, `MEMORYOS_VECTOR_DIMENSION=384`, `MEMORYOS_WORKER_MAX_ATTEMPTS=5`, `MEMORYOS_CORS_ORIGINS=https://YOUR-VERCEL-APP.vercel.app`.

Frontend only: `VITE_API_BASE_URL=https://YOUR-API.example.com` and `VITE_CLOUD_AUTH=true`. Never put database, Gemini, auth, or service-role secrets in `VITE_*`.

### 8. Migration

Run the migration after environment setup. It does not delete local memories or change existing IDs.

### 9. Worker

```powershell
.\.venv\Scripts\python.exe scripts\process_cloud_jobs.py
```

Deploy this separately from the API. It resumes queued/retryable work, persists real stage state, retries with backoff, reclaims only stale work, and sends a durable heartbeat.

### 10. Backend deployment

Use a long-running Python host for `uvicorn backend.main:app --host 0.0.0.0 --port $PORT`, and a second service for the worker. Vercel cannot host this worker.

### 11. Vercel deployment

Deploy `frontend/` with `npm run build`. Configure its HTTPS API URL and cloud-auth flag separately for development and production.

### 12. Mobile testing

Import Photos uses the browser/OS file chooser with multiple selection and camera where supported. Web apps cannot silently read a phone gallery; dismissed access leaves manual selection available.

### 13. Cross-device testing

Sign into one account on two real devices, upload on A, wait for ready, refresh/search/open it on B, then reverse the upload. This checkout has not performed that real-device test.

### 14. Troubleshooting

`/health` is liveness. `/health/ready` independently reports database, storage, vector, AI configuration, and worker heartbeat. A degraded dependency is not healthy.

### 15. Security

Private storage, owner checks, server-only secrets, upload MIME/size/pixel validation, CORS, rate limiting, security headers, and safe errors are required in production.

### 16. Backup

Configure and test managed Postgres backups plus Supabase Storage export/backup. Restore metadata and object keys together. This repository does not configure backups.

### 17. Production checklist

- [ ] Migration, private storage, API, and worker verified
- [ ] Gemini, OCR, embeddings, pgvector, and health heartbeat verified
- [ ] Two-user isolation and exact-image identity tested
- [ ] Real-device cross-device and mobile import tests completed
- [ ] Search acceptance matrix and backup restore recorded

## Backup and restore

Stop writes or take a filesystem-consistent snapshot, then back up `data/`
(originals, thumbnails, metadata, index, FAISS data, and feedback) plus the
non-secret configuration. Keep `.env` and certificates in a separate secret
manager/secure backup. Restore the same directories before starting the API;
do not delete originals or rewrite memory IDs during restore.
