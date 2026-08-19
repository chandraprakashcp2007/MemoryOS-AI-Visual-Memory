"""Cloud API: Postgres metadata + private object storage + server-owned identity.

Enabled only with MEMORYOS_CLOUD_MODE=true. This intentionally has the same
public paths as the legacy API, so the Vite client needs no data fallback.
"""
from __future__ import annotations
import base64, hashlib, hmac, json, secrets, uuid
from datetime import datetime, timedelta, timezone
from io import BytesIO
from typing import Any

from fastapi import APIRouter, BackgroundTasks, Depends, File, Header, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image, ImageOps, UnidentifiedImageError
from sqlalchemy import func, or_, select, text
from sqlalchemy.orm import Session

from backend.cloud_models import CloudJob, CloudMemory, CloudUser, CloudWorkerHeartbeat
from backend.config import settings
from backend.database import get_db
from backend.services.cloud_storage import CloudStorageError, SupabaseStorage

router = APIRouter(tags=["Cloud memories"])
MAX_BYTES, MAX_PIXELS = 25 * 1024 * 1024, 40_000_000
ALLOWED = {"image/jpeg", "image/png", "image/webp", "image/gif", "image/bmp"}

@router.get("/health")
def health():
    return {"status": "healthy", "service": "MemoryOS cloud API"}

@router.get("/health/ready")
def ready(db: Session = Depends(get_db)):
    checks: dict[str, dict[str, str]] = {}
    try:
        db.execute(text("SELECT 1")); checks["database"] = {"status": "healthy"}
    except Exception:
        checks["database"] = {"status": "unavailable"}
    try:
        checks["storage"] = {"status": "healthy" if SupabaseStorage().check() else "unavailable"}
    except Exception:
        checks["storage"] = {"status": "unavailable"}
    try:
        enabled = db.scalar(text("SELECT EXISTS (SELECT 1 FROM pg_extension WHERE extname='vector')"))
        checks["vector"] = {"status": "healthy" if enabled else "unavailable"}
    except Exception:
        checks["vector"] = {"status": "unavailable"}
    checks["vision"] = {"status": "healthy" if settings.gemini_available else "degraded"}
    try:
        heartbeat = db.get(CloudWorkerHeartbeat, "primary")
        fresh = heartbeat and heartbeat.updated_at >= datetime.now(timezone.utc) - timedelta(seconds=60)
        checks["worker"] = {"status": "healthy" if fresh else "unavailable"}
    except Exception:
        checks["worker"] = {"status": "unavailable"}
    # Gemini may be degraded without making durable cloud data unavailable.
    # The UI can then show truthful processing degradation instead of treating
    # all existing memories as offline.
    ready_state = all(checks[name]["status"] == "healthy" for name in ("database", "storage", "vector", "worker"))
    return {"ready": ready_state, "status": "healthy" if ready_state else "degraded", "checks": checks}

def _b64(raw: bytes) -> str: return base64.urlsafe_b64encode(raw).rstrip(b"=").decode()
def _unb64(value: str) -> bytes: return base64.urlsafe_b64decode(value + "=" * (-len(value) % 4))
def _password(password: str, salt: bytes | None = None) -> str:
    salt = salt or secrets.token_bytes(16)
    digest = hashlib.pbkdf2_hmac("sha256", password.encode(), salt, 310_000)
    return f"{_b64(salt)}${_b64(digest)}"
def _verify_password(password: str, stored: str) -> bool:
    try:
        salt, digest = stored.split("$", 1)
        return hmac.compare_digest(_password(password, _unb64(salt)).split("$", 1)[1], digest)
    except ValueError: return False
def _token(user_id: str) -> str:
    payload = _b64(json.dumps({"sub": user_id, "exp": int((datetime.now(timezone.utc) + timedelta(hours=settings.auth_token_ttl_hours)).timestamp())}, separators=(",", ":")).encode())
    sig = _b64(hmac.new(settings.auth_secret.encode(), payload.encode(), hashlib.sha256).digest())
    return f"v1.{payload}.{sig}"
def current_user(authorization: str | None = Header(default=None), db: Session = Depends(get_db)) -> CloudUser:
    if not authorization or not authorization.startswith("Bearer "): raise HTTPException(401, "Sign in is required.")
    try:
        _, payload, signature = authorization.split(".")
        expected = _b64(hmac.new(settings.auth_secret.encode(), payload.encode(), hashlib.sha256).digest())
        claims = json.loads(_unb64(payload))
        if not hmac.compare_digest(signature, expected) or claims["exp"] < int(datetime.now(timezone.utc).timestamp()): raise ValueError
        user = db.get(CloudUser, claims["sub"])
        if not user: raise ValueError
        return user
    except Exception as exc: raise HTTPException(401, "Your session is invalid or has expired.") from exc
def _safe_name(name: str | None) -> str:
    return "".join(ch if ch.isalnum() or ch in ".-_" else "_" for ch in (name or "image"))[:180]
def _thumb(data: bytes) -> bytes:
    with Image.open(BytesIO(data)) as image:
        image = ImageOps.exif_transpose(image).convert("RGB"); image.thumbnail((720, 720))
        output = BytesIO(); image.save(output, "WEBP", quality=82, method=4); return output.getvalue()
def _validate(data: bytes, content_type: str) -> tuple[int, int, str]:
    if not data or len(data) > MAX_BYTES or content_type not in ALLOWED: raise HTTPException(400, "Upload must be a JPEG, PNG, WebP, GIF, or BMP image up to 25 MB.")
    try:
        with Image.open(BytesIO(data)) as image:
            image.verify()
        with Image.open(BytesIO(data)) as image:
            if image.width * image.height > MAX_PIXELS: raise HTTPException(400, "Image has too many pixels.")
            return image.width, image.height, image.format or "IMAGE"
    except (UnidentifiedImageError, OSError) as exc: raise HTTPException(400, "The uploaded file is not a valid image.") from exc
def _view(memory: CloudMemory) -> dict[str, Any]:
    vision = memory.visual_analysis or {}
    return {"memory_id": memory.memory_id, "id": memory.memory_id, "filename": memory.filename, "category": vision.get("category", "other"), "summary": vision.get("summary", ""), "visual_description": vision.get("visual_description", ""), "visual_concepts": vision.get("visual_concepts", []), "objects": vision.get("objects", []), "food_concepts": vision.get("food_concepts", []), "keywords": vision.get("keywords", []), "ocr_text": memory.ocr_text, "ocr_confidence": memory.ocr_confidence, "processing_status": memory.processing_status, "processing_error": memory.processing_error, "ocr_status": memory.ocr_status, "visual_status": memory.visual_status, "embedding_status": memory.embedding_status, "created_at": memory.created_at.isoformat(), "updated_at": memory.updated_at.isoformat(), "thumbnail_url": f"/upload/file/{memory.memory_id}?thumbnail=true", "original_image_url": f"/upload/file/{memory.memory_id}"}

def _vector_literal(vector: list[float]) -> str:
    return "[" + ",".join(f"{float(value):.8g}" for value in vector) + "]"

def _embed(text_value: str) -> list[float]:
    from backend.services.semantic_embedding_service import get_embedding_service
    return get_embedding_service().embed_text(text_value).astype(float).tolist()

def _upsert_vector(db: Session, memory: CloudMemory, vector: list[float]) -> None:
    """Use pgvector as the production retrieval index; JSON remains a recovery copy."""
    if settings.vector_backend != "pgvector":
        return
    db.execute(text("""INSERT INTO memory_embeddings (memory_id, user_id, embedding, embedding_model, updated_at)
        VALUES (:memory_id, :user_id, CAST(:embedding AS vector), :model, NOW())
        ON CONFLICT (memory_id) DO UPDATE SET embedding=EXCLUDED.embedding, embedding_model=EXCLUDED.embedding_model, updated_at=NOW()"""),
        {"memory_id": memory.memory_id, "user_id": memory.user_id, "embedding": _vector_literal(vector), "model": settings.vector_model})

@router.post("/auth/sign-up")
def sign_up(payload: dict[str, str], db: Session = Depends(get_db)):
    email, password = payload.get("email", "").strip().lower(), payload.get("password", "")
    if "@" not in email or len(password) < 10: raise HTTPException(400, "Use a valid email and a password of at least 10 characters.")
    if db.scalar(select(CloudUser).where(CloudUser.email == email)): raise HTTPException(409, "An account already exists for this email.")
    user = CloudUser(id=str(uuid.uuid4()), email=email, password_hash=_password(password)); db.add(user); db.commit()
    return {"access_token": _token(user.id), "user": {"email": user.email}}
@router.post("/auth/sign-in")
def sign_in(payload: dict[str, str], db: Session = Depends(get_db)):
    user = db.scalar(select(CloudUser).where(CloudUser.email == payload.get("email", "").strip().lower()))
    if not user or not _verify_password(payload.get("password", ""), user.password_hash): raise HTTPException(401, "Incorrect email or password.")
    return {"access_token": _token(user.id), "user": {"email": user.email}}
@router.get("/auth/session")
def session(user: CloudUser = Depends(current_user)): return {"user": {"email": user.email}}
@router.post("/auth/sign-out")
def sign_out(user: CloudUser = Depends(current_user)):
    # Tokens are short-lived, signed bearer sessions. The browser removes its
    # credential; deployments needing immediate server-side revocation can add
    # a shared token denylist without changing memory ownership semantics.
    return {"success": True}

def process_job(memory_id: str) -> None:
    """Idempotent worker body. It reads the durable original and updates DB state."""
    from backend.database import SessionLocal
    db = SessionLocal()
    try:
        memory, job = db.get(CloudMemory, memory_id), db.scalar(select(CloudJob).where(CloudJob.memory_id == memory_id))
        if not memory or not job or job.status == "completed": return
        # This lock makes duplicate delivery safe when a web task and worker
        # race for the same durable job.
        claim = select(CloudJob).where(CloudJob.id == job.id)
        # SQLite is useful for local tests, but does not support SKIP LOCKED.
        # Production PostgreSQL uses it so concurrent workers cannot process a
        # durable job twice.
        if db.bind and db.bind.dialect.name == "postgresql":
            claim = claim.with_for_update(skip_locked=True)
        else:
            claim = claim.with_for_update()
        job = db.scalar(claim)
        if not job or job.status in {"processing", "completed"}: return
        job.status, job.attempts, memory.processing_status = "processing", job.attempts + 1, "processing"; db.commit()
        data, _ = SupabaseStorage().get(memory.storage_key)
        ocr_text, confidence, ocr_state, stage_error = "", None, "unavailable", None
        try:
            from backend.services.ocr_service import extract_text
            result = extract_text(data); ocr_text, confidence = result.text if result.success else "", result.confidence if result.success else None; ocr_state = "complete" if result.available else "unavailable"
        except Exception:
            stage_error = "OCR is temporarily unavailable."
        vision, visual_state = {}, "unavailable"
        try:
            from backend.services.vision_service import analyze_image_with_ocr
            result = analyze_image_with_ocr(data, ocr_text, allow_fallback=False); vision = result.to_dict(); visual_state = "complete" if result.success else "failed"
        except Exception:
            stage_error = "Visual analysis is temporarily unavailable."
        searchable = " ".join(str(x) for x in [memory.filename, ocr_text, vision.get("summary", ""), vision.get("visual_description", ""), *vision.get("keywords", []), *vision.get("visual_concepts", [])])
        memory.ocr_text, memory.ocr_confidence, memory.ocr_status = ocr_text, confidence, ocr_state
        memory.processing_status, job.status = "ocr_complete", "ocr_complete"; db.commit()
        memory.visual_analysis, memory.visual_status, memory.search_text = vision, visual_state, searchable
        memory.processing_error = stage_error
        memory.processing_status, job.status = "vision_complete", "vision_complete"; db.commit()
        try:
            vector = _embed(searchable or memory.filename)
            memory.embedding = vector
            _upsert_vector(db, memory, vector)
            memory.embedding_status = "complete"
            memory.processing_status, job.status = "indexed", "indexed"; db.commit()
        except Exception:
            # Keep OCR/vision data searchable, but retain a durable retry for
            # the missing semantic index instead of claiming full readiness.
            from datetime import timedelta
            memory.embedding_status, memory.processing_status = "failed", "partial"
            memory.processing_error = "Semantic indexing is temporarily unavailable."
            if job.attempts < settings.worker_max_attempts:
                job.status = "retrying"
                job.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=min(3600, 2 ** max(1, job.attempts)))
            else:
                job.status = "failed"
            db.commit()
            return
        memory.processing_status, job.status, job.last_error = "ready", "completed", None; db.commit()
    except Exception as exc:
        if 'job' in locals() and job:
            from datetime import timedelta
            retry = job.attempts < settings.worker_max_attempts
            job.status, job.last_error = ("retrying" if retry else "failed"), "processing failed"
            job.next_attempt_at = datetime.now(timezone.utc) + timedelta(seconds=min(3600, 2 ** max(1, job.attempts))) if retry else None
            memory.processing_status = job.status
            memory.processing_error = "Processing failed and will be retried." if retry else "Processing failed after all retry attempts."
            db.commit()
    finally: db.close()

@router.post("/upload")
async def upload(file: UploadFile = File(...), background: BackgroundTasks = None, user: CloudUser = Depends(current_user), db: Session = Depends(get_db)):
    data = await file.read(); width, height, _ = _validate(data, (file.content_type or "").lower()); digest = hashlib.sha256(data).hexdigest()
    existing = db.scalar(select(CloudMemory).where(CloudMemory.user_id == user.id, CloudMemory.content_hash == digest))
    if existing: return {"success": True, "duplicate": True, **_view(existing)}
    memory_id, ext = str(uuid.uuid4()), {"image/jpeg":"jpg", "image/png":"png", "image/webp":"webp", "image/gif":"gif", "image/bmp":"bmp"}[file.content_type.lower()]
    key = f"{user.id}/memories/{memory_id}/original.{ext}"; thumbnail_key = f"{user.id}/memories/{memory_id}/thumbnail.webp"
    try: storage = SupabaseStorage(); storage.put(key, data, file.content_type.lower()); storage.put(thumbnail_key, _thumb(data), "image/webp")
    except CloudStorageError as exc: raise HTTPException(503, "Image storage is unavailable. Nothing was saved.") from exc
    memory = CloudMemory(memory_id=memory_id, user_id=user.id, filename=_safe_name(file.filename), mime_type=file.content_type.lower(), size_bytes=len(data), width=width, height=height, content_hash=digest, storage_key=key, thumbnail_key=thumbnail_key, processing_status="queued")
    job = CloudJob(id=str(uuid.uuid4()), memory_id=memory_id); db.add_all([memory, job]); db.commit(); background.add_task(process_job, memory_id)
    return {"success": True, "duplicate": False, **_view(memory)}
@router.post("/upload/batch")
async def upload_batch(files: list[UploadFile] = File(...), background: BackgroundTasks = None, user: CloudUser = Depends(current_user), db: Session = Depends(get_db)):
    if len(files) > 25: raise HTTPException(400, "A batch may contain at most 25 images.")
    results=[]
    for file in files:
        try: results.append(await upload(file, background, user, db))
        except HTTPException as exc: results.append({"success":False,"filename":_safe_name(file.filename),"message":str(exc.detail)})
    return {"success": all(r["success"] for r in results), "results": results, "successful":sum(r["success"] for r in results), "failed":sum(not r["success"] for r in results)}
@router.get("/upload/file/{memory_id}")
def image(memory_id: str, thumbnail: bool = False, preview: bool = False, user: CloudUser = Depends(current_user), db: Session = Depends(get_db)):
    memory = db.scalar(select(CloudMemory).where(CloudMemory.memory_id == memory_id, CloudMemory.user_id == user.id))
    if not memory: raise HTTPException(404, "Memory not found.")
    try: data, mime = SupabaseStorage().get(memory.thumbnail_key if thumbnail or preview else memory.storage_key)
    except CloudStorageError as exc: raise HTTPException(503, "Image is temporarily unavailable.") from exc
    return Response(data, media_type=mime, headers={"Cache-Control":"private, max-age=300", "X-Content-Type-Options":"nosniff"})
@router.get("/memories")
def memories(page:int=1, limit:int=50, category:str|None=None, user:CloudUser=Depends(current_user), db:Session=Depends(get_db)):
    query=select(CloudMemory).where(CloudMemory.user_id==user.id)
    # JSON category filtering is deliberately performed after the owner
    # predicate; category is derived from model output, never client input.
    if category:
        query = query.where(CloudMemory.visual_analysis["category"].as_string() == category)
    safe_limit = max(1, min(limit, 100)); page = max(1, page)
    rows=db.scalars(query.order_by(CloudMemory.created_at.desc()).offset((page-1)*safe_limit).limit(safe_limit)).all()
    total=db.scalar(select(func.count()).select_from(query.subquery())) or 0
    return {"success":True,"memories":[_view(x) for x in rows],"total":total,"page":page,"has_next":page*safe_limit<total}

@router.get("/memories/{memory_id}")
def memory_detail(memory_id: str, user: CloudUser = Depends(current_user), db: Session = Depends(get_db)):
    memory = db.scalar(select(CloudMemory).where(CloudMemory.memory_id == memory_id, CloudMemory.user_id == user.id))
    if not memory:
        raise HTTPException(404, "Memory not found.")
    return {"success": True, "memory": _view(memory)}

@router.delete("/memories/{memory_id}")
def delete_memory(memory_id: str, user: CloudUser = Depends(current_user), db: Session = Depends(get_db)):
    memory = db.scalar(select(CloudMemory).where(CloudMemory.memory_id == memory_id, CloudMemory.user_id == user.id))
    if not memory:
        raise HTTPException(404, "Memory not found.")
    try:
        storage = SupabaseStorage()
        storage.delete(memory.storage_key)
        if memory.thumbnail_key:
            storage.delete(memory.thumbnail_key)
    except CloudStorageError as exc:
        raise HTTPException(503, "Image storage is temporarily unavailable; memory was not deleted.") from exc
    db.delete(memory)
    db.commit()
    return {"success": True}
@router.post("/search")
def search(payload:dict[str,Any], user:CloudUser=Depends(current_user), db:Session=Depends(get_db)):
    term=payload.get("query","").strip(); limit=min(int(payload.get("limit",20)),100)
    if not term: return {"success":True,"results":[]}
    rows=db.scalars(select(CloudMemory).where(CloudMemory.user_id==user.id, CloudMemory.processing_status.in_(("ready", "partial")), or_(CloudMemory.search_text.ilike(f"%{term}%"),CloudMemory.filename.ilike(f"%{term}%"))).order_by(CloudMemory.created_at.desc()).limit(limit)).all()
    semantic_scores: dict[str, float] = {}
    if settings.vector_backend == "pgvector":
        try:
            vector = _vector_literal(_embed(term))
            semantic_scores = {row[0]: float(row[1]) for row in db.execute(text("""SELECT memory_id, embedding <=> CAST(:embedding AS vector) AS distance FROM memory_embeddings
                WHERE user_id=:user_id ORDER BY embedding <=> CAST(:embedding AS vector) LIMIT :limit"""), {"user_id":user.id,"embedding":vector,"limit":limit}).all()}
        except Exception:
            # Keyword/concept evidence remains available during a transient
            # embedding or vector-index outage; do not pretend it was semantic.
            semantic_scores = {}
    by_id = {row.memory_id: row for row in rows}
    if semantic_scores:
        for row in db.scalars(select(CloudMemory).where(CloudMemory.user_id == user.id, CloudMemory.memory_id.in_(semantic_scores))).all(): by_id[row.memory_id] = row
    ordered = sorted(by_id.values(), key=lambda row: (semantic_scores.get(row.memory_id, 2.0), -row.created_at.timestamp()))
    results=[]
    for row in ordered[:limit]:
        item=_view(row); searchable=row.search_text.lower(); evidence=[]
        if row.memory_id in semantic_scores: evidence.append("Semantic match")
        if term.lower() in row.ocr_text.lower(): evidence.append("OCR match")
        concepts = " ".join(str(x) for x in (row.visual_analysis or {}).get("visual_concepts", []))
        if term.lower() in concepts.lower(): evidence.append("Concept match")
        if term.lower() in row.filename.lower(): evidence.append("Filename match")
        item["why_matched"] = evidence
        # This is a real cosine distance returned by pgvector, not a made-up
        # relevance percentage. It is omitted for lexical-only candidates.
        if row.memory_id in semantic_scores: item["semantic_distance"] = semantic_scores[row.memory_id]
        results.append(item)
    return {"success":True,"results":results}

@router.get("/search/related/{memory_id}")
def related_memories(memory_id: str, limit: int = 4, user: CloudUser = Depends(current_user), db: Session = Depends(get_db)):
    """Return real, owner-scoped pgvector neighbours for the image viewer."""
    memory = db.scalar(select(CloudMemory).where(
        CloudMemory.memory_id == memory_id,
        CloudMemory.user_id == user.id,
    ))
    if not memory:
        raise HTTPException(404, "Memory not found.")
    if settings.vector_backend != "pgvector" or not memory.embedding:
        return {"success": True, "results": []}
    try:
        candidate_ids = [row[0] for row in db.execute(text("""SELECT memory_id
            FROM memory_embeddings
            WHERE user_id = :user_id AND memory_id != :memory_id
            ORDER BY embedding <=> CAST(:embedding AS vector)
            LIMIT :limit"""), {
                "user_id": user.id,
                "memory_id": memory_id,
                "embedding": _vector_literal(memory.embedding),
                "limit": max(1, min(limit, 20)),
            }).all()]
    except Exception:
        return {"success": True, "results": []}
    candidates = {row.memory_id: row for row in db.scalars(select(CloudMemory).where(
        CloudMemory.user_id == user.id,
        CloudMemory.memory_id.in_(candidate_ids),
    )).all()}
    return {"success": True, "results": [{**_view(candidates[row_id]), "why_matched": ["Semantic match"]}
        for row_id in candidate_ids if row_id in candidates]}

@router.get("/search/suggestions")
def search_suggestions(q: str, limit: int = 6, user: CloudUser = Depends(current_user), db: Session = Depends(get_db)):
    """Return only owner-scoped filename/concept suggestions for the UI."""
    term = q.strip()
    if len(term) < 2:
        return {"suggestions": []}
    rows = db.scalars(select(CloudMemory).where(
        CloudMemory.user_id == user.id,
        CloudMemory.search_text.ilike(f"%{term}%"),
    ).order_by(CloudMemory.created_at.desc()).limit(max(1, min(limit, 20)))).all()
    values: list[str] = []
    for row in rows:
        for value in [row.filename, *((row.visual_analysis or {}).get("visual_concepts", []))]:
            value = str(value).strip()
            if value and value.lower() not in {item.lower() for item in values}:
                values.append(value)
            if len(values) >= limit:
                return {"suggestions": values}
    return {"suggestions": values}
@router.get("/stats")
def stats(user:CloudUser=Depends(current_user), db:Session=Depends(get_db)):
    base=select(CloudMemory).where(CloudMemory.user_id==user.id); total=db.scalar(select(func.count()).select_from(base.subquery())) or 0
    ready=db.scalar(select(func.count()).select_from(CloudMemory).where(CloudMemory.user_id==user.id,CloudMemory.processing_status=="ready")) or 0
    return {"success":True,"memories":{"total":total,"processed":ready,"processing":total-ready}}
@router.post("/processing/retry/{memory_id}")
def retry(memory_id:str, background:BackgroundTasks, user:CloudUser=Depends(current_user), db:Session=Depends(get_db)):
    memory=db.scalar(select(CloudMemory).where(CloudMemory.memory_id==memory_id,CloudMemory.user_id==user.id)); job=db.scalar(select(CloudJob).where(CloudJob.memory_id==memory_id))
    if not memory or not job: raise HTTPException(404,"Memory not found.")
    job.status="queued"; memory.processing_status="uploaded"; db.commit(); background.add_task(process_job,memory_id); return {"success":True}
