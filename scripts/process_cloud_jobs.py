"""Run queued/retryable cloud processing jobs from durable PostgreSQL state."""
from __future__ import annotations
import signal
import time
from datetime import datetime, timedelta, timezone
from sqlalchemy import or_, select, update
from backend.cloud_models import CloudJob, CloudWorkerHeartbeat
from backend.database import SessionLocal
from backend.api.cloud import process_job

running = True
def stop(*_args):
    global running
    running = False
signal.signal(signal.SIGINT, stop)
signal.signal(signal.SIGTERM, stop)

while running:
    db = SessionLocal()
    try:
        heartbeat = db.get(CloudWorkerHeartbeat, "primary")
        if heartbeat is None:
            db.add(CloudWorkerHeartbeat(id="primary"))
        else:
            heartbeat.updated_at = datetime.now(timezone.utc)
        db.commit()
        # A process can be terminated after claiming a job. Re-queue only
        # genuinely stale work; active work is never discarded on restart.
        db.execute(update(CloudJob).where(
            CloudJob.status == "processing",
            CloudJob.updated_at < datetime.now(timezone.utc) - timedelta(minutes=15),
        ).values(status="retrying", next_attempt_at=datetime.now(timezone.utc), last_error="worker lease expired"))
        db.commit()
        job_ids = db.scalars(select(CloudJob.memory_id).where(
            or_(CloudJob.status == "queued", (CloudJob.status == "retrying") & (CloudJob.next_attempt_at <= datetime.now(timezone.utc)))
        ).order_by(CloudJob.created_at).limit(10)).all()
    finally:
        db.close()
    for memory_id in job_ids:
        process_job(memory_id)
    time.sleep(2 if job_ids else 10)
