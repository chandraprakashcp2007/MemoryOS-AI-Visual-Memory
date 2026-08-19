"""Private Supabase Storage adapter; no object URL or service key reaches clients."""
from __future__ import annotations

import httpx

from backend.config import settings


class CloudStorageError(RuntimeError):
    pass


class SupabaseStorage:
    def __init__(self) -> None:
        if not settings.supabase_url or not settings.supabase_service_role_key:
            raise CloudStorageError("Object storage is not configured.")
        self.base = settings.supabase_url.rstrip("/")
        self.bucket = settings.supabase_storage_bucket
        self.headers = {"Authorization": f"Bearer {settings.supabase_service_role_key}", "apikey": settings.supabase_service_role_key}

    def put(self, key: str, payload: bytes, mime_type: str) -> None:
        response = httpx.post(f"{self.base}/storage/v1/object/{self.bucket}/{key}", content=payload,
                              headers={**self.headers, "Content-Type": mime_type, "x-upsert": "false"}, timeout=60)
        if response.status_code not in (200, 201):
            raise CloudStorageError("Could not store image.")

    def get(self, key: str) -> tuple[bytes, str]:
        response = httpx.get(f"{self.base}/storage/v1/object/{self.bucket}/{key}", headers=self.headers, timeout=60)
        if response.status_code != 200:
            raise CloudStorageError("Could not retrieve image.")
        return response.content, response.headers.get("content-type", "application/octet-stream")

    def delete(self, key: str) -> None:
        response = httpx.delete(f"{self.base}/storage/v1/object/{self.bucket}/{key}", headers=self.headers, timeout=30)
        if response.status_code not in (200, 204, 404):
            raise CloudStorageError("Could not delete image.")

    def check(self) -> bool:
        """Verify the configured private bucket without exposing its contents."""
        response = httpx.get(f"{self.base}/storage/v1/bucket/{self.bucket}", headers=self.headers, timeout=10)
        return response.status_code == 200
