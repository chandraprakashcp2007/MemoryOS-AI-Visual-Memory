"""
MemoryOS fast progressive gallery ingestion.

Visual index becomes available first.
OCR / Gemini / MiniLM enrichment continues after the visual batch.
"""

from __future__ import annotations

import json
import io
import logging
import os

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from threading import RLock
from typing import Any

from PIL import Image

from fastapi import (
    APIRouter,
    BackgroundTasks,
    File,
    Form,
    HTTPException,
    UploadFile,
)

from backend.api.upload import (
    METADATA_DIR,
    SUPPORTED_EXTENSIONS,
    UploadServiceError,
    calculate_sha256,
    clean_string,
    create_thumbnail,
    find_duplicate_by_hash,
    generate_memory_id,
    load_metadata,
    persist_search_memory,
    process_uploaded_memory,
    save_metadata,
    safe_filename,
    store_original,
    thumbnail_file,
    utc_now_iso,
    validate_image_bytes,
)

from backend.services.image_embedding_service import (
    get_image_embedding_service,
)

from backend.services.visual_vector_store import (
    get_visual_vector_store,
)


logger = logging.getLogger(
    "memoryos.gallery"
)

router = APIRouter(
    prefix="/gallery",
    tags=["Gallery"],
)


FAST_BATCH_LIMIT = max(
    1,
    min(
        int(
            os.getenv(
                "MEMORYOS_GALLERY_BATCH_LIMIT",
                "24",
            )
        ),
        48,
    ),
)


ENRICH_WORKERS = max(
    1,
    min(
        int(
            os.getenv(
                "MEMORYOS_GALLERY_ENRICH_WORKERS",
                "2",
            )
        ),
        6,
    ),
)


_PROGRESS_LOCK = RLock()

_PROGRESS = {
    "phase": "idle",
    "discovered": 0,
    "visual_ready": 0,
    "duplicates": 0,
    "visual_failed": 0,
    "enrichment_total": 0,
    "enriched": 0,
    "enrichment_failed": 0,
    "active": False,
    "updated_at": None,
}


_HASH_CACHE_LOCK = RLock()
_HASH_CACHE = None


def _hash_cache():

    global _HASH_CACHE

    with _HASH_CACHE_LOCK:

        if _HASH_CACHE is not None:
            return _HASH_CACHE

        cache = {}

        for path in METADATA_DIR.glob("*.json"):

            try:

                data = json.loads(
                    path.read_text(
                        encoding="utf-8"
                    )
                )

            except Exception:
                continue

            if not isinstance(data, dict):
                continue

            image_hash = clean_string(
                data.get("image_hash")
            ).lower()

            if image_hash:
                cache[image_hash] = data

        _HASH_CACHE = cache

        return _HASH_CACHE


def _find_duplicate_fast(
    image_hash: str,
):

    return _hash_cache().get(
        str(image_hash).lower()
    )


def _register_hash(
    image_hash: str,
    metadata: dict,
):

    with _HASH_CACHE_LOCK:

        _hash_cache()[
            str(image_hash).lower()
        ] = dict(metadata)



def update_progress(
    **values,
):

    with _PROGRESS_LOCK:

        _PROGRESS.update(
            values
        )

        _PROGRESS[
            "updated_at"
        ] = utc_now_iso()

        return dict(
            _PROGRESS
        )


def increment_progress(
    **values,
):

    with _PROGRESS_LOCK:

        for (
            key,
            value,
        ) in values.items():

            _PROGRESS[key] = (
                int(
                    _PROGRESS.get(
                        key,
                        0,
                    )
                )
                + int(value)
            )

        _PROGRESS[
            "updated_at"
        ] = utc_now_iso()

        return dict(
            _PROGRESS
        )


def captured_at(
    metadata,
    image_bytes=None,
):

    # Prefer EXIF DateTimeOriginal where available.
    if image_bytes:

        try:

            with Image.open(
                io.BytesIO(image_bytes)
            ) as image:

                exif = image.getexif()

                raw = (
                    exif.get(36867)
                    or exif.get(36868)
                    or exif.get(306)
                )

                if raw:

                    parsed = datetime.strptime(
                        str(raw),
                        "%Y:%m:%d %H:%M:%S",
                    )

                    return parsed.isoformat()

        except Exception:
            pass

    # Browser File.lastModified fallback.
    try:

        milliseconds = float(
            metadata.get(
                "last_modified_ms"
            )
        )

        if milliseconds <= 0:
            return None

        # Server local time is intentional for local judge demos,
        # so "yesterday" aligns with the local machine date.
        return datetime.fromtimestamp(
            milliseconds / 1000
        ).isoformat()

    except Exception:
        return None


def minimal_processing():

    return {
        "success": True,
        "processed": False,
        "status": "visual_ready",
        "ocr_text": "",
        "description": "",
        "category": "other",
        "keywords": [],
        "entities": [],
        "vision": {},
        "visual_indexed": True,
    }


def enrich_one(
    memory_id: str,
):

    metadata = load_metadata(
        memory_id
    )

    if not metadata:
        return False

    source = Path(
        str(
            metadata.get(
                "original_path"
            )
            or ""
        )
    )

    if not source.is_file():
        return False

    try:

        processing = (
            process_uploaded_memory(
                source=source,
                memory_id=memory_id,
                metadata=metadata,
            )
        )

        processed = bool(
            processing.get(
                "processed",
                False,
            )
        )

        if processed:

            persist_search_memory(
                memory_id=memory_id,
                metadata=metadata,
                processing=processing,
            )

        metadata.update(
            {
                "processed":
                    processed,

                "status":
                    clean_string(
                        processing.get(
                            "status"
                        )
                    )
                    or (
                        "processed"
                        if processed
                        else "visual_ready"
                    ),

                "processing_result":
                    processing,

                "visual_indexed":
                    True,

                "updated_at":
                    utc_now_iso(),
            }
        )

        save_metadata(
            memory_id,
            metadata,
        )

        return processed

    except Exception as exc:

        logger.warning(
            "Background enrichment failed %s: %s",
            memory_id,
            exc,
        )

        return False


def enrich_pending():

    pending = []

    for path in (
        METADATA_DIR.glob(
            "*.json"
        )
    ):

        try:

            data = json.loads(
                path.read_text(
                    encoding="utf-8"
                )
            )

        except Exception:
            continue

        if not isinstance(
            data,
            dict,
        ):
            continue

        if not data.get(
            "visual_indexed"
        ):
            continue

        if data.get(
            "processed"
        ) is True:
            continue

        memory_id = clean_string(
            data.get(
                "memory_id"
            )
            or data.get(
                "id"
            )
        )

        if memory_id:
            pending.append(
                memory_id
            )

    update_progress(
        phase="enriching",
        active=True,
        enrichment_total=
            len(pending),
        enriched=0,
        enrichment_failed=0,
    )

    if not pending:

        update_progress(
            phase="ready",
            active=False,
        )

        return

    with ThreadPoolExecutor(
        max_workers=
            ENRICH_WORKERS
    ) as executor:

        futures = [
            executor.submit(
                enrich_one,
                memory_id,
            )
            for memory_id
            in pending
        ]

        for future in as_completed(
            futures
        ):

            try:

                success = bool(
                    future.result()
                )

            except Exception:
                success = False

            if success:

                increment_progress(
                    enriched=1
                )

            else:

                increment_progress(
                    enrichment_failed=1
                )

    update_progress(
        phase="ready",
        active=False,
    )


@router.post(
    "/ingest",
    summary="Fast gallery ingestion",
)
async def ingest_gallery(
    background_tasks: BackgroundTasks,
    files: list[UploadFile] = File(...),
    metadata_json: str = Form("[]"),
    total_hint: int = Form(0),
):

    if not files:

        raise HTTPException(
            status_code=400,
            detail="No images provided.",
        )

    if len(files) > FAST_BATCH_LIMIT:

        raise HTTPException(
            status_code=400,
            detail=(
                f"Maximum fast batch "
                f"is {FAST_BATCH_LIMIT}."
            ),
        )

    try:

        client_metadata = json.loads(
            metadata_json
            or "[]"
        )

    except Exception:

        client_metadata = []

    if not isinstance(
        client_metadata,
        list,
    ):

        client_metadata = []

    update_progress(
        phase="visual_indexing",
        active=True,
        discovered=max(
            int(
                total_hint
                or 0
            ),
            int(
                _PROGRESS.get(
                    "discovered",
                    0,
                )
            ),
        ),
    )

    ready_items = []
    results = []

    for index, file in enumerate(
        files
    ):

        filename = safe_filename(
            file.filename
        )

        try:

            data = await file.read()

            image_hash = (
                calculate_sha256(
                    data
                )
            )

            duplicate = (
                _find_duplicate_fast(
                    image_hash
                )
            )

            if duplicate:

                results.append(
                    {
                        "success":
                            True,

                        "duplicate":
                            True,

                        "visual_ready":
                            True,

                        "memory_id":
                            clean_string(
                                duplicate.get(
                                    "memory_id"
                                )
                                or duplicate.get(
                                    "id"
                                )
                            ),

                        "filename":
                            filename,
                    }
                )

                increment_progress(
                    duplicates=1
                )

                continue

            (
                extension,
                width,
                height,
            ) = validate_image_bytes(
                data,
                filename=filename,
                content_type=
                    file.content_type,
            )

            memory_id = generate_memory_id(
                image_hash
            )

            original = store_original(
                data,
                memory_id=memory_id,
                extension=extension,
            )

            thumbnail = thumbnail_file(
                memory_id
            )

            try:

                create_thumbnail(
                    data,
                    thumbnail,
                )

            except Exception:
                pass

            source_meta = (
                client_metadata[
                    index
                ]
                if (
                    index
                    < len(
                        client_metadata
                    )
                    and isinstance(
                        client_metadata[
                            index
                        ],
                        dict,
                    )
                )
                else {}
            )

            metadata = {
                "memory_id":
                    memory_id,

                "id":
                    memory_id,

                "filename":
                    filename,

                "original_filename":
                    filename,

                "mime_type":
                    file.content_type,

                "extension":
                    extension,

                "file_size":
                    len(data),

                "width":
                    width,

                "height":
                    height,

                "image_hash":
                    image_hash,

                "original_path":
                    str(
                        original.resolve()
                    ),

                "file_path":
                    str(
                        original.resolve()
                    ),

                "source_path":
                    str(
                        original.resolve()
                    ),

                "thumbnail_path":
                    (
                        str(
                            thumbnail.resolve()
                        )
                        if thumbnail.exists()
                        else None
                    ),

                "captured_at":
                    captured_at(
                        source_meta,
                        data,
                    ),

                "source_last_modified_ms":
                    source_meta.get(
                        "last_modified_ms"
                    ),

                "relative_path":
                    clean_string(
                        source_meta.get(
                            "relative_path"
                        )
                    ),

                "created_at":
                    utc_now_iso(),

                "updated_at":
                    utc_now_iso(),

                "processed":
                    False,

                "visual_indexed":
                    False,

                "status":
                    "stored",

                "source":
                    "gallery",
            }

            save_metadata(
                memory_id,
                metadata,
            )

            _register_hash(
                image_hash,
                metadata,
            )

            ready_items.append(
                {
                    "memory_id":
                        memory_id,

                    "filename":
                        filename,

                    "source":
                        original,

                    "metadata":
                        metadata,
                }
            )

        except Exception as exc:

            results.append(
                {
                    "success":
                        False,

                    "filename":
                        filename,

                    "error":
                        str(exc),
                }
            )

            increment_progress(
                visual_failed=1
            )

    if ready_items:

        service = (
            get_image_embedding_service()
        )

        store = (
            get_visual_vector_store()
        )

        try:

            matrix = (
                service.embed_images(
                    [
                        item[
                            "source"
                        ]
                        for item
                        in ready_items
                    ],
                    batch_size=
                        FAST_BATCH_LIMIT,
                )
            )

            store.add_many(
                [
                    (
                        item[
                            "memory_id"
                        ],
                        vector,
                    )
                    for (
                        item,
                        vector,
                    )
                    in zip(
                        ready_items,
                        matrix,
                    )
                ]
            )

            for item in ready_items:

                metadata = item[
                    "metadata"
                ]

                metadata.update(
                    {
                        "processed":
                            False,

                        "visual_indexed":
                            True,

                        "status":
                            "visual_ready",

                        "processing_status":
                            "visual_ready",

                        "updated_at":
                            utc_now_iso(),
                    }
                )

                save_metadata(
                    item[
                        "memory_id"
                    ],
                    metadata,
                )

                persist_search_memory(
                    memory_id=
                        item[
                            "memory_id"
                        ],
                    metadata=
                        metadata,
                    processing=
                        minimal_processing(),
                )

                results.append(
                    {
                        "success":
                            True,

                        "duplicate":
                            False,

                        "visual_ready":
                            True,

                        "memory_id":
                            item[
                                "memory_id"
                            ],

                        "filename":
                            item[
                                "filename"
                            ],
                    }
                )

            increment_progress(
                visual_ready=
                    len(
                        ready_items
                    )
            )

        except Exception as exc:

            logger.exception(
                "Batch visual indexing failed."
            )

            increment_progress(
                visual_failed=
                    len(
                        ready_items
                    )
            )

            for item in ready_items:

                results.append(
                    {
                        "success":
                            False,

                        "filename":
                            item[
                                "filename"
                            ],

                        "error":
                            str(exc),
                    }
                )

    return {
        "success":
            all(
                item.get(
                    "success"
                )
                for item
                in results
            ),

        "total":
            len(files),

        "visual_ready":
            sum(
                1
                for item
                in results
                if (
                    item.get(
                        "success"
                    )
                    and item.get(
                        "visual_ready"
                    )
                    and not item.get(
                        "duplicate"
                    )
                )
            ),

        "duplicates":
            sum(
                1
                for item
                in results
                if item.get(
                    "duplicate"
                )
            ),

        "failed":
            sum(
                1
                for item
                in results
                if not item.get(
                    "success"
                )
            ),

        "device":
            get_image_embedding_service()
            .device,

        "progress":
            dict(
                _PROGRESS
            ),

        "results":
            results,
    }


@router.post(
    "/enrich",
    summary="Start background enrichment",
)
async def start_enrichment(
    background_tasks: BackgroundTasks,
):

    background_tasks.add_task(
        enrich_pending
    )

    update_progress(
        phase="enriching",
        active=True,
    )

    return {
        "success": True,
        "message":
            "Background OCR and AI enrichment started.",
    }


@router.get(
    "/progress",
)
async def gallery_progress():

    progress = dict(
        _PROGRESS
    )

    store = (
        get_visual_vector_store()
    )

    progress[
        "visual_index_count"
    ] = store.count

    return progress


@router.get(
    "/capabilities",
)
async def capabilities():

    service = (
        get_image_embedding_service()
    )

    return {
        "batch_size":
            FAST_BATCH_LIMIT,

        "workers":
            ENRICH_WORKERS,

        "device":
            service.device,

        "visual_dimension":
            service.embedding_dimension,

        "progressive_search":
            True,

        "background_enrichment":
            True,
    }
