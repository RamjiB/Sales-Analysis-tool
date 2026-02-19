from __future__ import annotations

import re
import uuid
from pathlib import Path

from fastapi import UploadFile

from app.config import settings
from app.models import DatasetRecord
from app.storage import upsert_dataset

_ALLOWED_EXT = {"csv", "xlsx", "xls", "json"}


def _safe_stem(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_]+", "_", Path(name).stem).strip("_").lower() or "dataset"


def save_upload(upload: UploadFile) -> DatasetRecord:
    suffix = Path(upload.filename or "").suffix.lower().lstrip(".")
    if suffix not in _ALLOWED_EXT:
        raise ValueError("Only CSV, Excel, or JSON files are supported")

    dataset_id = str(uuid.uuid4())
    table_name = f"sales_{_safe_stem(upload.filename or 'dataset')}_{dataset_id[:8]}"
    target = settings.uploads_dir / f"{dataset_id}_{upload.filename}"

    content = upload.file.read()
    target.write_bytes(content)

    record = DatasetRecord(
        dataset_id=dataset_id,
        filename=upload.filename or target.name,
        file_path=str(target),
        file_type=suffix,
        table_name=table_name,
    )
    upsert_dataset(record.model_dump())
    return record
