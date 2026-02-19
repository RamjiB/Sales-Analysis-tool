from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from app.config import settings


def ensure_dirs() -> None:
    settings.uploads_dir.mkdir(parents=True, exist_ok=True)
    settings.meta_file.parent.mkdir(parents=True, exist_ok=True)
    settings.data_root.mkdir(parents=True, exist_ok=True)


def load_registry() -> dict[str, Any]:
    ensure_dirs()
    if not settings.meta_file.exists():
        return {"datasets": {}, "active_dataset_id": None}
    return json.loads(settings.meta_file.read_text())


def save_registry(registry: dict[str, Any]) -> None:
    ensure_dirs()
    settings.meta_file.write_text(json.dumps(registry, indent=2))


def upsert_dataset(dataset: dict[str, Any]) -> None:
    registry = load_registry()
    registry["datasets"][dataset["dataset_id"]] = dataset
    if registry.get("active_dataset_id") is None:
        registry["active_dataset_id"] = dataset["dataset_id"]
    save_registry(registry)


def set_active_dataset(dataset_id: str) -> None:
    registry = load_registry()
    registry["active_dataset_id"] = dataset_id
    save_registry(registry)
