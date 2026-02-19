from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class DatasetRecord(BaseModel):
    dataset_id: str
    filename: str
    file_path: str
    file_type: str
    table_name: str
    status: str = "uploaded"
    ingestion_plan: dict[str, Any] | None = None


class IngestionRequest(BaseModel):
    dataset_id: str


class IngestionResponse(BaseModel):
    dataset_id: str
    table_name: str
    status: str
    plan: dict[str, Any]


class ChatMessage(BaseModel):
    type: str = Field(description="user_message")
    session_id: str
    text: str
    dataset_id: str | None = None
