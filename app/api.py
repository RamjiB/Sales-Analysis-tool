from __future__ import annotations

import json
import threading
import time

from fastapi import FastAPI, File, HTTPException, UploadFile, WebSocket, WebSocketDisconnect
from fastapi import Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.chat.service import get_llm_diagnostics, stream_chat_events
from app.ingestion.pipeline import run_ingestion
from app.models import IngestionResponse
from app.storage import ensure_dirs, load_registry, save_registry, set_active_dataset
from app.upload_service import save_upload

app = FastAPI(title="Sales Chat Ingestion App")
ingestion_jobs: dict[str, dict] = {}


@app.on_event("startup")
def startup() -> None:
    ensure_dirs()


@app.post("/upload")
def upload_dataset(file: UploadFile = File(...)) -> dict:
    try:
        record = save_upload(file)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    set_active_dataset(record.dataset_id)
    _start_ingestion_job(record.model_dump())
    return {
        "dataset": record.model_dump(),
        "message": "Upload successful. Ingestion started automatically.",
    }


@app.get("/datasets")
def list_datasets() -> dict:
    return load_registry()


@app.get("/llm/diagnostics")
def llm_diagnostics() -> dict:
    return get_llm_diagnostics()


@app.post("/datasets/{dataset_id}/activate")
def activate_dataset(dataset_id: str) -> dict:
    registry = load_registry()
    if dataset_id not in registry["datasets"]:
        raise HTTPException(status_code=404, detail="Dataset not found")
    set_active_dataset(dataset_id)
    return {"active_dataset_id": dataset_id}


@app.post("/ingest/{dataset_id}", response_model=IngestionResponse)
def ingest(dataset_id: str) -> IngestionResponse:
    registry = load_registry()
    record = registry["datasets"].get(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found")
    _start_ingestion_job(record)
    set_active_dataset(dataset_id)
    return IngestionResponse(
        dataset_id=dataset_id,
        table_name=record["table_name"],
        status="ingesting",
        plan={"message": "Ingestion started"},
    )


@app.get("/ingest/status/{dataset_id}")
def ingestion_status(dataset_id: str) -> dict:
    registry = load_registry()
    record = registry["datasets"].get(dataset_id)
    if not record:
        raise HTTPException(status_code=404, detail="Dataset not found")
    job = ingestion_jobs.get(dataset_id, {"progress": 0, "stage": "unknown", "message": "No ingestion job"})
    return {"dataset_id": dataset_id, "status": record.get("status"), "job": job}


@app.websocket("/ws/chat")
async def ws_chat(websocket: WebSocket) -> None:
    await websocket.accept()
    try:
        while True:
            text = await websocket.receive_text()
            payload = json.loads(text)
            try:
                for event in stream_chat_events(payload):
                    if event.get("type") == "final_answer" and "trace" in event.get("payload", {}):
                        event = dict(event)
                        event["payload"] = dict(event["payload"])
                        event["payload"].pop("trace", None)
                    await websocket.send_json(event)
            except Exception as exc:
                await websocket.send_json({"type": "error", "message": f"Chat processing failed: {exc}"})
    except WebSocketDisconnect:
        return


app.mount("/static", StaticFiles(directory="app/ui"), name="static")


@app.get("/")
def index() -> FileResponse:
    return FileResponse("app/ui/index.html")


@app.get("/socket.io/")
@app.get("/socket.io/{path:path}")
def socketio_noop(path: str = "") -> Response:
    return Response(status_code=204)


def _set_dataset_status(dataset_id: str, status: str) -> None:
    registry = load_registry()
    record = registry["datasets"].get(dataset_id)
    if not record:
        return
    record["status"] = status
    registry["datasets"][dataset_id] = record
    save_registry(registry)


def _start_ingestion_job(record: dict) -> None:
    dataset_id = record["dataset_id"]
    current = ingestion_jobs.get(dataset_id)
    if current and current.get("stage") in {"queued", "planning", "loading", "finalizing"}:
        return

    ingestion_jobs[dataset_id] = {
        "progress": 2,
        "stage": "queued",
        "message": "Queued for ingestion",
        "done": False,
        "updated_at": time.time(),
    }
    _set_dataset_status(dataset_id, "ingesting")

    def progress_cb(progress: int, stage: str, message: str) -> None:
        ingestion_jobs[dataset_id] = {
            "progress": progress,
            "stage": stage,
            "message": message,
            "done": stage in {"ingested", "failed"},
            "updated_at": time.time(),
        }

    def worker() -> None:
        try:
            run_ingestion(record, progress_cb=progress_cb)
        except Exception as exc:
            ingestion_jobs[dataset_id] = {
                "progress": 100,
                "stage": "failed",
                "message": str(exc),
                "done": True,
                "updated_at": time.time(),
            }
            _set_dataset_status(dataset_id, "failed")

    threading.Thread(target=worker, daemon=True).start()
