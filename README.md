# Sales Chat Ingestion App

FastAPI + DuckDB application for uploading tabular datasets and querying them in natural language through a multi-agent pipeline.

## What It Does

- Accepts `csv`, `xlsx`, `xls`, `json` uploads
- Ingests data into DuckDB
- Builds metadata (including low-cardinality categorical value hints)
- Runs a chat graph:
  - Resolver -> SQL generator/validator -> Extractor -> Response
- Streams intermediate thinker steps over WebSocket

## Tech Stack

- FastAPI
- DuckDB
- LangGraph
- Gemini API (optional)

## Architecture Diagram

![Sales Assistant Architecture](sales_assistant_azure_100gb_architecture.drawio.png)

## Project Structure

```text
app/
  api.py
  config.py
  models.py
  storage.py
  upload_service.py
  ingestion/
    llm_planner.py
    pipeline.py
  chat/
    prompts.py
    service.py
  ui/
    index.html
    main.js
data/
  meta/.gitkeep
  uploads/.gitkeep
```

Runtime files like `data/ingested.duckdb` and `data/meta/datasets.json` are created automatically when the app runs.

## Run

```bash
uv sync
uv run uvicorn app.api:app --reload --port 8000
```

Open `http://localhost:8000`.

## Environment

Set values in `.env` (optional):

```bash
GEMINI_API_KEY=your_key
GEMINI_MODEL=gemini-3-pro-preview
DATA_ROOT=data
UPLOADS_DIR=data/uploads
META_FILE=data/meta/datasets.json
DUCKDB_PATH=data/ingested.duckdb
```

If `GEMINI_API_KEY` is not set, the system falls back to deterministic logic where applicable.

## API Endpoints

- `POST /upload`
- `GET /datasets`
- `POST /datasets/{dataset_id}/activate`
- `POST /ingest/{dataset_id}`
- `GET /ingest/status/{dataset_id}`
- `GET /llm/diagnostics`
- `WS /ws/chat`

## Notes

- SQL is read-only by validation policy.
- Upload + ingestion is asynchronous.
- Dataset registry is stored in `data/meta/datasets.json`.
