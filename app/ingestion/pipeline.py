from __future__ import annotations

from pathlib import Path
from typing import Any

import duckdb

from app.config import settings
from app.ingestion.llm_planner import IngestionPlanner
from app.models import IngestionResponse
from app.storage import load_registry, save_registry


def _escape(path: str) -> str:
    return path.replace("'", "''")


def _quote_identifier(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


def _is_categorical_type(type_name: str) -> bool:
    normalized = (type_name or "").upper()
    return any(token in normalized for token in ("CHAR", "VARCHAR", "STRING", "TEXT", "BOOL"))


def _collect_low_cardinality_categorical_values(
    con: duckdb.DuckDBPyConnection, table_name: str, threshold_ratio: float = 0.10
) -> dict[str, Any]:
    quoted_table = _quote_identifier(table_name)
    rows = con.execute(f"DESCRIBE {quoted_table}").fetchall()

    columns: dict[str, list[str]] = {}
    for row in rows:
        col_name = str(row[0])
        col_type = str(row[1]) if len(row) > 1 else ""
        if not _is_categorical_type(col_type):
            continue

        quoted_col = _quote_identifier(col_name)
        non_null_count, distinct_count = con.execute(
            f"""
            SELECT
              COUNT(*) FILTER (WHERE {quoted_col} IS NOT NULL) AS non_null_count,
              COUNT(DISTINCT {quoted_col}) AS distinct_count
            FROM {quoted_table}
            """
        ).fetchone()

        non_null = int(non_null_count or 0)
        distinct = int(distinct_count or 0)
        if non_null == 0:
            continue
        if (distinct / non_null) >= threshold_ratio:
            continue

        values = con.execute(
            f"""
            SELECT DISTINCT CAST({quoted_col} AS VARCHAR) AS value
            FROM {quoted_table}
            WHERE {quoted_col} IS NOT NULL
            ORDER BY value
            """
        ).fetchall()
        columns[col_name] = [str(v[0]) for v in values if v and v[0] is not None]

    return {"threshold_ratio": threshold_ratio, "columns": columns}


def _ingest_csv(con: duckdb.DuckDBPyConnection, table_name: str, file_path: str) -> None:
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_csv_auto('{_escape(file_path)}', HEADER=TRUE, SAMPLE_SIZE=-1)")


def _ingest_json(con: duckdb.DuckDBPyConnection, table_name: str, file_path: str) -> None:
    con.execute(f"CREATE OR REPLACE TABLE {table_name} AS SELECT * FROM read_json_auto('{_escape(file_path)}')")


def _ingest_excel(con: duckdb.DuckDBPyConnection, table_name: str, file_path: str) -> None:
    import pandas as pd

    tmp_csv = Path(file_path).with_suffix(".tmp.csv")
    pd.read_excel(file_path).to_csv(tmp_csv, index=False)
    try:
        _ingest_csv(con, table_name, str(tmp_csv))
    finally:
        tmp_csv.unlink(missing_ok=True)


def run_ingestion(record: dict, progress_cb=None) -> IngestionResponse:
    if progress_cb:
        progress_cb(10, "planning", "Building ingestion plan")
    planner = IngestionPlanner()
    plan = planner.create_plan(record["file_path"], record["file_type"], record["table_name"])

    if progress_cb:
        progress_cb(45, "loading", "Loading data into DuckDB")
    settings.duckdb_path.parent.mkdir(parents=True, exist_ok=True)
    categorical_unique_values = {"threshold_ratio": 0.10, "columns": {}}
    with duckdb.connect(str(settings.duckdb_path)) as con:
        if record["file_type"] == "csv":
            _ingest_csv(con, record["table_name"], record["file_path"])
        elif record["file_type"] == "json":
            _ingest_json(con, record["table_name"], record["file_path"])
        elif record["file_type"] in {"xlsx", "xls"}:
            _ingest_excel(con, record["table_name"], record["file_path"])
        else:
            raise ValueError("Unsupported file type for ingestion")
        try:
            categorical_unique_values = _collect_low_cardinality_categorical_values(con, record["table_name"])
        except Exception:
            # Metadata enrichment should never block ingestion.
            categorical_unique_values = {"threshold_ratio": 0.10, "columns": {}}

    if progress_cb:
        progress_cb(85, "finalizing", "Updating dataset metadata")
    registry = load_registry()
    ds = registry["datasets"][record["dataset_id"]]
    ds["status"] = "ingested"
    plan = dict(plan)
    plan["categorical_unique_values"] = categorical_unique_values
    ds["ingestion_plan"] = plan
    registry["datasets"][record["dataset_id"]] = ds
    registry["active_dataset_id"] = record["dataset_id"]
    save_registry(registry)

    if progress_cb:
        progress_cb(100, "ingested", "Ingestion complete")
    return IngestionResponse(
        dataset_id=record["dataset_id"],
        table_name=record["table_name"],
        status="ingested",
        plan=plan,
    )
