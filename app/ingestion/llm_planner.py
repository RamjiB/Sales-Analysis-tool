from __future__ import annotations

import csv
import json
import re
from pathlib import Path
from typing import Any

from app.config import settings


class IngestionPlanner:
    def create_plan(self, file_path: str, file_type: str, table_name: str) -> dict[str, Any]:
        sample = self._read_sample(file_path, file_type)
        base = {
            "file_path": file_path,
            "file_type": file_type,
            "table_name": table_name,
            "sample_rows": sample,
            "target_domain": "sales",
        }

        if settings.gemini_api_key:
            try:
                return self._create_with_llm(base)
            except Exception:
                pass
        return self._create_fallback(base)

    def _read_sample(self, file_path: str, file_type: str) -> list[dict[str, Any]]:
        path = Path(file_path)
        if file_type == "csv":
            with path.open("r", encoding="utf-8", errors="replace", newline="") as f:
                rows = list(csv.DictReader(f))[:10]
            return rows
        if file_type == "json":
            data = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(data, list):
                return data[:10]
            if isinstance(data, dict):
                return [data]
            return []
        if file_type in {"xlsx", "xls"}:
            import pandas as pd

            df = pd.read_excel(path, nrows=10)
            return df.fillna("").to_dict(orient="records")
        return []

    def _create_with_llm(self, base: dict[str, Any]) -> dict[str, Any]:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        prompt = (
            "You are an ingestion planner. Return JSON only with keys:"
            "domain, table_name, columns[{name,inferred_type,role}], cleaning_rules[list],"
            "time_columns[list], numeric_columns[list], entity_columns[list]."
            f" Input metadata: {base}"
        )
        response = model.generate_content(prompt)
        text = (response.text or "").strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.DOTALL).strip()
        plan = json.loads(text)
        plan["planner"] = "llm"
        return plan

    def _create_fallback(self, base: dict[str, Any]) -> dict[str, Any]:
        sample_rows = base["sample_rows"]
        columns = []
        if sample_rows:
            for key, value in sample_rows[0].items():
                inferred_type = "string"
                if isinstance(value, (int, float)):
                    inferred_type = "numeric"
                elif isinstance(value, str):
                    low = key.lower()
                    if any(t in low for t in ["date", "month", "year"]):
                        inferred_type = "date_or_time"
                    elif any(t in low for t in ["amount", "price", "qty", "quantity", "sales", "revenue"]):
                        inferred_type = "numeric"
                columns.append({"name": key, "inferred_type": inferred_type, "role": "feature"})

        return {
            "planner": "fallback",
            "domain": "sales",
            "table_name": base["table_name"],
            "columns": columns,
            "cleaning_rules": [
                "Trim whitespace from column names",
                "Keep all columns as ingestable fields",
                "Parse date-like columns where possible",
                "Retain nulls without dropping rows",
            ],
            "time_columns": [c["name"] for c in columns if c["inferred_type"] == "date_or_time"],
            "numeric_columns": [c["name"] for c in columns if c["inferred_type"] == "numeric"],
            "entity_columns": [c["name"] for c in columns if c["inferred_type"] == "string"],
        }
