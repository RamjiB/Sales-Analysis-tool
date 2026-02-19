from __future__ import annotations

import json
import re
from datetime import date, datetime
from typing import Any, TypedDict

import duckdb
from langgraph.graph import END, StateGraph
from pydantic import BaseModel, Field

from app.chat.prompts import RESPONSE_PROMPT, RESOLVER_PROMPT, SQL_PROMPT, SYSTEM_PROMPT
from app.config import settings
from app.storage import load_registry


class QueryPlan(BaseModel):
    mode: str = "qa"
    intent: str = "analysis"
    rationale: str = ""


class SqlDraft(BaseModel):
    sql: str = ""
    assumptions: list[str] = Field(default_factory=list)
    detected_mode: str = "qa"


class ChatState(BaseModel):
    session_id: str
    question: str
    dataset_id: str
    table_name: str
    schema_columns: list[str] = Field(default_factory=list)
    categorical_value_hints: dict[str, list[str]] = Field(default_factory=dict)
    row_count: int = 0
    history: list[dict[str, str]] = Field(default_factory=list)
    plan: QueryPlan | None = None
    sql: str = ""
    rows: list[dict[str, Any]] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    answer: str = ""
    trace: list[dict[str, Any]] = Field(default_factory=list)


class GraphState(TypedDict):
    state: ChatState


SESSION_MEMORY: dict[str, list[dict[str, str]]] = {}
LAST_LLM_ERROR: dict[str, str] = {"json": "", "text": ""}


FORBIDDEN_SQL = re.compile(r"\b(insert|update|delete|drop|alter|create|attach|copy|truncate)\b", re.IGNORECASE)
GREETING_PATTERN = re.compile(r"^\s*(hi|hello|hey|good morning|good afternoon|good evening|hola)\b", re.IGNORECASE)


def _json_safe(value: Any) -> Any:
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    return value


def _active_dataset(payload: dict) -> tuple[str | None, str | None, dict[str, Any] | None]:
    registry = load_registry()
    dataset_id = payload.get("dataset_id") or registry.get("active_dataset_id")
    if not dataset_id:
        return None, None, None
    ds = registry.get("datasets", {}).get(dataset_id)
    if not ds or ds.get("status") != "ingested":
        return dataset_id, None, ds
    return dataset_id, ds.get("table_name"), ds


def _categorical_hints_from_dataset(dataset_record: dict[str, Any] | None) -> dict[str, list[str]]:
    if not isinstance(dataset_record, dict):
        return {}
    plan = dataset_record.get("ingestion_plan")
    if not isinstance(plan, dict):
        return {}
    hints = plan.get("categorical_unique_values")
    if not isinstance(hints, dict):
        return {}
    columns = hints.get("columns")
    if not isinstance(columns, dict):
        return {}

    parsed: dict[str, list[str]] = {}
    for key, value in columns.items():
        if not isinstance(value, list):
            continue
        parsed[str(key)] = [str(v) for v in value if v is not None]
    return parsed


def _is_greeting(text: str) -> bool:
    if not text:
        return False
    stripped = text.strip().lower()
    if len(stripped.split()) <= 6 and GREETING_PATTERN.search(stripped):
        return True
    return stripped in {"hi", "hello", "hey"}


def _greeting_response(dataset_ready: bool) -> str:
    if dataset_ready:
        return (
            "## Hi there!\n"
            "I’m ready to help analyze your sales data.\n\n"
            "- Ask for a **summary** of performance\n"
            "- Ask ad-hoc questions like **top categories**, **MoM trends**, or **underperforming segments**\n"
            "- I can explain results in simple business language"
        )
    return (
        "## Hi there!\n"
        "I’m ready to help with your sales analysis.\n\n"
        "Please upload a dataset first, and I’ll automatically ingest it.\n"
        "Once ingestion completes, you can ask questions in natural language."
    )


def _safe_table_name(table_name: str) -> str:
    return "".join(ch for ch in table_name if ch.isalnum() or ch == "_")


def _schema_for_table(table_name: str) -> tuple[list[str], int]:
    t = _safe_table_name(table_name)
    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        cols = [r[0] for r in con.execute(f"DESCRIBE {t}").fetchall()]
        cnt = con.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
    return cols, int(cnt)


def _call_gemini_json(prompt: str) -> dict[str, Any] | None:
    if not settings.gemini_api_key:
        LAST_LLM_ERROR["json"] = "GEMINI_API_KEY is not set."
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(
            prompt,
            generation_config={"temperature": 0, "response_mime_type": "application/json"},
        )
        text = (response.text or "").strip()
        parsed = _parse_json_text(text)
        if parsed is None:
            LAST_LLM_ERROR["json"] = "Gemini returned non-JSON response."
        return parsed if isinstance(parsed, dict) else None
    except Exception as exc:
        LAST_LLM_ERROR["json"] = f"{exc.__class__.__name__}: {exc}"
        return None


def _call_gemini_text(prompt: str) -> str | None:
    if not settings.gemini_api_key:
        LAST_LLM_ERROR["text"] = "GEMINI_API_KEY is not set."
        return None
    try:
        import google.generativeai as genai

        genai.configure(api_key=settings.gemini_api_key)
        model = genai.GenerativeModel(settings.gemini_model)
        response = model.generate_content(prompt, generation_config={"temperature": 0.2})
        text = (response.text or "").strip()
        if not text:
            LAST_LLM_ERROR["text"] = "Gemini returned empty text response."
        return text or None
    except Exception as exc:
        LAST_LLM_ERROR["text"] = f"{exc.__class__.__name__}: {exc}"
        return None


def _parse_json_text(text: str) -> Any:
    cleaned = text.strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", cleaned, flags=re.DOTALL).strip()
    try:
        return json.loads(cleaned)
    except Exception:
        pass
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start != -1 and end != -1 and end > start:
        candidate = cleaned[start : end + 1]
        try:
            return json.loads(candidate)
        except Exception:
            return None
    return None


def _resolver_agent(
    question: str, columns: list[str], categorical_hints: dict[str, list[str]], history: list[dict[str, str]]
) -> QueryPlan:
    prompt = (
        f"{SYSTEM_PROMPT}\n{RESOLVER_PROMPT}\n"
        "Return JSON: {mode, intent, rationale}.\n"
        f"Columns: {columns}\n"
        f"Categorical value hints (low-cardinality): {categorical_hints}\n"
        f"Recent history: {history[-6:]}\n"
        f"Question: {question}"
    )
    data = _call_gemini_json(prompt)
    if data is None:
        # Non-heuristic fallback: classify as qa and continue.
        reason = LAST_LLM_ERROR.get("json") or "Unknown LLM JSON error"
        return QueryPlan(mode="qa", intent="analysis", rationale=f"Fallback: no LLM response ({reason})")

    mode = str(data.get("mode", "qa")).lower()
    if mode not in {"summary", "qa"}:
        mode = "qa"
    return QueryPlan(mode=mode, intent=str(data.get("intent", "analysis")), rationale=str(data.get("rationale", "")))


def _sql_agent(
    question: str,
    table_name: str,
    columns: list[str],
    categorical_hints: dict[str, list[str]],
    history: list[dict[str, str]],
    plan: QueryPlan,
) -> SqlDraft:
    safe_table = _safe_table_name(table_name)
    prompt = (
        f"{SYSTEM_PROMPT}\n{SQL_PROMPT}\n"
        f"Active table: {safe_table}\n"
        f"Columns: {columns}\n"
        f"Categorical value hints (low-cardinality): {categorical_hints}\n"
        f"Mode from resolver: {plan.mode}\n"
        f"Intent from resolver: {plan.intent}\n"
        f"Recent history: {history[-6:]}\n"
        f"User question: {question}"
    )
    data = _call_gemini_json(prompt)
    if data is None:
        reason = LAST_LLM_ERROR.get("json") or "Unknown LLM JSON error"
        return SqlDraft(
            sql=f"SELECT * FROM {safe_table} LIMIT 25",
            assumptions=[f"SQL agent returned no JSON payload ({reason})."],
            detected_mode=plan.mode,
        )

    sql = str(data.get("sql", "")).strip()
    assumptions = data.get("assumptions", [])
    if not isinstance(assumptions, list):
        assumptions = []
    detected_mode = str(data.get("detected_mode", plan.mode)).lower()
    if detected_mode not in {"summary", "qa"}:
        detected_mode = plan.mode

    if not sql:
        assumptions.append("SQL missing from SQL-agent output.")
        sql = ""

    return SqlDraft(sql=sql, assumptions=[str(x) for x in assumptions], detected_mode=detected_mode)


def _sql_agent_retry(
    question: str,
    table_name: str,
    columns: list[str],
    categorical_hints: dict[str, list[str]],
    history: list[dict[str, str]],
    plan: QueryPlan,
    previous_error: str,
    previous_sql: str,
) -> SqlDraft:
    safe_table = _safe_table_name(table_name)
    corrective_prompt = (
        f"{SYSTEM_PROMPT}\n{SQL_PROMPT}\n"
        "Previous SQL output was invalid. Regenerate a corrected query.\n"
        f"Active table: {safe_table}\n"
        f"Columns: {columns}\n"
        f"Categorical value hints (low-cardinality): {categorical_hints}\n"
        f"Mode from resolver: {plan.mode}\n"
        f"Intent from resolver: {plan.intent}\n"
        f"Previous SQL: {previous_sql}\n"
        f"Validation error: {previous_error}\n"
        f"Recent history: {history[-6:]}\n"
        f"User question: {question}"
    )
    data = _call_gemini_json(corrective_prompt)
    if data is None:
        reason = LAST_LLM_ERROR.get("json") or "Unknown LLM JSON error"
        return SqlDraft(sql="", assumptions=[f"Retry SQL-agent returned no JSON payload ({previous_error}; {reason})."], detected_mode=plan.mode)
    sql = str(data.get("sql", "")).strip()
    assumptions = data.get("assumptions", [])
    if not isinstance(assumptions, list):
        assumptions = []
    detected_mode = str(data.get("detected_mode", plan.mode)).lower()
    if detected_mode not in {"summary", "qa"}:
        detected_mode = plan.mode
    return SqlDraft(sql=sql, assumptions=[str(x) for x in assumptions], detected_mode=detected_mode)


def _validate_sql(sql: str, table_name: str, allowed_columns: list[str]) -> tuple[bool, str]:
    txt = sql.strip()
    low = txt.lower()
    if not (low.startswith("select") or low.startswith("with")):
        return False, "Only read-only SELECT/WITH queries are allowed"
    if FORBIDDEN_SQL.search(low):
        return False, "Mutating SQL detected"

    safe_table = _safe_table_name(table_name).lower()
    if safe_table not in low:
        return False, "SQL must reference active ingested table"

    # Parse-check with DuckDB explain. If this fails, SQL is rejected.
    try:
        with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
            con.execute(f"EXPLAIN {txt}")
    except Exception as exc:
        return False, f"SQL parse/plan failed: {exc}"

    return True, "ok"


def _resolve_node(data: GraphState) -> GraphState:
    state = data["state"]
    state.trace.append({"stage": "resolver", "message": "Language-to-query resolution in progress"})
    state.plan = _resolver_agent(state.question, state.schema_columns, state.categorical_value_hints, state.history)
    state.trace.append(
        {
            "stage": "resolver",
            "message": "Plan generated",
            "details": state.plan.model_dump(),
        }
    )
    return {"state": state}


def _validator_node(data: GraphState) -> GraphState:
    state = data["state"]
    state.trace.append({"stage": "validator", "message": "Generating SQL from NL plan"})

    assert state.plan is not None
    draft = _sql_agent(
        state.question,
        state.table_name,
        state.schema_columns,
        state.categorical_value_hints,
        state.history,
        state.plan,
    )
    state.sql = draft.sql
    state.warnings.extend(draft.assumptions)

    ok, msg = _validate_sql(state.sql, state.table_name, state.schema_columns)
    if not ok:
        retry = _sql_agent_retry(
            state.question,
            state.table_name,
            state.schema_columns,
            state.categorical_value_hints,
            state.history,
            state.plan,
            previous_error=msg,
            previous_sql=state.sql,
        )
        if retry.sql:
            retry_ok, retry_msg = _validate_sql(retry.sql, state.table_name, state.schema_columns)
            state.warnings.extend(retry.assumptions)
            if retry_ok:
                state.sql = retry.sql
                ok, msg = True, "ok (after retry)"
            else:
                ok, msg = False, f"{msg}; retry_failed: {retry_msg}"

    if not ok:
        state.warnings.append(msg)
        state.sql = ""

    state.trace.append(
        {
            "stage": "validator",
            "message": "Validation complete",
            "details": {"ok": ok, "message": msg, "sql_preview": state.sql[:280]},
        }
    )
    return {"state": state}


def _extractor_node(data: GraphState) -> GraphState:
    state = data["state"]
    state.trace.append({"stage": "extractor", "message": "Executing query on data layer"})

    if not state.sql:
        state.rows = []
        state.trace.append({"stage": "extractor", "message": "Execution skipped due to invalid SQL"})
        return {"state": state}

    with duckdb.connect(str(settings.duckdb_path), read_only=True) as con:
        result = con.execute(state.sql)
        cols = [c[0] for c in result.description]
        rows = result.fetchall()
        state.rows = [{k: _json_safe(v) for k, v in dict(zip(cols, row)).items()} for row in rows]

    state.trace.append({"stage": "extractor", "message": f"Fetched {len(state.rows)} row(s)"})
    return {"state": state}


def _response_agent(state: ChatState) -> str | None:
    prompt = (
        f"{SYSTEM_PROMPT}\n{RESPONSE_PROMPT}\n"
        f"User question: {state.question}\n"
        f"Mode: {state.plan.mode if state.plan else 'qa'}\n"
        f"Rows: {state.rows[:30]}\n"
        f"Warnings: {state.warnings}\n"
        f"Recent history: {state.history[-6:]}"
    )
    return _call_gemini_text(prompt)


def _normalize_answer_text(text: str) -> str:
    raw = (text or "").strip()
    if not raw:
        return raw
    try:
        parsed = _parse_json_text(raw)
        if isinstance(parsed, dict):
            if isinstance(parsed.get("answer"), str):
                return parsed["answer"].strip()
            # If model returns structured payload without answer, compact it for readability.
            return json.dumps(parsed, indent=2)
    except Exception:
        pass
    return raw


def _friendly_fallback_answer(state: ChatState) -> str:
    if not state.rows:
        return "I could not find matching records for that request. Try rephrasing with a metric, period, or segment."

    first = state.rows[0]
    numeric_items = [(k, v) for k, v in first.items() if isinstance(v, (int, float)) and v is not None]
    text_items = [(k, v) for k, v in first.items() if isinstance(v, str) and v]

    summary_parts: list[str] = []
    if numeric_items:
        top_metrics = ", ".join(f"{k}={v:,.2f}" if isinstance(v, float) else f"{k}={v}" for k, v in numeric_items[:3])
        summary_parts.append(f"Key metrics from the top result: {top_metrics}.")
    if text_items:
        context = ", ".join(f"{k}: {v}" for k, v in text_items[:2])
        summary_parts.append(f"Leading context: {context}.")

    summary_parts.append(f"I evaluated {len(state.rows)} result records for this question.")
    summary_parts.append("If you want, I can break this down by month, region, or category next.")
    return " ".join(summary_parts)


def _responder_node(data: GraphState) -> GraphState:
    state = data["state"]
    state.trace.append({"stage": "responder", "message": "Preparing user-friendly response"})

    llm_answer = _response_agent(state)
    if llm_answer:
        state.answer = _normalize_answer_text(llm_answer)
    else:
        state.answer = _friendly_fallback_answer(state)

    if state.warnings:
        state.answer += "\n\nNotes: " + " | ".join(state.warnings[:5])

    state.trace.append({"stage": "responder", "message": "Final answer ready"})
    return {"state": state}


def _compile_graph():
    graph = StateGraph(GraphState)
    graph.add_node("resolver", _resolve_node)
    graph.add_node("validator", _validator_node)
    graph.add_node("extractor", _extractor_node)
    graph.add_node("responder", _responder_node)

    graph.set_entry_point("resolver")
    graph.add_edge("resolver", "validator")
    graph.add_edge("validator", "extractor")
    graph.add_edge("extractor", "responder")
    graph.add_edge("responder", END)
    return graph.compile()


GRAPH = _compile_graph()


def _build_final_response(out: ChatState) -> dict:
    return {
        "type": "final_answer",
        "payload": {
            "answer": out.answer,
            "sql": out.sql,
            "rows": out.rows[:50],
            "trace": out.trace,
            "mode": out.plan.mode if out.plan else "qa",
        },
    }


def stream_chat_events(payload: dict):
    if payload.get("type") != "user_message":
        yield {"type": "error", "message": "Only user_message is supported"}
        return

    session_id = str(payload.get("session_id") or "")
    question = str(payload.get("text") or "").strip()
    dataset_id, table_name, dataset_record = _active_dataset(payload)

    if not question:
        yield {"type": "error", "message": "Question is empty"}
        return

    if _is_greeting(question):
        ready = bool(dataset_id and table_name)
        yield {
            "type": "final_answer",
            "payload": {
                "answer": _greeting_response(ready),
                "sql": "",
                "rows": [],
                "trace": [{"stage": "responder", "message": "Handled greeting conversationally"}],
                "mode": "qa",
            },
        }
        return

    if not dataset_id or not table_name:
        yield {
            "type": "final_answer",
            "payload": {
                "answer": "No ingested dataset is active. Upload and ingest a dataset first.",
                "sql": "",
                "rows": [],
                "trace": [],
                "mode": "qa",
            },
        }
        return

    schema_cols, row_count = _schema_for_table(table_name)
    categorical_hints = _categorical_hints_from_dataset(dataset_record)
    history = SESSION_MEMORY.get(session_id, [])

    state = ChatState(
        session_id=session_id,
        question=question,
        dataset_id=dataset_id,
        table_name=table_name,
        schema_columns=schema_cols,
        categorical_value_hints=categorical_hints,
        row_count=row_count,
        history=history,
    )

    latest: ChatState | None = None
    emitted = 0
    for update in GRAPH.stream({"state": state}, stream_mode="updates"):
        for _, node_data in update.items():
            maybe_state = node_data.get("state")
            if maybe_state is None:
                continue
            latest = maybe_state
            new_steps = latest.trace[emitted:]
            for step in new_steps:
                yield {"type": "thinker_step", "payload": step}
            emitted = len(latest.trace)

    if latest is None:
        latest = GRAPH.invoke({"state": state})["state"]

    turns = history + [{"role": "user", "content": question}, {"role": "assistant", "content": latest.answer}]
    SESSION_MEMORY[session_id] = turns[-20:]
    yield _build_final_response(latest)


def handle_chat_message(payload: dict) -> dict:
    last = None
    for event in stream_chat_events(payload):
        last = event
    return last or {"type": "error", "message": "No response generated"}


def get_llm_diagnostics() -> dict[str, Any]:
    return {
        "configured": bool(settings.gemini_api_key),
        "model": settings.gemini_model,
        "last_json_error": LAST_LLM_ERROR.get("json") or None,
        "last_text_error": LAST_LLM_ERROR.get("text") or None,
    }
