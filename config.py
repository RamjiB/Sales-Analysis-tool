from dataclasses import dataclass
from pathlib import Path


def read_env(env_path=None):
    env = {}
    if env_path is None:
        # Resolve .env from project root (parent of app/) so it's found regardless of cwd
        env_path = Path(__file__).resolve().parent.parent / ".env"
    p = Path(env_path)
    if p.exists():
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            env[key.strip()] = value.strip().strip("'").strip('"')
    return env

_env = read_env()

def _env_float(key: str, default: float) -> float:
    raw = _env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except Exception:
        return default


def _env_int(key: str, default: int) -> int:
    raw = _env.get(key)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except Exception:
        return default


@dataclass(frozen=True)
class Settings:
    app_name: str = "Sales Chat Ingestion App"
    data_root: Path = Path(_env.get("DATA_ROOT", "data"))
    uploads_dir: Path = Path(_env.get("UPLOADS_DIR", "data/uploads"))
    meta_file: Path = Path(_env.get("META_FILE", "data/meta/datasets.json"))
    duckdb_path: Path = Path(_env.get("DUCKDB_PATH", "data/ingested.duckdb"))
    gemini_api_key: str | None = _env.get("GEMINI_API_KEY")
    gemini_model: str = _env.get("GEMINI_MODEL") or "gemini-3-pro-preview"

    # Metadata enrichment knobs (kept conservative to avoid huge datasets.json files).
    categorical_hint_threshold_ratio: float = _env_float("CATEGORICAL_HINT_THRESHOLD_RATIO", 0.10)
    categorical_hint_max_distinct: int = _env_int("CATEGORICAL_HINT_MAX_DISTINCT", 200)
    categorical_hint_max_values: int = _env_int("CATEGORICAL_HINT_MAX_VALUES", 10)

settings = Settings()
