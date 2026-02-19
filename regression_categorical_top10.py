from __future__ import annotations

import duckdb

from app.ingestion.pipeline import _collect_low_cardinality_categorical_values


def main() -> None:
    con = duckdb.connect()
    con.execute("CREATE TABLE t(color VARCHAR)")

    # Make 12 distinct values with descending frequencies so only top 10 are kept.
    rows: list[tuple[str]] = []
    for i in range(12):
        value = f"c{i}"
        rows.extend([(value,)] * (12 - i))
    con.executemany("INSERT INTO t VALUES (?)", rows)

    out = _collect_low_cardinality_categorical_values(
        con,
        "t",
        threshold_ratio=1.0,  # ensure we don't filter out the column
        max_distinct=10_000,
        max_values_per_column=10,
    )
    values = out.get("columns", {}).get("color", [])
    if len(values) != 10:
        raise SystemExit(f"Expected 10 values, got {len(values)}: {values}")
    if values[0] != "c0" or values[-1] != "c9":
        raise SystemExit(f"Unexpected ordering: {values}")

    print("OK: categorical unique values are top-10 by count")


if __name__ == "__main__":
    main()

