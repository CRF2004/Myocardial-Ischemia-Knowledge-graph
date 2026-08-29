"""
Simple evaluation runner: scan results JSONs, compute metrics, output CSV.
"""

from __future__ import annotations

import csv
import importlib.util
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List


def _load_json_records(path: Path) -> List[Dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        results = data.get("results")
        if isinstance(results, list):
            return results
    raise ValueError(f"Unsupported JSON format in {path}")


def _load_accuracy_module():
    module_path = Path(__file__).parent / "metrics" / "accuracy.py"
    spec = importlib.util.spec_from_file_location("accuracy", module_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load accuracy module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _unique_value(records: Iterable[Dict[str, Any]], key: str) -> str:
    values = {str(item.get(key)) for item in records if item.get(key) is not None}
    if not values:
        return ""
    if len(values) == 1:
        return next(iter(values))
    return "mixed"


def _collect_rows(results_dir: Path) -> List[Dict[str, Any]]:
    accuracy = _load_accuracy_module()
    rows: List[Dict[str, Any]] = []
    for path in sorted(results_dir.glob("*.json")):
        records = _load_json_records(path)
        if not records:
            continue
        metrics = accuracy.compute(records)
        row = {
            "file": path.name,
            "method": _unique_value(records, "method"),
            "model": _unique_value(records, "model"),
            **metrics,
        }
        rows.append(row)
    return rows


def _write_csv(rows: List[Dict[str, Any]], output_path: Path) -> None:
    fieldnames = [
        "file",
        "method",
        "model",
        "accuracy",
        "macro_accuracy",
        "total",
        "correct",
        "num_questions",
        "missing_question_id",
        "metric",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key) for key in fieldnames})


def _print_df(rows: List[Dict[str, Any]]) -> None:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        total_files = len(rows)
        avg_accuracy = (
            sum(row.get("accuracy", 0.0) for row in rows) / total_files
            if total_files
            else 0.0
        )
        avg_macro = (
            sum(row.get("macro_accuracy", 0.0) for row in rows) / total_files
            if total_files
            else 0.0
        )
        print(f"Files: {total_files}")
        print(f"Mean accuracy: {avg_accuracy:.4f}")
        print(f"Mean macro_accuracy: {avg_macro:.4f}")
        print("Install pandas to see full DataFrame stats.")
        return

    df = pd.DataFrame(rows)
    print(df)
    numeric_cols = df.select_dtypes(include="number")
    if not numeric_cols.empty:
        print(numeric_cols.describe())


def main() -> int:
    results_dir = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/results")
    output_csv = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/eval/eval_results.csv")

    if not results_dir.exists():
        print(f"Results directory not found: {results_dir}")
        return 1

    rows = _collect_rows(results_dir)
    if not rows:
        print(f"No JSON results found in {results_dir}")
        return 0

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    _write_csv(rows, output_csv)
    _print_df(rows)
    print(f"Wrote CSV: {output_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
