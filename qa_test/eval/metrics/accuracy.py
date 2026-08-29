"""
Accuracy metric helpers.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Dict, Iterable, List, Optional


def _coerce_bool(value: Any) -> Optional[bool]:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "false"}:
            return normalized == "true"
    return None


def compute_accuracy(
    records: Iterable[Dict[str, Any]],
    *,
    correctness_key: str = "is_correct",
    group_key: str = "q_id",
) -> Dict[str, Any]:
    """
    Compute overall accuracy and macro accuracy (per question average).

    Expected record fields:
      - is_correct: bool (or 0/1 or "true"/"false")
      - q_id: used for macro accuracy grouping
    """
    total = 0
    correct = 0
    grouped: Dict[str, List[bool]] = defaultdict(list)
    missing_group = 0

    for item in records:
        value = _coerce_bool(item.get(correctness_key))
        if value is None:
            continue
        total += 1
        if value:
            correct += 1
        q_id = item.get(group_key)
        if isinstance(q_id, str) and q_id:
            grouped[q_id].append(value)
        else:
            missing_group += 1

    accuracy = correct / total if total else 0.0

    macro_accuracy = 0.0
    if grouped:
        per_question = [sum(vals) / len(vals) for vals in grouped.values()]
        macro_accuracy = sum(per_question) / len(per_question)

    return {
        "metric": "accuracy",
        "accuracy": accuracy,
        "macro_accuracy": macro_accuracy,
        "total": total,
        "correct": correct,
        "num_questions": len(grouped),
        "missing_question_id": missing_group,
    }


def compute(records: Iterable[Dict[str, Any]]) -> Dict[str, Any]:
    """Default entrypoint for eval.py."""
    return compute_accuracy(records)
