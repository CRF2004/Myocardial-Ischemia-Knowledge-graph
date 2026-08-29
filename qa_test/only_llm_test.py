"""
Zero-context QA baseline: prompt the model with question + options only.
"""

from __future__ import annotations

import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from tqdm import tqdm
from typing import Any, Dict, Iterable, List, Optional


Choice = str

def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of questions in {path}")
    return data


def build_prompt(item: Dict[str, Any]) -> str:
    question = item.get("question", "").strip()
    options = item.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("Expected options to be a dict of choice -> text")

    ordered_keys = sorted(options.keys())
    options_lines = [f"{key}. {options[key]}" for key in ordered_keys]

    prompt = "\n".join(
        [
            "Answer the following multiple-choice question.",
            "",
            question,
            "",
            "Options:",
            *options_lines,
            "",
            "Reply in JSON only, using this schema:",
            '{"answer": "A"}',
            "Use the single best option letter as the value.",
        ]
    )
    return prompt


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return None


def parse_choice(text: str, valid_choices: Iterable[str]) -> Optional[Choice]:
    valid_set = {c.upper() for c in valid_choices}
    parsed = _extract_json_object(text)
    if isinstance(parsed, dict):
        raw_answer = parsed.get("answer")
        if isinstance(raw_answer, str):
            candidate = raw_answer.strip().upper()
            if candidate in valid_set:
                return candidate
    match = re.search(r"\b([A-Z])\b", text.upper())
    if not match:
        return None
    choice = match.group(1)
    return choice if choice in valid_set else None


def call_llm(model: str, prompt: str) -> str:
    try:
        import LLM_CALL  # type: ignore
    except Exception as exc:  # pragma: no cover - best-effort import
        raise RuntimeError("LLM_CALL not available") from exc
    return LLM_CALL.chat(model, prompt)


def choose_answer(
    prompt: str,
    item: Dict[str, Any],
    model: str,
) -> Dict[str, Any]:
    options = item.get("options", {})
    option_keys = sorted(options.keys())
    raw_response = call_llm(model, prompt)
    answer = parse_choice(raw_response, option_keys)
    if answer is None:
        answer = option_keys[0] if option_keys else None

    return {
        "answer": answer,
        "raw_response": raw_response,
    }


def run(
    data_path: Path,
    limit: Optional[int],
    output_path: Optional[Path],
    model: str,
    workers: int,
) -> int:
    data = load_questions(data_path)
    if limit is not None:
        data = data[:limit]

    def process_item(index: int, item: Dict[str, Any]) -> tuple[int, Dict[str, Any]]:
        prompt = build_prompt(item)
        result = choose_answer(prompt, item, model)
        expected = item.get("answer")
        record = {
            "q_id": item.get("q_id"),
            "method": "only_llm",
            "model": model,
            "answer": result["answer"],
            "is_correct": result["answer"] == expected if expected else None,
            "raw_response": result["raw_response"] or None,
        }
        return index, record

    results: List[Optional[Dict[str, Any]]] = [None] * len(data)
    if workers <= 1 or len(data) <= 1:
        for index, item in tqdm(enumerate(data), total=len(data)):
            _, record = process_item(index, item)
            results[index] = record
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_item, index, item)
                for index, item in enumerate(data)
            ]
            for future in tqdm(as_completed(futures), total=len(futures)):
                index, record = future.result()
                results[index] = record

    results = [item for item in results if item is not None]

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=True, indent=2)
    else:
        correct = sum(1 for item in results if item.get("is_correct"))
        total = len(results)
        accuracy = correct / total if total else 0.0
        print(f"Processed {total} questions. Accuracy={accuracy:.3f}")

    return 0


def main() -> int:
    # Simple defaults: edit these if you want to change behavior.
    DATA_PATH = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_2_filtered.json")
    OUTPUT_PATH = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/results/only_llm_2.json")
    MODEL_NAME = "gpt-4o-mini"
    LIMIT = None # 设置参数可以仅测试部分数据
    WORKERS = 4
    
    return run(
        data_path=DATA_PATH,
        limit=LIMIT,
        output_path=OUTPUT_PATH,
        model=MODEL_NAME,
        workers=WORKERS,
    )

if __name__ == "__main__":
    main()
