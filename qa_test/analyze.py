from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List, Optional


def load_json_list(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list in {path}")
    return data


def load_chunks_by_qid(path: Path) -> Dict[str, List[Dict[str, Any]]]:
    """
    Build q_id -> chunks mapping from a retrieval results file.
    """
    data = load_json_list(path)
    mapping: Dict[str, List[Dict[str, Any]]] = {}
    for item in data:
        q_id = str(item.get("q_id", ""))
        context = item.get("context", {}) if isinstance(item, dict) else {}
        chunks = context.get("chunks", []) if isinstance(context, dict) else []
        mapping[q_id] = chunks if isinstance(chunks, list) else []
    return mapping


def build_prompt(item: Dict[str, Any], chunks: List[Dict[str, Any]]) -> str:
    question = item.get("question", "").strip()
    options = item.get("options", {})
    option_lines: List[str] = []
    if isinstance(options, dict):
        for key in sorted(options.keys()):
            option_lines.append(f"{key}. {options[key]}")

    chunk_lines: List[str] = []
    for idx, chunk in enumerate(chunks, start=1):
        text = str(chunk.get("text", "")).strip()
        file_name = chunk.get("file", "unknown")
        chunk_index = chunk.get("chunk_index", "unknown")
        header = f"[Chunk {idx} | {file_name} | {chunk_index}]"
        chunk_lines.append(f"{header}\n{text}")

    prompt_parts = [
        "You are analyzing how helpful each text chunk is for answering the question.",
        "For each chunk, rate:",
        "- helpfulness: 0 (no help), 1 (some help), 2 (direct help)",
        "- relevance: 0 (irrelevant), 1 (partially relevant), 2 (highly relevant)",
        "Give a brief reason (<= 20 words).",
        "Reply in JSON only as a list of objects with keys:",
        '["chunk_id", "helpfulness", "relevance", "reason"].',
        "Use chunk_id like \"Chunk 1\", \"Chunk 2\", ...",
        "",
        "Question:",
        question,
    ]
    if option_lines:
        prompt_parts.extend(["", "Options:", *option_lines])

    prompt_parts.extend(["", "Chunks:", *chunk_lines])
    return "\n".join(prompt_parts)


def _extract_json_array(text: str) -> Optional[List[Dict[str, Any]]]:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, list):
        return parsed
    return None


def _extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = text[start : end + 1]
    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    if isinstance(parsed, dict):
        return parsed
    return None


def parse_response(text: str) -> Optional[List[Dict[str, Any]]]:
    parsed_list = _extract_json_array(text)
    if parsed_list is not None:
        return parsed_list
    parsed_obj = _extract_json_object(text)
    if parsed_obj and isinstance(parsed_obj.get("chunks"), list):
        return parsed_obj["chunks"]
    return None


def call_llm(model: str, prompt: str) -> str:
    try:
        import LLM_CALL  # type: ignore
    except Exception as exc:  # pragma: no cover - best-effort import
        raise RuntimeError("LLM_CALL not available") from exc
    return LLM_CALL.chat(model, prompt)


def analyze_item(
    item: Dict[str, Any],
    chunks: List[Dict[str, Any]],
    model: str,
) -> Dict[str, Any]:
    if not chunks:
        return {
            "q_id": item.get("q_id"),
            "question": item.get("question"),
            "analysis": [],
            "chunks": [],
            "raw_response": None,
        }

    prompt = build_prompt(item, chunks)
    raw_response = call_llm(model, prompt)
    analysis = parse_response(raw_response)
    if analysis is None:
        analysis = []

    return {
        "q_id": item.get("q_id"),
        "question": item.get("question"),
        "analysis": analysis,
        "chunks": chunks,
        "raw_response": raw_response,
    }


def run(
    data_path: Path,
    context_path: Path,
    output_path: Path,
    model: str,
    limit: Optional[int],
    workers: int,
) -> int:
    data = load_json_list(data_path)
    if limit is not None:
        data = data[:limit]

    chunks_by_qid = load_chunks_by_qid(context_path) if context_path.exists() else {}

    results: List[Optional[Dict[str, Any]]] = [None] * len(data)
    total = len(data)
    processed = 0

    def report_progress() -> None:
        print(f"Progress: {processed}/{total}", end="\r", flush=True)

    def process_item(index: int, item: Dict[str, Any]) -> Dict[str, Any]:
        q_id = str(item.get("q_id", ""))
        chunks = chunks_by_qid.get(q_id, [])
        return analyze_item(item, chunks, model)

    if workers <= 1 or len(data) <= 1:
        for idx, item in enumerate(data):
            results[idx] = process_item(idx, item)
            processed += 1
            report_progress()
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = {
                executor.submit(process_item, idx, item): idx
                for idx, item in enumerate(data)
            }
            for future in as_completed(futures):
                idx = futures[future]
                results[idx] = future.result()
                processed += 1
                report_progress()

    if total:
        print(f"Progress: {processed}/{total}")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        json.dump(
            [item for item in results if item is not None],
            handle,
            ensure_ascii=True,
            indent=2,
        )
    return 0


def main() -> int:
    DATA_PATH = Path("qa_test/test_data/relevance_2_data.json")
    CONTEXT_PATH = Path("qa_test/results/all_naive_rag_2.json")
    OUTPUT_PATH = Path("qa_test/results/relevance_2_chunk_analysis.json")
    MODEL_NAME = "gpt-4o-mini"
    LIMIT = None
    WORKERS = 2

    return run(
        data_path=DATA_PATH,
        context_path=CONTEXT_PATH,
        output_path=OUTPUT_PATH,
        model=MODEL_NAME,
        limit=LIMIT,
        workers=WORKERS,
    )


if __name__ == "__main__":
    main()

