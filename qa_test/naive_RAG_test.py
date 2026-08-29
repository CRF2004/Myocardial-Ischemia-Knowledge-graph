"""
Naive RAG baseline: retrieve top-k chunks from FAISS, then answer with LLM.
"""

from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

try:
    import faiss  # type: ignore
except Exception as exc:  # pragma: no cover - optional dependency
    faiss = None
    _FAISS_IMPORT_ERROR = exc
else:
    _FAISS_IMPORT_ERROR = None

from LLM_CALL import get_embedding


Choice = str


def load_questions(path: Path) -> List[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, list):
        raise ValueError(f"Expected list of questions in {path}")
    return data


def load_index_and_meta(index_path: Path, meta_path: Path) -> Tuple[Any, Dict[str, Any]]:
    if _FAISS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "faiss is required to run RAG retrieval. "
            "Install faiss-cpu or faiss-gpu before running."
        ) from _FAISS_IMPORT_ERROR
    if not index_path.exists():
        raise FileNotFoundError(f"index not found: {index_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"metadata not found: {meta_path}")
    index = faiss.read_index(str(index_path))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    return index, meta


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


def build_prompt(item: Dict[str, Any], context: str) -> str:
    question = item.get("question", "").strip()
    options = item.get("options", {})
    if not isinstance(options, dict):
        raise ValueError("Expected options to be a dict of choice -> text")

    ordered_keys = sorted(options.keys())
    options_lines = [f"{key}. {options[key]}" for key in ordered_keys]

    prompt = "\n".join(
        [
            "Answer the following multiple-choice question using the structured context entries.",
            "The context may include irrelevant information. Use only evidence directly related to the question.",
            "Cite chunk numbers as evidence in the JSON (e.g., 'evidence': ['Chunk 2', 'Chunk 5']).",
            "",
            "Context:",
            context.strip(),
            "",
            "Question:",
            question,
            "",
            "Options:",
            *options_lines,
            "",
            "Reply in JSON only, using this schema:",
            '{"answer": "A", "evidence": ["Chunk 1"]}',
            "Use the single best option letter as the value.",
        ]
    )
    return prompt


def retrieve_top_k(
    index: Any,
    meta: Dict[str, Any],
    query: str,
    top_k: int,
    model: str,
) -> List[Dict[str, Any]]:
    embedding = get_embedding(model, query)
    vector = np.asarray([embedding], dtype="float32")
    faiss.normalize_L2(vector)
    scores, indices = index.search(vector, top_k)

    items = meta.get("items", [])
    results: List[Dict[str, Any]] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(items):
            continue
        entry = dict(items[idx])
        entry["score"] = float(score)
        results.append(entry)
    return results


_SECTION_RE = re.compile(r"^\s*(?:\u7ae0\u8282|section)\s*[:\uff1a]\s*(.+)$", re.IGNORECASE | re.MULTILINE)


def _extract_section(text: str) -> str:
    match = _SECTION_RE.search(text)
    if not match:
        return "unknown"
    return match.group(1).strip()


def assemble_context(chunks: List[Dict[str, Any]]) -> str:
    lines: List[str] = []
    for rank, chunk in enumerate(chunks, start=1):
        text = chunk.get("text", "")
        file_name = chunk.get("file", "unknown")
        score = chunk.get("score")
        section = _extract_section(text)
        score_text = f"{score:.4f}" if isinstance(score, (float, int)) else "unknown"
        header = f"[Chunk {rank} | {file_name} | {section} | {score_text}]"
        lines.append(f"{header}\n{text}")
    return "\n\n".join(lines)


def choose_answer(
    item: Dict[str, Any],
    model: str,
    top_k: int,
    index: Any,
    meta: Dict[str, Any],
    embedding_model: str,
) -> Dict[str, Any]:
    options = item.get("options", {})
    option_keys = sorted(options.keys())
    query = f"{item.get('question', '')}\n\nOptions:\n" + "\n".join(
        f"{key}. {options[key]}" for key in option_keys
    )
    retrieved = retrieve_top_k(index, meta, query, top_k, embedding_model)
    context = assemble_context(retrieved)
    prompt = build_prompt(item, context)
    raw_response = call_llm(model, prompt)
    answer = parse_choice(raw_response, option_keys)
    if answer is None:
        answer = option_keys[0] if option_keys else None
    return {
        "answer": answer,
        "raw_response": raw_response,
        "context": context,
        "retrieved": retrieved,
    }


def run(
    data_path: Path,
    index_path: Path,
    meta_path: Path,
    output_path: Optional[Path],
    model: str,
    embedding_model: str,
    top_k: int,
    limit: Optional[int],
    workers: int,
) -> int:
    data = load_questions(data_path)
    if limit is not None:
        data = data[:limit]

    index, meta = load_index_and_meta(index_path, meta_path)

    def process_item(index_id: int, item: Dict[str, Any]) -> Tuple[int, Dict[str, Any]]:
        result = choose_answer(item, model, top_k, index, meta, embedding_model)
        expected = item.get("answer")
        record = {
            "q_id": item.get("q_id"),
            "method": "naive_rag",
            "model": model,
            "answer": result["answer"],
            "is_correct": result["answer"] == expected if expected else None,
            "context": {
                "type": "chunk",
                "chunks": result["retrieved"],
                "linearized_text": result["context"],
            },
            "retrieval_meta": {
                "top_k": top_k,
                "num_chunks": len(result["retrieved"]),
                "index_path": str(index_path),
            },
            "generation_meta": {
                "model": model,
                "prompt_version": "naive_rag_v2",
            },
            "raw_response": result["raw_response"] or None,
        }
        return index_id, record

    results: List[Optional[Dict[str, Any]]] = [None] * len(data)
    total = len(data)
    processed = 0

    def report_progress() -> None:
        print(f"Progress: {processed}/{total}", end="\r", flush=True)

    if workers <= 1 or len(data) <= 1:
        for idx, item in enumerate(data):
            _, record = process_item(idx, item)
            results[idx] = record
            processed += 1
            report_progress()
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(process_item, idx, item)
                for idx, item in enumerate(data)
            ]
            for future in as_completed(futures):
                idx, record = future.result()
                results[idx] = record
                processed += 1
                report_progress()

    if total:
        print(f"Progress: {processed}/{total}")

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
    parser = argparse.ArgumentParser(description="Naive RAG baseline using chunk index.")
    parser.add_argument(
        "--data-path",
        default="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_1_filtered.json",
        help="QA dataset JSON file.",
    )
    parser.add_argument(
        "--index-path",
        default="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/index/chunk_index.faiss",
        help="FAISS index path.",
    )
    parser.add_argument(
        "--meta-path",
        default="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/index/chunk_meta.json",
        help="Chunk metadata path.",
    )
    parser.add_argument(
        "--output-path",
        default="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/results/naive_rag_1.json",
        help="Output results JSON file.",
    )
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--embedding-model", default="text-embedding-3-small")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()

    output_path = Path(args.output_path) if args.output_path else None
    return run(
        data_path=Path(args.data_path),
        index_path=Path(args.index_path),
        meta_path=Path(args.meta_path),
        output_path=output_path,
        model=args.model,
        embedding_model=args.embedding_model,
        top_k=args.top_k,
        limit=args.limit,
        workers=args.workers,
    )


if __name__ == "__main__":
    main()
