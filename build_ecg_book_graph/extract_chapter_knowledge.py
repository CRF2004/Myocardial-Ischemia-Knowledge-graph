# 代码流程概览:
# 1) 读取章节JSON与抽取Prompt模板。
# 2) 将段落拆分为句子，再按MAX_CHARS合并为块，兼顾上下文与长度限制。
# 3) 对每个块调用LLM抽取‘心电图所见 -> 知识实体’关系。
# 4) 从LLM回复中解析JSON数组，规范字段并剔除不完整项。
# 5) 基于(所见, 实体, 关系, 证据)去重，保证高召回同时避免重复。
# 6) 追加来源与定位信息(来源URL/章节标题/段落与块索引)，输出为JSON数组。

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import LLM_CALL

BASE_DIR = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph")
INPUT_PATH = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/get_ecg_book_data/chapter_text.json")
PROMPT_PATH = BASE_DIR / Path("extraction_prompt_chapter.md")
OUTPUT_PATH = BASE_DIR / Path("chapter_extraction_results.json")

MODEL_NAME = "gpt-4o-mini"
MAX_CHARS = 1200


def load_prompt_template() -> str:
    return PROMPT_PATH.read_text(encoding="utf-8")

def sentence_split(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [part.strip() for part in parts if part.strip()]

def chunk_sentences(sentences: list[str], max_chars: int) -> list[str]:
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0
    for sentence in sentences:
        add_len = len(sentence) + (1 if current else 0)
        if current and current_len + add_len > max_chars:
            chunks.append(" ".join(current))
            current = [sentence]
            current_len = len(sentence)
        else:
            current.append(sentence)
            current_len += add_len
    if current:
        chunks.append(" ".join(current))
    return chunks


def build_blocks(sections: list[dict[str, object]]) -> list[dict[str, object]]:
    blocks: list[dict[str, object]] = []
    for section_index, section in enumerate(sections):
        title = section.get("title")
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list):
            continue
        section_title = title if isinstance(title, str) else f"Section {section_index + 1}"
        for paragraph_index, paragraph in enumerate(paragraphs):
            if not isinstance(paragraph, str):
                continue
            text = paragraph.strip()
            if not text:
                continue
            sentences = sentence_split(text)
            if not sentences:
                continue
            chunked = chunk_sentences(sentences, MAX_CHARS)
            for chunk_index, chunk in enumerate(chunked):
                blocks.append({
                    "section_title": section_title,
                    "paragraph_index": paragraph_index,
                    "chunk_index": chunk_index,
                    "text": chunk,
                })
    return blocks


def extract_json_array(text: str) -> list[dict[str, object]] | None:
    start = text.find("[")
    end = text.rfind("]")
    if start == -1 or end == -1 or end <= start:
        return None
    json_str = text[start:end + 1]
    try:
        data = json.loads(json_str)
    except json.JSONDecodeError:
        return None
    if isinstance(data, list):
        return [item for item in data if isinstance(item, dict)]
    return None


def make_prompt(template: str, input_text: str) -> str:
    return template + f"\n## input: {input_text} \n## output [your output]"


def call_llm(prompt: str) -> str:
    return LLM_CALL.chat(MODEL_NAME, prompt)


def iter_sections(data: dict[str, object]) -> Iterable[dict[str, object]]:
    sections = data.get("sections")
    if isinstance(sections, list):
        for section in sections:
            if isinstance(section, dict):
                yield section


def main() -> None:

    data = json.loads(INPUT_PATH.read_text(encoding="utf-8"))
    template = load_prompt_template()

    blocks = build_blocks(list(iter_sections(data)))
    results: list[dict[str, object]] = []
    seen: set[tuple[str, str, str, str]] = set()

    for index, block in enumerate(blocks, start=1):
        text = block["text"]
        prompt = make_prompt(template, text)
        response = call_llm(prompt)
        extracted = extract_json_array(response)
        if not extracted:
            continue
        for item in extracted:
            ecg_finding = str(item.get("ecg_finding", "")).strip()
            knowledge_entity = str(item.get("knowledge_entity", "")).strip()
            relation = str(item.get("relation", "")).strip()
            evidence_text = str(item.get("evidence_text", "")).strip()
            if not (ecg_finding and knowledge_entity and relation and evidence_text):
                continue
            key = (ecg_finding, knowledge_entity, relation, evidence_text)
            if key in seen:
                continue
            seen.add(key)
            results.append({
                "ecg_finding": ecg_finding,
                "knowledge_entity": knowledge_entity,
                "relation": relation,
                "evidence_text": evidence_text,
                "source": data.get("source", ""),
                "section_title": block.get("section_title", ""),
                "paragraph_index": block.get("paragraph_index", 0),
                "chunk_index": block.get("chunk_index", 0),
            })

        if index % 10 == 0:
            print(f"Processed {index}/{len(blocks)} blocks")

    OUTPUT_PATH.write_text(json.dumps(results, ensure_ascii=True, indent=2), encoding="utf-8")
    print(f"Wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
