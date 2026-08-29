#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os

import json
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from py2neo import Graph


JSON_PATH = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph/ecgfinding_alignment/ptbxl_scp_to_book_ecgfinding.json")

# Neo4j 连接信息（按需修改）
NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

BATCH_SIZE = 500

def sanitize_rel_type(raw: Optional[str]) -> str:
    """
    将任意字符串转为合法 Neo4j Relationship Type：
    - 转大写
    - 非 [A-Z0-9_] 替换为 _
    - 合并连续 _
    - 不能以数字开头；必要时加前缀 REL_
    - 为空则用 RELATED_TO
    """
    if not raw:
        return "RELATED_TO"
    s = raw.strip().upper()
    s = re.sub(r"[^A-Z0-9_]", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    if not s:
        return "RELATED_TO"
    if re.match(r"^[0-9]", s):
        s = f"REL_{s}"
    return s


def build_rel_props(mapping: Dict[str, Any]) -> Dict[str, Any]:
    """
    关系属性：relation_type, reason
    不包括：generated_at, top_k, rank, similarity
    """
    props: Dict[str, Any] = {}

    if "relation_type" in mapping:
        props["relation_type"] = mapping["relation_type"]
    if "reason" in mapping:
        props["reason"] = mapping["reason"]

    return props

def load_json(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON 顶层不是 dict：{type(data)}")
    if "items" not in data or not isinstance(data["items"], list):
        raise ValueError("JSON 缺少 items 或 items 不是 list")
    return data





def ingest(graph: Graph, data: Dict[str, Any]) -> None:
    items: List[Dict[str, Any]] = data["items"]

    total_rels = 0
    skipped_items = 0
    skipped_mappings = 0

    tx = graph.begin()
    in_tx = 0

    for item in items:
        scp_code = item.get("scp_code")
        label_text = item.get("label_text")
        statement_category = item.get("statement_category")
        mappings = item.get("mappings")

        if not scp_code or not label_text or not statement_category or not isinstance(mappings, list) or len(mappings) == 0:
            skipped_items += 1
            continue

        for m in mappings:
            node_text = m.get("node_text")

            if not node_text:
                skipped_mappings += 1
                continue

            rel_props = build_rel_props(m)

            # 创建 FindingLabel 节点
            # 创建 ECGFinding 节点（使用MERGE，避免重复）
            # 创建关系 ECGFinding -[RELATED_TO]-> FindingLabel
            cypher = """
            MERGE (b:FindingLabel {scp_code: $scp_code})
            SET b.label_text = $label_text
            SET b.statement_category = $statement_category
            MERGE (a:ECGFinding {name: $ecg_name})
            MERGE (a)-[r:RELATED_TO]->(b)
            SET r += $props
            """
            tx.run(
                cypher,
                scp_code=scp_code,
                label_text=label_text,
                statement_category=statement_category,
                ecg_name=node_text,
                props=rel_props
            )

            total_rels += 1
            in_tx += 1

            if in_tx >= BATCH_SIZE:
                tx.commit()
                tx = graph.begin()
                in_tx = 0
                print(
                    f"[进度] 已写入关系 {total_rels} 条；"
                    f"跳过 items {skipped_items}；跳过 mappings {skipped_mappings}..."
                )

    tx.commit()
    print(
        f"[完成] 写入关系 {total_rels} 条；"
        f"跳过 items {skipped_items}；跳过 mappings {skipped_mappings}；"
        f"总 items 输入 {len(items)} 条。"
    )


def main() -> None:
    if not JSON_PATH.exists():
        raise FileNotFoundError(f"找不到 JSON 文件：{JSON_PATH}")

    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    data = load_json(JSON_PATH)
    ingest(graph, data)


if __name__ == "__main__":
    main()