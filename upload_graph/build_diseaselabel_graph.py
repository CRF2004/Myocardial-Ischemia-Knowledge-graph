import os
"""
读取 /mnt/chengrongfeng_private/SRP心肌缺血知识图谱/disease_alignment/disease_label_to_node_multi.json
构建label到图谱关系
"""
import json
from typing import Any, Dict, List, Optional

from py2neo import Graph


def load_alignment_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def build_rows(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    把 JSON 转成适合 UNWIND 的行数据，每行对应一条边：
      (DiseaseLabel)-[:RELATED_TO {props}]->(ExistingNodeByInternalId)

    跳过 relation_type == "NONE"。
    """
    items = payload.get("items", []) or []
    rows: List[Dict[str, Any]] = []

    for item in items:
        disease = item.get("disease", {}) or {}
        label = (disease.get("label") or "").strip()
        snomed_id = (disease.get("snomed_id") or "").strip()

        if not label:
            continue

        for m in (item.get("mappings") or []):
            relation_type = (m.get("relation_type") or "").strip().upper()
            if relation_type == "NONE":
                continue

            node_id_raw = m.get("node_id", None)
            if node_id_raw is None or str(node_id_raw).strip() == "":
                continue

            # node_id 是 Neo4j 内部 id(n)
            try:
                node_internal_id = int(str(node_id_raw))
            except ValueError:
                continue

            rows.append(
                {
                    "label": label,
                    "snomed_id": snomed_id if snomed_id else None,
                    "target_internal_id": node_internal_id,
                    "edge_type": "RELATED_TO",  # 统一关系类型
                    "relation_type": relation_type,  # EXACT/BROADER/NARROWER/RELATED
                    "reason": (m.get("reason") or "").strip(),
                    "rank": m.get("rank", None),
                    "similarity": m.get("similarity", None),
                }
            )

    return rows


def upload_alignment_to_neo4j(
    graph: Graph,
    rows: List[Dict[str, Any]],
    dataset_label_label: str = "DiseaseLabel",
    create_missing_target_policy: str = "skip",
    batch_size: int = 1000,
) -> None:
    """
    create_missing_target_policy:
      - "skip": 找不到 target 节点就跳过（推荐）
      - "error": 找不到就抛错
    """
    if not rows:
        print("No rows to upload.")
        return
    
    cypher = f"""
    UNWIND $rows AS row
    // 1) 合并 DiseaseLabel
    MERGE (dl:{dataset_label_label} {{
    label: row.label,
    snomed_id: row.snomed_id
    }})
    ON CREATE SET
    dl.created_at = datetime(),
    dl.source = "model_2_alignment"
    SET
    dl.updated_at = datetime()

    // 关键：把 dl 和 row 传递到下一段
    WITH dl, row

    // 2) 用内部 id 找到目标节点
    MATCH (n)
    WHERE id(n) = row.target_internal_id

    // 3) 幂等建边并写属性
    MERGE (dl)-[r:RELATED_TO]->(n)
    SET
    r.relation_type = row.relation_type,
    r.reason = row.reason,
    r.rank = row.rank,
    r.similarity = row.similarity,
    r.updated_at = datetime()
    """

    # MERGE 更利于幂等导入（重复跑不产生重复边）。

    total = len(rows)
    print(f"Uploading {total} relationships...")

    for start in range(0, total, batch_size):
        batch = rows[start : start + batch_size]
        if create_missing_target_policy == "skip":
            # 当前写法：MATCH 找不到 n 会自动丢弃该行，不会报错
            graph.run(cypher, rows=batch)
        elif create_missing_target_policy == "error":
            # 强制检查 target 存在：先验证再写入
            missing = graph.run(
                """
                UNWIND $rows AS row
                OPTIONAL MATCH (n) WHERE id(n) = row.target_internal_id
                WITH row, n WHERE n IS NULL
                RETURN collect(row.target_internal_id) AS missing_ids
                """,
                rows=batch,
            ).evaluate()
            if missing:
                raise RuntimeError(f"Missing target nodes for internal ids: {missing[:20]} ... (total {len(missing)})")
            graph.run(cypher, rows=batch)
        else:
            raise ValueError("create_missing_target_policy must be 'skip' or 'error'")

        print(f"  Uploaded {min(start + batch_size, total)}/{total}")

    print("Done.")


def main():
    NEO4J_URI = "bolt://localhost:7687"
    NEO4J_USER = "neo4j"
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

    INPUT_JSON = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/disease_alignment/disease_label_to_node_multi.json"

    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    payload = load_alignment_json(INPUT_JSON)
    rows = build_rows(payload)

    print(f"Rows prepared (excluding NONE): {len(rows)}")
    if rows:
        # 看一下前几条
        print("Sample row:", rows[0])

    upload_alignment_to_neo4j(
        graph=graph,
        rows=rows,
        dataset_label_label="DiseaseLabel",
        create_missing_target_policy="skip",
        batch_size=1000,
    )


if __name__ == "__main__":
    main()
