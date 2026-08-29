import os
"""
N0 未定义类型节点
N1 已定义类型节点
N2 本体节点
A) 歧义消除：与现有的已定义类型的知识节点进行比对
    （已定义类型节点指：拥有 :Knowledge 标签 + ontology_id 属性 + 至少一个其他类型标签的节点）
    使用 FAISS 相似度 + LLM 决策。
    如果 LLM 返回 MAPPED_TO / INSTANTIATES：
        将 N1 的 ontology_id 复制到 N0
        将 N1 的类型标签添加到 N0
        根据决策结果创建相应关系，并附带原因 reason 和相似度分数 sim_score
    如果返回 IS_A：
        创建 IS_A 关系，附带原因 reason 和相似度分数 sim_score（不强制复制 ontology_id）
    如果返回 NONE：
        进入本体对齐步骤

B) 本体对齐：
    从同一个 FAISS 索引中检索 topK 个候选 Ontology 节点，由 LLM 选择 BEST 或 NONE
    如果选择 BEST（N2）：
        设置 N0.ontology_id = N2.ontology_id
        在 N0 上添加对应的类型标签（5种类型之一）
        创建/合并关系 (N0)-[:HAS_ONTOLOGY]->(N2)
    如果选择 NONE：
        仍然为 N0 分配一个临时的 ontology_id，格式必须为：TEMP_<type>_<neo4j_id>
        （type 必须是以下之一：diseases/symptoms/treatments/examinations/patient_characteristics）
        在 N0 上添加对应的类型标签
        不创建 HAS_ONTOLOGY 关系
"""
import json
import re
from typing import Any, Dict, List, Optional, Tuple

import faiss
import pickle
from py2neo import Graph

import LLM_CALL

# =========================
# Config (edit here)
# =========================

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

FAISS_INDEX_PATH = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/graph_embedding/knowledge_embeddings.faiss"
FAISS_META_PATH = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/graph_embedding/knowledge_embeddings_metadata.pkl"

TOPK_KNOWLEDGE = 30
TOPK_ONTOLOGY = 50
SIM_THRESHOLD_KNOWLEDGE = 0.70
SIM_THRESHOLD_ONTOLOGY = 0.65

# Whether to always create SIMILAR_TO edges for inspected candidates
CREATE_SIMILAR_TO_EDGES = False

# Safety: cap number of N0 processed in one run (None = all)
LIMIT_N0 = None

# Relationship names (keep your originals)
REL_SIMILAR_TO = "SIMILAR_TO"
REL_MAPPED_TO = "MAPPED_TO"
REL_IS_A = "IS_A"
REL_INSTANTIATES = "INSTANTIATES"
REL_HAS_ONTOLOGY = "HAS_ONTOLOGY"

# Five ontology types required by you
ONTO_TYPES = [
    "diseases",
    "symptoms",
    "treatments",
    "examinations",
    "patient_characteristics",
]

# Map ontology_type -> Knowledge node extra label (adjust to your real label taxonomy if needed)
TYPE_LABEL_MAP = {
    "diseases": "Disease",
    "symptoms": "Symptoms",
    "treatments": "Treatment",
    "examinations": "DiagnosticTest",
    "patient_characteristics": "PatientCharacteristics",
}

# =========================
# Utilities
# =========================


def _load_faiss_and_meta(index_path: str, meta_path: str):
    index = faiss.read_index(index_path)
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)
    node_ids: List[int] = meta["node_ids"]
    texts: List[str] = meta["texts"]
    dim: int = meta["dimension"]

    if index.d != dim:
        raise ValueError(f"FAISS dim mismatch: index.d={index.d} meta.dimension={dim}")

    # Build node_id -> faiss_row_index
    id_to_row = {nid: i for i, nid in enumerate(node_ids)}
    return index, node_ids, texts, id_to_row, dim


def _safe_parse_json(llm_text: str) -> Optional[Dict[str, Any]]:
    """
    Extract and parse the first JSON object found in the LLM output.
    """
    if not llm_text:
        return None

    # Try direct parse first
    try:
        return json.loads(llm_text)
    except Exception:
        pass

    # Try to extract first {...}
    m = re.search(r"\{.*\}", llm_text, flags=re.DOTALL)
    if not m:
        return None
    blob = m.group(0)
    try:
        return json.loads(blob)
    except Exception:
        return None


def _normalize_sim(score: float) -> float:
    # IndexFlatIP with L2-normalized vectors => cosine similarity in [-1, 1]
    # Keep as float for properties.
    return float(score)


def _get_node_text_props(n: Dict[str, Any]) -> str:
    # Prefer normalized_name if present; else name; else fallback.
    return (n.get("normalized_name") or n.get("name") or "").strip()


def _labels_of_node(graph: Graph, node_id: int) -> List[str]:
    q = "MATCH (n) WHERE id(n)=$id RETURN labels(n) AS labels"
    r = graph.run(q, id=node_id).data()
    if not r:
        return []
    return r[0]["labels"] or []


def _add_labels(graph: Graph, node_id: int, labels: List[str]) -> None:
    labels = [lab for lab in labels if lab and isinstance(lab, str)]
    if not labels:
        return
    # dynamic labels must be appended with SET n:Label
    # Do it in one query using apoc if available; if not, do iterative.
    # Here, no APOC assumption: iterative.
    for lab in labels:
        graph.run(f"MATCH (n) WHERE id(n)=$id SET n:`{lab}`", id=node_id)


def _set_props(graph: Graph, node_id: int, props: Dict[str, Any]) -> None:
    if not props:
        return
    q = "MATCH (n) WHERE id(n)=$id SET n += $props"
    graph.run(q, id=node_id, props=props)


def _merge_relationship(
    graph: Graph,
    src_id: int,
    rel_type: str,
    dst_id: int,
    props: Optional[Dict[str, Any]] = None,
) -> None:
    props = props or {}
    # MERGE relationship; then set/overwrite props
    q = f"""
    MATCH (a) WHERE id(a)=$src
    MATCH (b) WHERE id(b)=$dst
    MERGE (a)-[r:{rel_type}]->(b)
    SET r += $props
    """
    graph.run(q, src=src_id, dst=dst_id, props=props)


def _transfer_relationships_and_delete_node(graph: Graph, from_id: int, to_id: int) -> None:
    """
    Merge-like refactor:
    - Move all relationships (incoming and outgoing) from `from_id` to `to_id`.
    - Delete the original node `from_id` afterwards.

    Notes:
    - Relationship types and properties are preserved (MERGE to avoid duplicates).
    - Properties on the destination node are NOT modified.
    """

    # Outgoing: (from)-[r]->(x)  ==>  (to)-[r]->(x)
    q_out = """
    MATCH (a)-[r]->(b)
    WHERE id(a)=$from_id AND id(b)<>$to_id
    RETURN id(r) AS rid, type(r) AS rtype, properties(r) AS props, id(b) AS other_id
    """
    out_rels = graph.run(q_out, from_id=from_id, to_id=to_id).data() or []
    for row in out_rels:
        rid = row.get("rid")
        rtype = row.get("rtype")
        props = row.get("props") or {}
        other_id = row.get("other_id")
        if not rtype or not isinstance(other_id, int):
            continue
        _merge_relationship(graph, to_id, rtype, other_id, props=props)
        if isinstance(rid, int):
            graph.run("MATCH ()-[r]->() WHERE id(r)=$rid DELETE r", rid=rid)

    # Incoming: (x)-[r]->(from)  ==>  (x)-[r]->(to)
    q_in = """
    MATCH (a)-[r]->(b)
    WHERE id(b)=$from_id AND id(a)<>$to_id
    RETURN id(r) AS rid, type(r) AS rtype, properties(r) AS props, id(a) AS other_id
    """
    in_rels = graph.run(q_in, from_id=from_id, to_id=to_id).data() or []
    for row in in_rels:
        rid = row.get("rid")
        rtype = row.get("rtype")
        props = row.get("props") or {}
        other_id = row.get("other_id")
        if not rtype or not isinstance(other_id, int):
            continue
        _merge_relationship(graph, other_id, rtype, to_id, props=props)
        if isinstance(rid, int):
            graph.run("MATCH ()-[r]->() WHERE id(r)=$rid DELETE r", rid=rid)

    # Finally, remove the node itself (and any remaining relationships)
    graph.run("MATCH (n) WHERE id(n)=$id DETACH DELETE n", id=from_id)


def _get_untyped_knowledge_nodes(graph: Graph, limit: Optional[int]) -> List[Dict[str, Any]]:
    q = """
    MATCH (n:Knowledge)
    WHERE size(labels(n)) = 1
    RETURN id(n) AS id, n.name AS name, n.normalized_name AS normalized_name
    ORDER BY id(n)
    """
    if isinstance(limit, int) and limit > 0:
        q += " LIMIT $limit"
        return graph.run(q, limit=limit).data()
    return graph.run(q).data()


def _get_typed_knowledge_candidate_info(graph: Graph, node_id: int) -> Optional[Dict[str, Any]]:
    """
    Typed Knowledge definition (as you described):
    - has :Knowledge label
    - has ontology_id property (string)
    - has other type labels (>=2 labels total)
    """
    q = """
    MATCH (n:Knowledge)
    WHERE id(n)=$id
    RETURN id(n) AS id,
           n.name AS name,
           n.normalized_name AS normalized_name,
           n.ontology_id AS ontology_id,
           labels(n) AS labels
    """
    r = graph.run(q, id=node_id).data()
    if not r:
        return None
    row = r[0]
    labels = row.get("labels") or []
    onto_id = row.get("ontology_id")
    if onto_id is None or onto_id == "":
        return None
    if len(labels) <= 1:
        return None
    return row


def _get_ontology_candidate_info(graph: Graph, node_id: int) -> Optional[Dict[str, Any]]:
    q = """
    MATCH (n:Ontology)
    WHERE id(n)=$id
    RETURN id(n) AS id,
           n.ontology_id AS ontology_id,
           n.name_en AS name_en,
           n.name_cn AS name_cn,
           labels(n) AS labels
    """
    r = graph.run(q, id=node_id).data()
    if not r:
        return None
    row = r[0]
    if not row.get("ontology_id"):
        return None
    return row


def _extract_type_labels(labels: List[str]) -> List[str]:
    """
    From candidate typed Knowledge labels, keep only the ontology-type-related ones
    in your five buckets (based on TYPE_LABEL_MAP).
    """
    keep = set(TYPE_LABEL_MAP.values())
    return [lab for lab in labels if lab in keep]


# =========================
# LLM Prompts
# =========================


def build_llm_disambiguation_prompt(
    n0: Dict[str, Any],
    candidates: List[Dict[str, Any]],
) -> str:
    """
    Ask LLM to decide relation between N0 and each candidate Knowledge N1.
    Output JSON only, in English.
    """
    n0_text = _get_node_text_props(n0) or f"node_{n0['id']}"

    cand_lines = []
    for c in candidates:
        c_text = (c.get("normalized_name") or c.get("name") or "").strip()
        cand_lines.append(
            {
                "id": c["id"],
                "text": c_text,
                "ontology_id": c.get("ontology_id"),
                "type_labels": _extract_type_labels(c.get("labels") or []),
                "sim_score": c.get("sim_score"),
            }
        )

    schema = {
        "decisions": [
            {
                "candidate_id": 123,
                "decision": "MAPPED_TO | IS_A | INSTANTIATES | NONE",
                "reason": "English short reason",
            }
        ]
    }

    prompt = f"""
    You are a medical knowledge graph entity disambiguation expert.

    Target node N0 (untyped Knowledge, only :Knowledge label):
    - id: {n0['id']}
    - text: "{n0_text}"

    Candidate typed Knowledge nodes N1 (already aligned to ontology, have ontology_id and type labels):
    {json.dumps(cand_lines, ensure_ascii=False, indent=2)}

    Task:
    For EACH candidate, decide the best relation between N0 and that candidate.

    Decision set (choose exactly one per candidate):
    - MAPPED_TO: N0 is the same clinical concept as N1 (synonym/alias). If chosen, N0 should be merged into N1 (delete N0 and transfer its relationships to N1).
    - IS_A: N0 is a subtype / more specific concept than N1.
    - INSTANTIATES: N0 is an instance/example of N1 (rare; use only if clearly instance vs class).
    - NONE: no valid semantic alignment.

    Output requirements:
    - Output JSON only (no markdown, no extra text).
    - All strings must be in English, including "reason".
    - Follow this JSON schema exactly:
    {json.dumps(schema, ensure_ascii=False, indent=2)}
    """
    return prompt.strip()


def build_llm_ontology_alignment_prompt(
    n0: Dict[str, Any],
    onto_candidates: List[Dict[str, Any]],
) -> str:
    """
    Ask LLM to pick best Ontology node N2 or NONE, and ALWAYS provide ontology_type (one of five).
    """
    n0_text = _get_node_text_props(n0) or f"node_{n0['id']}"

    cand_lines = []
    for c in onto_candidates:
        cand_lines.append(
            {
                "id": c["id"],
                "ontology_id": c["ontology_id"],
                "name_en": (c.get("name_en") or "").strip(),
                "name_cn": (c.get("name_cn") or "").strip(),
                "labels": c.get("labels") or [],
                "sim_score": c.get("sim_score"),
            }
        )

    schema = {
        "ontology_type": "diseases | symptoms | treatments | examinations | patient_characteristics",
        "selected": {"ontology_node_id": 123, "ontology_id": "D001.001", "reason": "English short reason"}
        # or selected = null (if NONE)
    }

    prompt = f"""
    You are a medical ontology alignment expert.

    Target node N0 (untyped Knowledge, only :Knowledge label):
    - id: {n0['id']}
    - text: "{n0_text}"

    Ontology candidates N2 (each is (:Ontology) with ontology_id):
    {json.dumps(cand_lines, ensure_ascii=False, indent=2)}

    Task:
    1) Determine the correct ontology_type for N0. Must be exactly one of:
    {", ".join(ONTO_TYPES)}
    2) Select the single best matching Ontology node (N2) if there is a clear match; otherwise choose NONE.

    Output requirements:
    - Output JSON only (no markdown, no extra text).
    - All strings must be in English, including "reason".
    - If no match, set "selected" to null.
    - Follow this JSON schema exactly:
    {json.dumps(schema, ensure_ascii=False, indent=2)}
    """
    return prompt.strip()


# =========================
# Core logic
# =========================


def faiss_search_by_node_id(
    index,
    id_to_row: Dict[int, int],
    query_node_id: int,
    topk: int,
) -> Tuple[List[int], List[float]]:
    """
    Use the stored vector of query_node_id from IndexFlatIP, then search.
    Returns (neighbor_node_ids, sim_scores).
    """
    if query_node_id not in id_to_row:
        return [], []

    row = id_to_row[query_node_id]
    vec = index.reconstruct(row).reshape(1, -1)
    # vec already normalized (stored); safe anyway:
    faiss.normalize_L2(vec)

    sims, idxs = index.search(vec, topk)
    sims = sims[0].tolist()
    idxs = idxs[0].tolist()

    return idxs, sims


def build_knowledge_candidates(
    graph: Graph,
    faiss_row_idxs: List[int],
    sim_scores: List[float],
    node_ids: List[int],
    n0_id: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    out = []
    for row_i, sim in zip(faiss_row_idxs, sim_scores):
        if row_i < 0:
            continue
        cand_id = node_ids[row_i]
        if cand_id == n0_id:
            continue
        sim = _normalize_sim(sim)
        if sim < threshold:
            continue

        info = _get_typed_knowledge_candidate_info(graph, cand_id)
        if not info:
            continue
        info["sim_score"] = sim
        out.append(info)
    # sort by sim desc
    out.sort(key=lambda x: x.get("sim_score", 0.0), reverse=True)
    return out


def build_ontology_candidates(
    graph: Graph,
    faiss_row_idxs: List[int],
    sim_scores: List[float],
    node_ids: List[int],
    n0_id: int,
    threshold: float,
) -> List[Dict[str, Any]]:
    out = []
    for row_i, sim in zip(faiss_row_idxs, sim_scores):
        if row_i < 0:
            continue
        cand_id = node_ids[row_i]
        if cand_id == n0_id:
            continue
        sim = _normalize_sim(sim)
        if sim < threshold:
            continue
        info = _get_ontology_candidate_info(graph, cand_id)
        if not info:
            continue
        info["sim_score"] = sim
        out.append(info)
    out.sort(key=lambda x: x.get("sim_score", 0.0), reverse=True)
    return out



def apply_disambiguation_result(
    graph: Graph,
    n0_id: int,
    n0: Dict[str, Any],
    candidates: List[Dict[str, Any]],
    llm_json: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Apply disambiguation decisions.

    Returns:
      - {"action": "MERGED", "into_id": <N1 id>} if N0 is merged into an existing typed Knowledge node.
      - {"action": "TYPED", "typed": True} if N0 is typed via INSTANTIATES (inherit ontology_id + type labels).
      - {"action": "UNTYPED"} if no typing/merge happened (all NONE / only IS_A without typing).
    """
    decisions = llm_json.get("decisions", [])
    if not isinstance(decisions, list) or not decisions:
        return {"action": "UNTYPED"}

    cand_by_id = {c["id"]: c for c in candidates}

    typed = False

    for d in decisions:
        if not isinstance(d, dict):
            continue
        cand_id = d.get("candidate_id")
        decision_raw = (d.get("decision") or "").strip().upper()
        reason = (d.get("reason") or "").strip()
        if cand_id not in cand_by_id:
            continue

        cand = cand_by_id[cand_id]
        sim = cand.get("sim_score")

        # Optional: always keep SIMILAR_TO edges for audit
        if CREATE_SIMILAR_TO_EDGES:
            _merge_relationship(
                graph,
                n0_id,
                REL_SIMILAR_TO,
                cand_id,
                props={"sim_score": sim},
            )

        # Accept both "MERGE" and legacy "MAPPED_TO" as semantic equivalence
        if decision_raw == "MERGE":
            decision = "MAPPED_TO"
        else:
            decision = decision_raw

        if decision not in {"MAPPED_TO", "IS_A", "INSTANTIATES", "NONE"}:
            decision = "NONE"

        if decision == "NONE":
            continue

        # If semantic equivalent: merge node (delete N0, transfer relationships to N1)
        if decision == "MAPPED_TO":
            _transfer_relationships_and_delete_node(graph, from_id=n0_id, to_id=cand_id)
            return {"action": "MERGED", "into_id": cand_id, "reason": reason, "sim_score": sim}

        # Otherwise, keep explicit semantic edges
        rel_type = {
            "IS_A": REL_IS_A,
            "INSTANTIATES": REL_INSTANTIATES,
        }[decision]

        _merge_relationship(
            graph,
            n0_id,
            rel_type,
            cand_id,
            props={"reason": reason, "sim_score": sim},
        )

        # For disambiguation, if it's INSTANTIATES, we "type" N0 by inheriting ontology_id + type labels
        if decision == "INSTANTIATES":
            onto_id = cand.get("ontology_id")
            type_labels = _extract_type_labels(cand.get("labels") or [])
            if onto_id:
                _set_props(graph, n0_id, {"ontology_id": onto_id})
                if type_labels:
                    _add_labels(graph, n0_id, type_labels)
                typed = True

    return {"action": "TYPED", "typed": True} if typed else {"action": "UNTYPED"}


def apply_ontology_alignment_result(
    graph: Graph,
    n0_id: int,
    llm_json: Dict[str, Any],
) -> bool:
    """
    Apply ontology alignment. Return True if typed, else False.
    """
    onto_type = (llm_json.get("ontology_type") or "").strip().lower()
    if onto_type not in ONTO_TYPES:
        # If LLM fails, do not proceed
        return False

    type_label = TYPE_LABEL_MAP.get(onto_type)
    selected = llm_json.get("selected", None)

    if selected is None:
        # NONE: assign TEMP id and label
        temp_onto_id = f"TEMP_{onto_type}_{n0_id}"
        _set_props(graph, n0_id, {"ontology_id": temp_onto_id})
        if type_label:
            _add_labels(graph, n0_id, [type_label])
        return True

    if not isinstance(selected, dict):
        return False

    onto_node_id = selected.get("ontology_node_id")
    onto_id = (selected.get("ontology_id") or "").strip()
    reason = (selected.get("reason") or "").strip()

    if not isinstance(onto_node_id, int) or not onto_id:
        # fall back to TEMP
        temp_onto_id = f"TEMP_{onto_type}_{n0_id}"
        _set_props(graph, n0_id, {"ontology_id": temp_onto_id})
        if type_label:
            _add_labels(graph, n0_id, [type_label])
        return True

    # Type N0 and connect to Ontology
    _set_props(graph, n0_id, {"ontology_id": onto_id})
    if type_label:
        _add_labels(graph, n0_id, [type_label])

    _merge_relationship(
        graph,
        n0_id,
        REL_HAS_ONTOLOGY,
        onto_node_id,
        props={"reason": reason},
    )

    return True


def process_one_node(
    graph: Graph,
    index,
    node_ids: List[int],
    id_to_row: Dict[int, int],
    n0: Dict[str, Any],
) -> Dict[str, Any]:
    """
    Returns a status dict for reporting.
    """
    n0_id = n0["id"]

    # 1) Knowledge disambiguation
    faiss_rows, sims = faiss_search_by_node_id(index, id_to_row, n0_id, TOPK_KNOWLEDGE)
    knowledge_cands = build_knowledge_candidates(
        graph, faiss_rows, sims, node_ids, n0_id, SIM_THRESHOLD_KNOWLEDGE
    )

    if knowledge_cands:
        prompt = build_llm_disambiguation_prompt(n0, knowledge_cands)
        llm_text = LLM_CALL.chat("gpt-4o-mini", prompt)
        llm_json = _safe_parse_json(llm_text) or {}

        disamb_res = apply_disambiguation_result(graph, n0_id, n0, knowledge_cands, llm_json)
        if disamb_res.get("action") == "MERGED":
            return {
                "n0_id": n0_id,
                "status": "MERGED_BY_KNOWLEDGE_DISAMBIGUATION",
                "merged_into_id": disamb_res.get("into_id"),
                "knowledge_candidates": len(knowledge_cands),
            }
        if disamb_res.get("action") == "TYPED" and disamb_res.get("typed"):
            return {
                "n0_id": n0_id,
                "status": "TYPED_BY_KNOWLEDGE_DISAMBIGUATION",
                "knowledge_candidates": len(knowledge_cands),
            }

    # 2) Ontology alignment (either no knowledge candidates or all NONE)
    faiss_rows2, sims2 = faiss_search_by_node_id(index, id_to_row, n0_id, TOPK_ONTOLOGY)
    onto_cands = build_ontology_candidates(graph, faiss_rows2, sims2, node_ids, n0_id, SIM_THRESHOLD_ONTOLOGY)

    prompt2 = build_llm_ontology_alignment_prompt(n0, onto_cands[:20])  # keep prompt compact
    llm_text2 = LLM_CALL.chat("gpt-4o-mini", prompt2)
    llm_json2 = _safe_parse_json(llm_text2) or {}

    typed2 = apply_ontology_alignment_result(graph, n0_id, llm_json2)
    return {
        "n0_id": n0_id,
        "status": "TYPED_BY_ONTOLOGY_ALIGNMENT" if typed2 else "FAILED",
        "ontology_candidates": len(onto_cands),
    }


def main():
    graph = Graph(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

    index, node_ids, texts, id_to_row, dim = _load_faiss_and_meta(FAISS_INDEX_PATH, FAISS_META_PATH)
    print(f"[info] Loaded FAISS index with {len(node_ids)} nodes, dimension={dim}")
    
    n0_list = _get_untyped_knowledge_nodes(graph, LIMIT_N0)

    stats = {
        "total_untyped": len(n0_list),
        "merged_by_knowledge": 0,
        "typed_by_knowledge": 0,
        "typed_by_ontology": 0,
        "failed": 0,
    }

    for i, n0 in enumerate(n0_list, start=1):
        res = process_one_node(graph, index, node_ids, id_to_row, n0)

        if res["status"] == "MERGED_BY_KNOWLEDGE_DISAMBIGUATION":
            stats["merged_by_knowledge"] += 1
        elif res["status"] == "TYPED_BY_KNOWLEDGE_DISAMBIGUATION":
            stats["typed_by_knowledge"] += 1
        elif res["status"] == "TYPED_BY_ONTOLOGY_ALIGNMENT":
            stats["typed_by_ontology"] += 1
        else:
            stats["failed"] += 1

        if i % 50 == 0:
            print(f"[progress] {i}/{len(n0_list)} stats={stats}")

    print("[done] stats=", stats)


if __name__ == "__main__":
    main()