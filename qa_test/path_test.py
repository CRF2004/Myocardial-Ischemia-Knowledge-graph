"""

"""

# main.py
import json
import os
import re
from tqdm import tqdm
from typing import Any, Dict, List, Tuple, Set
from collections import defaultdict, deque
from itertools import combinations
from typing import Optional
import tiktoken
enc = tiktoken.encoding_for_model("gpt-4o-mini")

from py2neo import Graph

import vector_search
from LLM_CALL import chat, parse_llm_response
import en_prompt


# ----------------------------
# Paths / Params
# ----------------------------
INPUT_FILES = [
    # "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_1_filtered.json",
    "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_2_filtered.json",
]
OUTPUT_DIR = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/results/"

FAISS_PATH = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/graph_embedding/knowledge_embeddings.faiss"
META_PATH = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/graph_embedding/knowledge_embeddings_metadata.pkl"

NEO4J_URI = "bolt://localhost:7687"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")

EMBED_SIM_THRESHOLD = 0.6  # 过滤低相似度 anchor
TOPK_ANCHOR_PER_ENTITY = 3  # 每个实体取 top-K 相似节点作为候选 anchor
MAX_ENTITIES = 15          # 每题最多用多少个实体做对齐
MAX_HOPS = 2             # 1/2/3-hop，按需修改
MAX_SUBGRAPH_LINES = 180   # 防止上下文爆炸

INTERSECTION_K = 1          # k-of-m intersection; 2 is usually safe
PAIR_TOPM_ANCHORS = 12      # only consider top-M anchors for pairing to avoid O(n^2)
TOPK_PAIRS = 30             # keep top-K anchor pairs for shortest path
MAX_PATH_LEN = 6            # max allowed shortest path length (in edges)
DIRECTED_PATH = False       # treat candidate graph as undirected for shortest-path BFS

# 优先保留的关系类型列表
PREFERRED_RELATION_TYPES = {
    "AFFECTS",
    "ASSOCIATED_WITH",
    "CAUSES",
    "DIAGNOSES",
    "INDICATES",
    # "IS_A",
    "IS_SPECIFIC_FOR",
    # "MAPPED_TO",
    "MEASURES",
    "PART_OF",
    "PRECEDES",
    "PREVENTS",
    # "RELATED_TO",
    "SUPPORTS",
    "TREATS",
}

# 兜底关系类型
FALLBACK_RELATION_TYPES = {"HAS_CHILD"}


# ----------------------------
# Utils
# ----------------------------
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _dump_json(path: str, obj: Any) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def _normalize_ws(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def _safe_get_options(item: Dict[str, Any]) -> Dict[str, str]:
    opts = item.get("options", {})
    if isinstance(opts, dict):
        return {str(k): _normalize_ws(str(v)) for k, v in opts.items()}
    if isinstance(opts, list):
        # tolerate [{"A": "..."}] like shapes
        out = {}
        for i, v in enumerate(opts):
            out[str(i)] = _normalize_ws(str(v))
        return out
    return {}


def build_prompt(item: Dict[str, Any], context: str) -> str:
    question = _normalize_ws(item.get("question", ""))
    options = _safe_get_options(item)
    ordered_keys = sorted(options.keys())

    options_lines = [f"{k}. {options[k]}" for k in ordered_keys]

    return "\n".join(
        [
            "Answer the following multiple-choice question using the structured context entries.",
            "Follow these steps carefully to generate the answer:",
            "1. **Understand the Question**: Start by carefully analyzing the question to understand what it is asking.",
            "2. **Analyze the Context**: Review each piece of context and determine if it is relevant to the question. If it is, consider how it can help answer the question.",
            "3. **Select Relevant Evidence**: Identify the most relevant chunks of context that help answer the question. If there is no relevant context, make a note that no evidence supports your answer.",
            "4. **Answer the Question**: Based on the relevant context, select the best answer from the options provided. If no strong evidence is available, choose the most likely answer based on the question alone.",
            "If the context does not contain helpful information, rely on your understanding of the question and select the most likely option.",
            "Cite chunk numbers as evidence in the JSON (e.g., 'evidence': ['Chunk 2', 'Chunk 5']) where applicable.",
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
            '{"answer": "A/B/C...", "evidence": ["Chunk 1"]}',
            "Use the single best option letter as the value. If no evidence supports a clear answer, select the option you believe is most likely.",
        ]
    )


# ----------------------------
# Vector search adapter
# ----------------------------
def get_searcher():
    """
    Try to adapt to your vector_search module.
    Expected: vector_search.VectorSearcher(faiss_path=..., metadata_path=...)
    """
    if hasattr(vector_search, "VectorSearcher"):
        return vector_search.VectorSearcher(faiss_path=FAISS_PATH, metadata_path=META_PATH)

    # If you exposed a singleton or factory, edit here:
    raise RuntimeError("vector_search module has no VectorSearcher. Please expose it or edit get_searcher().")


# ----------------------------
# Entity extraction (LLM)
# ----------------------------
def _map_ontology_type(ontology_type: str) -> str:
    """
    Map ontology_type from prompt.py to legacy type format.
    """
    mapping = {
        "Symptom_Sign": "symptom",
        "ECG_Finding": "test",
        "Biomarker": "test",
        "Diagnostic_Test": "test",
        "Disease": "disease",
        "Treatment": "treatment",
        "Risk_Factor": "disease",
    }
    return mapping.get(ontology_type, "disease")


def extract_entities_llm(question: str, options: Dict[str, str]) -> List[Dict[str, str]]:
    """
    Returns: [{"text": "...", "type": "disease|symptom|test|treatment"}, ...]
    """
    q = _normalize_ws(question)
    opts = "\n".join([f"{k}. {_normalize_ws(v)}" for k, v in sorted(options.items())])
    question_text = f"Question:\n{q}\n\nOptions:\n{opts}"

    prompt = en_prompt.query_understanding_prompt.format(question_text=question_text)
    s = chat("gpt-4o-mini", prompt)
    data = parse_llm_response(s)
    
    if data is None:
        print("error parsing entity extraction LLM output")
        return []
    
    ents = data.get("entities", [])
    out = []
    for e in ents:
        if not isinstance(e, dict):
            continue
        name = _normalize_ws(str(e.get("name", "")))
        ontology_type = _normalize_ws(str(e.get("ontology_type", "")))
        if not name:
            continue
        ty = _map_ontology_type(ontology_type)
        out.append({"text": name, "type": ty})
    # de-dup by text
    seen = set()
    dedup = []
    for e in out:
        if e["text"].lower() in seen:
            continue
        seen.add(e["text"].lower())
        dedup.append(e)
    return dedup[:15]


# ----------------------------
# Neo4j subgraph
# ----------------------------
def connect_graph() -> Graph:
    return Graph(uri=NEO4J_URI, user=NEO4J_USER, password=NEO4J_PASSWORD)

def get_anchor_neighborhood(
    g: Graph,
    anchor_id: int,
    max_hops: int,
) -> Tuple[Set[int], Dict[Tuple[int, str, int], Dict[str, Any]]]:
    """
    Fetch n-hop neighborhood edges for ONE anchor and return:
      - nodes: set(node_id)
      - edges_map: {(src, rel_type, dst): edge_info}
    We pull both preferred and fallback relations, but later intersection/filter decides what survives.
    """
    max_hops = max(1, int(max_hops))

    # We keep the query broad, then filter relation types in python (same as your old logic).
    cypher = f"""
    MATCH p=(a)-[r*1..{max_hops}]-(b)
    WHERE id(a) = $anchor_id
    UNWIND relationships(p) AS rel
    RETURN id(startNode(rel)) AS s, type(rel) AS t, id(endNode(rel)) AS o,
           rel.source_sentences AS source_sentences,
           rel.evidence_text AS evidence_text,
           rel.reason AS reason
    """
    rows = g.run(cypher, anchor_id=int(anchor_id)).data()

    edges_map: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    nodes: Set[int] = {int(anchor_id)}

    for row in rows:
        s, t, o = row.get("s"), row.get("t"), row.get("o")
        if s is None or o is None or not t:
            continue
        s, o, t = int(s), int(o), str(t)

        # Relation filtering: keep preferred; if none later, fallback will still exist in map.
        if t not in PREFERRED_RELATION_TYPES and t not in FALLBACK_RELATION_TYPES:
            continue

        key = (s, t, o)
        if key not in edges_map:
            edges_map[key] = {
                "src_id": s,
                "rel_type": t,
                "dst_id": o,
                "source_sentences": row.get("source_sentences"),
                "evidence_text": row.get("evidence_text"),
                "reason": row.get("reason"),
            }
        nodes.add(s)
        nodes.add(o)

    return nodes, edges_map


def intersect_neighborhoods_k_of_m(
    anchors: List[Dict[str, Any]],
    neighborhoods: Dict[int, Tuple[Set[int], Dict[Tuple[int, str, int], Dict[str, Any]]]],
    k: int,
) -> Tuple[Set[int], Dict[Tuple[int, str, int], Dict[str, Any]]]:
    """
    k-of-m intersection over nodes and edges:
      keep node if it appears in >=k anchor neighborhoods (support>=k)
      keep edge if it appears in >=k neighborhoods AND both endpoints survive

    Always keep all anchor nodes.
    For edges: if multiple anchors contribute same edge, keep the first edge_info (they should be identical props).
    """
    k = max(1, int(k))
    node_support = defaultdict(int)
    edge_support = defaultdict(int)

    # Accumulate supports
    for aid, (n_nodes, n_edges_map) in neighborhoods.items():
        for nid in n_nodes:
            node_support[int(nid)] += 1
        for ekey in n_edges_map.keys():
            edge_support[ekey] += 1

    # Nodes meeting support
    keep_nodes = {nid for nid, c in node_support.items() if c >= k}

    # Always keep anchors
    anchor_ids = {int(a["node_id"]) for a in anchors}
    keep_nodes |= anchor_ids

    # Edges meeting support, with endpoints kept
    keep_edges: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    for ekey, c in edge_support.items():
        if c < k:
            continue
        s, t, o = ekey
        if s not in keep_nodes or o not in keep_nodes:
            continue

        # pick one copy of edge_info
        # (iterate anchors until found)
        for aid, (_n_nodes, n_edges_map) in neighborhoods.items():
            if ekey in n_edges_map:
                keep_edges[ekey] = n_edges_map[ekey]
                break

    # If we kept zero preferred edges, but there are fallback edges with enough support, we already kept them.
    # If everything is empty, caller can decide to relax k or increase hops.
    return keep_nodes, keep_edges


# ----------------------------
# Pairing: choose which anchor pairs to connect
# ----------------------------
def select_topk_anchor_pairs(
    anchors: List[Dict[str, Any]],
    topm: int = PAIR_TOPM_ANCHORS,
    topk_pairs: int = TOPK_PAIRS,
) -> List[Tuple[int, int]]:
    """
    Choose anchor pairs to run shortest-path on.
    Strategy: take top-M anchors by similarity, score pairs by sim_i + sim_j, keep top-K.
    """
    if len(anchors) < 2:
        return []
    topm = max(2, min(int(topm), len(anchors)))
    topk_pairs = max(1, int(topk_pairs))

    a_sorted = sorted(anchors, key=lambda x: float(x.get("similarity", 0.0)), reverse=True)[:topm]
    ids = [int(a["node_id"]) for a in a_sorted]
    sim = {int(a["node_id"]): float(a.get("similarity", 0.0)) for a in a_sorted}

    scored = []
    for u, v in combinations(ids, 2):
        scored.append((sim[u] + sim[v], u, v))
    scored.sort(reverse=True)

    pairs = [(u, v) for _s, u, v in scored[:topk_pairs]]
    return pairs


# ----------------------------
# Shortest path on candidate subgraph (Python BFS)
# ----------------------------
def build_undirected_adj(
    edges_map: Dict[Tuple[int, str, int], Dict[str, Any]]
) -> Dict[int, List[Tuple[int, Tuple[int, str, int]]]]:
    """
    Build adjacency: node -> list of (neighbor, edge_key).
    Treat each directed edge as undirected for path search (unless DIRECTED_PATH=True later).
    """
    adj = defaultdict(list)
    for (s, t, o), _info in edges_map.items():
        adj[s].append((o, (s, t, o)))
        adj[o].append((s, (s, t, o)))
    return adj


def bfs_shortest_path_nodes(
    adj: Dict[int, List[Tuple[int, Tuple[int, str, int]]]],
    src: int,
    dst: int,
    max_len: int,
) -> Optional[List[int]]:
    """
    Return one shortest path (node id list) from src to dst within max_len edges.
    BFS on unweighted graph.
    """
    if src == dst:
        return [src]
    max_len = max(1, int(max_len))
    q = deque()
    q.append(src)
    prev = {src: None}
    depth = {src: 0}

    while q:
        u = q.popleft()
        du = depth[u]
        if du >= max_len:
            continue
        for v, _ekey in adj.get(u, []):
            if v in prev:
                continue
            prev[v] = u
            depth[v] = du + 1
            if v == dst:
                # reconstruct
                path = [dst]
                cur = dst
                while prev[cur] is not None:
                    cur = prev[cur]
                    path.append(cur)
                path.reverse()
                return path
            q.append(v)
    return None


def path_nodes_to_edges_with_dir(
    path_nodes: List[int],
    edges_map: Dict[Tuple[int, str, int], Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """
    Convert node path [n0,n1,...] to a list of edge_info dicts with direction.
    Because the BFS adjacency is undirected, for each step (u,v) we try:
      - direct edge (u,*,v)
      - otherwise reverse edge (v,*,u) and mark reversed=True
    Return list of edge dicts, each edge dict contains:
      src_id, dst_id, rel_type, reversed(bool), and evidence fields.
    """
    if not path_nodes or len(path_nodes) < 2:
        return []

    # Build quick index for O(1) lookups by endpoints
    # endpoints_index[(s,o)] -> list of edge_infos (could be multiple rel types)
    endpoints_index = defaultdict(list)
    for (s, t, o), info in edges_map.items():
        endpoints_index[(int(s), int(o))].append(info)

    out = []
    for u, v in zip(path_nodes[:-1], path_nodes[1:]):
        picked = None
        reversed_dir = False

        # try u -> v
        cand = endpoints_index.get((u, v), [])
        if cand:
            # if multiple, prefer preferred-rel types first, then stable by rel_type
            cand_sorted = sorted(
                cand,
                key=lambda x: (0 if x["rel_type"] in PREFERRED_RELATION_TYPES else 1, x["rel_type"])
            )
            picked = dict(cand_sorted[0])
        else:
            # try v -> u
            cand = endpoints_index.get((v, u), [])
            if cand:
                cand_sorted = sorted(
                    cand,
                    key=lambda x: (0 if x["rel_type"] in PREFERRED_RELATION_TYPES else 1, x["rel_type"])
                )
                picked = dict(cand_sorted[0])
                reversed_dir = True

        if picked is not None:
            picked["reversed"] = reversed_dir
            out.append(picked)

    return out


def get_k_hop_edges(g: Graph, anchor_ids: List[int], max_hops: int, anchors: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """
    Return unique directed edges with their properties from paths within hop range.
    Returns both outgoing and incoming edges for anchor nodes.

    Implements relation filtering:
    - First, try to get edges from PREFERRED_RELATION_TYPES
    - If none found, fallback to FALLBACK_RELATION_TYPES (HAS_CHILD)

    If no edges found and anchors info is provided, try to find fallback edges by lowering similarity threshold.

    Each edge includes source_sentences, evidence_text, reason if available.
    """
    if not anchor_ids:
        return []
    max_hops = max(1, int(max_hops))

    # 获取所有边和属性
    cypher = f"""
    MATCH p=(a)-[r*1..{max_hops}]-(b)
    WHERE id(a) IN $anchor_ids
    UNWIND relationships(p) AS rel
    RETURN id(startNode(rel)) AS s, type(rel) AS t, id(endNode(rel)) AS o,
           rel.source_sentences AS source_sentences,
           rel.evidence_text AS evidence_text,
           rel.reason AS reason
    """
    rows = g.run(cypher, anchor_ids=anchor_ids).data()

    # 收集所有边（去重）
    all_edges: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    for row in rows:
        s, t, o = row.get("s"), row.get("t"), row.get("o")
        if s is None or o is None or not t:
            continue
        key = (int(s), str(t), int(o))
        if key not in all_edges:
            all_edges[key] = {
                "src_id": int(s),
                "rel_type": str(t),
                "dst_id": int(o),
                "source_sentences": row.get("source_sentences"),
                "evidence_text": row.get("evidence_text"),
                "reason": row.get("reason"),
            }

    # 分离优先关系和兜底关系
    preferred_edges = []
    fallback_edges = []
    other_edges = []
    for edge in all_edges.values():
        if edge["rel_type"] in PREFERRED_RELATION_TYPES:
            preferred_edges.append(edge)
        elif edge["rel_type"] in FALLBACK_RELATION_TYPES:
            fallback_edges.append(edge)
        else:
            other_edges.append(edge)

    # 选择策略：如果有优先关系，使用优先关系；否则使用兜底关系
    if preferred_edges:
        return preferred_edges
    elif fallback_edges:
        return fallback_edges

    # 如果没有找到任何边，且提供了锚点信息，尝试降低相似度阈值
    if not fallback_edges and anchors is not None:
        # 构建当前锚点ID到相似度的映射
        anchor_sim_map = {int(a["node_id"]): float(a.get("similarity", 0.0)) for a in anchors}

        # 找出相似度大于阈值的候选节点（这些在原始选择时被保留）
        high_sim_anchors = [aid for aid, sim in anchor_sim_map.items() if sim >= EMBED_SIM_THRESHOLD]

        if high_sim_anchors:
            # 尝试查询这些高相似度节点的HAS_CHILD连接
            fallback_cypher = f"""
            MATCH (a)-[r:HAS_CHILD*1..{max_hops}]-(b)
            WHERE id(a) IN $anchor_ids
            RETURN id(startNode(r)) AS s, 'HAS_CHILD' AS t, id(endNode(r)) AS o,
                   r.source_sentences AS source_sentences,
                   r.evidence_text AS evidence_text,
                   r.reason AS reason
            """
            fallback_rows = g.run(fallback_cypher, anchor_ids=anchor_ids).data()

            # 收集HAS_CHILD边
            for row in fallback_rows:
                s, o = row.get("s"), row.get("o")
                if s is None or o is None:
                    continue
                key = (int(s), "HAS_CHILD", int(o))
                if key not in all_edges:
                    all_edges[key] = {
                        "src_id": int(s),
                        "rel_type": "HAS_CHILD",
                        "dst_id": int(o),
                        "source_sentences": row.get("source_sentences"),
                        "evidence_text": row.get("evidence_text"),
                        "reason": row.get("reason"),
                    }
                    fallback_edges.append(all_edges[key])

            if fallback_edges:
                print(f"[INFO] No preferred edges found, but found {len(fallback_edges)} HAS_CHILD edges for anchors with sim >= {EMBED_SIM_THRESHOLD}")
                return fallback_edges

    return []


def get_node_texts(g: Graph, node_ids: List[int]) -> Dict[int, str]:
    if not node_ids:
        return {}
    cypher = """
    MATCH (n)
    WHERE id(n) IN $ids
    RETURN id(n) AS id, coalesce(n.text, n.normalized_name, n.name_en, n.name, n.label, n.name_cn, toString(id(n))) AS label
    """
    rows = g.run(cypher, ids=list(set(node_ids))).data()
    return {int(r["id"]): _normalize_ws(str(r["label"])) for r in rows}


# ----------------------------
# Graph -> Text linearization
# ----------------------------
def linearize_subgraph(
        entities: List[Dict[str, str]],
        anchors: List[Dict[str, Any]],
        edges: List[Dict[str, Any]],
        node_text: Dict[int, str],
        max_lines: int = MAX_SUBGRAPH_LINES,
    ) -> str:
    """
    Produce chunked context. Each chunk: one anchor entity + its connected edges (filtered to subgraph set).
    Edges include supplementary information (source_sentences, evidence_text, reason).
    """
    # Build adjacency for fast grouping
    adj: Dict[int, List[Dict[str, Any]]] = {}
    for edge in edges:
        src_id = edge["src_id"]
        adj.setdefault(src_id, []).append(edge)

    # map anchor_id -> (entity_text, sim, anchor_text)
    anchor_blocks = []
    used_lines = 0
    chunk_id = 1

    for a in anchors:
        aid = int(a["node_id"])
        atext = node_text.get(aid, _normalize_ws(str(a.get("text", aid))))
        ent_text = _normalize_ws(str(a.get("query_entity", "")))
        sim = float(a.get("similarity", 0.0))

        lines = []
        lines.append(f"[Chunk {chunk_id}] [Anchor] {ent_text}  =>  {atext}  (sim={sim:.3f}, id={aid})")

        neighs = adj.get(aid, [])
        # sort neighbors: stable, small-to-large
        for edge in sorted(neighs, key=lambda x: (x["rel_type"], x["dst_id"])):
            rel = edge["rel_type"]
            nid = edge["dst_id"]
            ntext = node_text.get(nid, str(nid))

            # 构建基本关系输出
            rel_line = f"  ├─ {rel} → {ntext} (id={nid})"

            # 附加补充信息
            evidence_parts = []
            if edge.get("source_sentences"):
                evidence_parts.append(f"source: {_normalize_ws(str(edge['source_sentences']))}")
            if edge.get("evidence_text"):
                evidence_parts.append(f"evidence: {_normalize_ws(str(edge['evidence_text']))}")
            if edge.get("reason"):
                evidence_parts.append(f"reason: {_normalize_ws(str(edge['reason']))}")

            if evidence_parts:
                rel_line += f" [{'; '.join(evidence_parts)}]"

            lines.append(rel_line)

        # Also check for incoming edges (other nodes pointing to this anchor)
        for edge in edges:
            if edge["dst_id"] == aid:
                src_id = edge["src_id"]
                rel = edge["rel_type"]
                stext = node_text.get(src_id, str(src_id))

                # 构建基本关系输出
                rel_line = f"  ├─ {stext} (id={src_id}) → {rel} → [本节点]"

                # 附加补充信息
                evidence_parts = []
                if edge.get("source_sentences"):
                    evidence_parts.append(f"source: {_normalize_ws(str(edge['source_sentences']))}")
                if edge.get("evidence_text"):
                    evidence_parts.append(f"evidence: {_normalize_ws(str(edge['evidence_text']))}")
                if edge.get("reason"):
                    evidence_parts.append(f"reason: {_normalize_ws(str(edge['reason']))}")

                if evidence_parts:
                    rel_line += f" [{'; '.join(evidence_parts)}]"

                lines.append(rel_line)

        if len(lines) == 1:
            # lines.append("  ├─ (no linked edges in retrieved subgraph)")
            lines.pop() # remove anchor line
            chunk_id -= 1
            pass

        if used_lines + len(lines) > max_lines:
            break

        anchor_blocks.append("\n".join(lines))
        used_lines += len(lines)
        chunk_id += 1

    # If no blocks, still provide minimal context
    if not anchor_blocks:
        return "[Chunk 1] (No graph evidence retrieved.)"

    return "\n\n".join(anchor_blocks)

def linearize_paths(
    paths: List[Dict[str, Any]],
    node_text: Dict[int, str],
    max_lines: int = MAX_SUBGRAPH_LINES,
) -> str:
    """
    Each chunk = one shortest path between two anchors.
    Format:
      [Chunk i] [Path] A (id=..) -> REL -> B (id=..) -> REL -> C ...
      - edge evidence lines indented under each hop (optional)
    """
    if not paths:
        return "[Chunk 1] (No graph evidence retrieved.)"

    lines_used = 0
    chunks = []
    chunk_id = 1

    # simple ranking: shorter paths first, then more preferred edges, then stable
    def _path_score(p):
        pedges = p.get("path_edges", [])
        L = len(pedges)
        pref = sum(1 for e in pedges if e.get("rel_type") in PREFERRED_RELATION_TYPES)
        return (L, -pref)

    paths_sorted = sorted(paths, key=_path_score)

    for p in paths_sorted:
        pedges = p.get("path_edges", [])
        if not pedges:
            continue

        # build the main path line (compact)
        # reconstruct hop-by-hop node string using edge direction
        hop_str_parts = []
        # start node = first node in path_nodes
        pn = p.get("path_nodes", [])
        if not pn:
            continue

        # Use pn for node order; display edges with possibly reversed direction
        # For each hop i: show "Node(u) -[REL]-> Node(v)" respecting reversed flag
        # If reversed=True, the stored edge is v->u in graph, but in path order u->v,
        # so we render as "Node(u) <-[REL]- Node(v)" to be faithful.
        main_line = f"[Chunk {chunk_id}] [Path] "
        main_line += f"{node_text.get(pn[0], str(pn[0]))} (id={pn[0]})"
        for i, e in enumerate(pedges):
            u = pn[i]
            v = pn[i + 1]
            rel = str(e.get("rel_type", "RELATED_TO"))
            rev = bool(e.get("reversed", False))
            if not rev:
                main_line += f"  —{rel}→  {node_text.get(v, str(v))} (id={v})"
            else:
                main_line += f"  ←{rel}—  {node_text.get(v, str(v))} (id={v})"

        chunk_lines = [main_line]

        # add evidence per hop as sub-lines (only if exists)
        for i, e in enumerate(pedges):
            rel = str(e.get("rel_type", "RELATED_TO"))
            evidence_parts = []
            if e.get("source_sentences"):
                evidence_parts.append(f"source: {_normalize_ws(str(e['source_sentences']))}")
            if e.get("evidence_text"):
                evidence_parts.append(f"evidence: {_normalize_ws(str(e['evidence_text']))}")
            if e.get("reason"):
                evidence_parts.append(f"reason: {_normalize_ws(str(e['reason']))}")
            if evidence_parts:
                chunk_lines.append(f"  ├─ ({i+1}) {rel} [{'; '.join(evidence_parts)}]")

        if lines_used + len(chunk_lines) > max_lines:
            break

        chunks.append("\n".join(chunk_lines))
        lines_used += len(chunk_lines)
        chunk_id += 1

    return "\n\n".join(chunks) if chunks else "[Chunk 1] (No graph evidence retrieved.)"

# ----------------------------
# End-to-end per item
# ----------------------------
def retrieve_evidence(
        g: Graph,
        searcher,
        question: str,
        options: Dict[str, str],
    ) -> Tuple[str, Dict[str, Any]]:
    paths: List[Dict[str, Any]] = []
    success = 0 
    entities = extract_entities_llm(question, options)[:MAX_ENTITIES]
    anchors: List[Dict[str, Any]] = []
    anchor_ids: List[int] = []

    # --- Batch anchor retrieval using batch_search_with_triples ---
    # 收集所有实体查询文本，批量检索以提高效率
    entity_queries = [e["text"] for e in entities]
    
    # 批量检索所有实体的相似节点
    batch_results = searcher.batch_search_with_triples(entity_queries, top_k=TOPK_ANCHOR_PER_ENTITY)
    
    # 处理批量检索结果
    for i, e in enumerate(entities):
        q = e["text"]
        sr = batch_results[i]  # 获取当前实体的检索结果
        for r in sr.get("results", []):
            sim = float(r.get("similarity", r.get("score", 0.0)) or 0.0)
            if sim < EMBED_SIM_THRESHOLD:
                continue
            nid = int(r["node_id"])
            anchors.append(
                {
                    "query_entity": q,
                    "query_type": e["type"],
                    "node_id": nid,
                    "similarity": sim,
                    "text": r.get("text", ""),
                }
            )
            anchor_ids.append(nid)

    # de-dup anchors by node_id, keep best sim
    best_by_id: Dict[int, Dict[str, Any]] = {}
    for a in anchors:
        nid = int(a["node_id"])
        if nid not in best_by_id or float(a["similarity"]) > float(best_by_id[nid]["similarity"]):
            best_by_id[nid] = a
    anchors = sorted(best_by_id.values(), key=lambda x: float(x["similarity"]), reverse=True)
    anchor_ids = [int(a["node_id"]) for a in anchors]

    # If too few anchors, fallback to old n-hop behavior (optional but practical)
    if len(anchor_ids) < 2:
        edges = get_k_hop_edges(g, anchor_ids, MAX_HOPS, anchors)
        node_ids = set(anchor_ids)
        for edge in edges:
            node_ids.add(int(edge["src_id"]))
            node_ids.add(int(edge["dst_id"]))
        node_text = get_node_texts(g, list(node_ids))
        context = linearize_subgraph(entities, anchors, edges, node_text)
        debug = {
            "entities": entities,
            "anchors": anchors[:30],
            "anchor_count": len(anchors),
            "edge_count": len(edges),
            "mode": "fallback_n_hop",
            "path_success": success,
            "path_count": len(paths),
        }
        return context, debug

    # --- Step 1: per-anchor neighborhoods ---
    neighborhoods: Dict[int, Tuple[Set[int], Dict[Tuple[int, str, int], Dict[str, Any]]]] = {}
    for aid in anchor_ids:
        n_nodes, n_edges_map = get_anchor_neighborhood(g, aid, MAX_HOPS)
        neighborhoods[aid] = (n_nodes, n_edges_map)

    # --- Step 2: intersection pruning (k-of-m) ---
    keep_nodes, keep_edges_map = intersect_neighborhoods_k_of_m(
        anchors=anchors,
        neighborhoods=neighborhoods,
        k=INTERSECTION_K,
    )

    # If intersection is too strict, relax once (pragmatic)
    relaxed = False
    if len(keep_edges_map) == 0:
        # relax k -> 1 (i.e., union) to avoid empty context
        relaxed = True
        keep_nodes, keep_edges_map = intersect_neighborhoods_k_of_m(
            anchors=anchors,
            neighborhoods=neighborhoods,
            k=1,
        )

    # --- Step 3: shortest paths inside candidate subgraph ---
    adj = build_undirected_adj(keep_edges_map)
    pairs = select_topk_anchor_pairs(anchors, topm=PAIR_TOPM_ANCHORS, topk_pairs=TOPK_PAIRS)

    path_edges_union: Dict[Tuple[int, str, int], Dict[str, Any]] = {}
    path_nodes_union: Set[int] = set(anchor_ids)

    # NEW: store actual path instances for presentation
    paths: List[Dict[str, Any]] = []

    success = 0
    for u, v in pairs:
        pn = bfs_shortest_path_nodes(adj, u, v, MAX_PATH_LEN)
        if pn is None:
            continue

        pedges = path_nodes_to_edges_with_dir(pn, keep_edges_map)
        if not pedges:
            continue

        success += 1

        # record path instance (node list + edge list)
        paths.append({
            "src_anchor": int(u),
            "dst_anchor": int(v),
            "path_nodes": [int(x) for x in pn],
            "path_edges": pedges,  # list of edge dicts with reversed flag
        })

        for nid in pn:
            path_nodes_union.add(int(nid))

        # keep union edges as before (dedup)
        for e in pedges:
            key = (int(e["src_id"]), str(e["rel_type"]), int(e["dst_id"]))
            path_edges_union[key] = e

    edges = list(path_edges_union.values())


    # --- Fetch node texts for linearization ---
    node_ids = set(path_nodes_union)
    for edge in edges:
        node_ids.add(int(edge["src_id"]))
        node_ids.add(int(edge["dst_id"]))
    node_text = get_node_texts(g, list(node_ids))

    context = linearize_subgraph(entities, anchors, edges, node_text)

    debug = {
        "entities": entities,
        "anchors": anchors[:30],
        "anchor_count": len(anchors),
        "max_hops": MAX_HOPS,
        "intersection_k": INTERSECTION_K,
        "intersection_relaxed_to_union": relaxed,
        "candidate_nodes": len(keep_nodes),
        "candidate_edges": len(keep_edges_map),
        "pairs": len(pairs),
        "path_success": success,
        "final_nodes": len(node_ids),
        "final_edges": len(edges),
        "mode": "nhop_intersection_shortestpath",
    }
    context = linearize_paths(paths, node_text)

    return context, debug


def answer_one(item: Dict[str, Any], g: Graph, searcher) -> Dict[str, Any]:
    q = item.get("question", "")
    options = _safe_get_options(item)

    context, debug = retrieve_evidence(g, searcher, q, options)
    prompt = build_prompt(item, context)
    # print("########")
    # print(debug)
    # print("########")
    # print(prompt)
    # return  ##########
    model_name = "gpt-4o-mini"
    print(item.get("q_id"), "->", len(enc.encode(prompt)), "tokens")
    raw = chat(model_name, prompt)

    pred = {"answer": None, "evidence": []}
    try:
        pred = parse_llm_response(raw)
    except Exception:
        # keep raw
        pass

    # 判断答案是否正确
    gold_answer = item.get("answer")
    pred_answer = pred.get("answer") if pred else None
    is_correct = (gold_answer is not None and pred_answer is not None and 
                  str(gold_answer).strip().upper() == str(pred_answer).strip().upper())

    return {
        "q_id": item.get("q_id"),
        "method": "n_hop",
        "model": model_name,
        "answer": pred_answer,
        "is_correct": is_correct,
        "context": context,
        "retrieval_meta": {
            "embed_threshold": EMBED_SIM_THRESHOLD,
            "max_hops": MAX_HOPS,
            "max_entities": MAX_ENTITIES,
            "topk_anchor_per_entity": TOPK_ANCHOR_PER_ENTITY,
            "anchor_count": debug.get("anchor_count", 0),
            "edge_count": debug.get("edge_count", 0),
        },
        "generation_meta": {
            "model": model_name,
            "prompt_version": "n_hop_v1",
        },
        "raw_response": raw or None,
        "debug": debug,
    }


def main():
    g = connect_graph()
    searcher = get_searcher()

    for in_path in INPUT_FILES:
        data = _load_json(in_path)
        if not isinstance(data, list):
            raise ValueError(f"Expected a list in {in_path}")

        out = []
        for item in tqdm(data):
            if not isinstance(item, dict):
                continue
            out.append(answer_one(item, g, searcher))
            # return  ##########
        out_path = os.path.join(OUTPUT_DIR, os.path.basename(in_path).replace(".json", "_path_test_results.json"))
        _dump_json(out_path, out)
        print(f"[OK] wrote: {out_path}  (n={len(out)})")


if __name__ == "__main__":
    main()
