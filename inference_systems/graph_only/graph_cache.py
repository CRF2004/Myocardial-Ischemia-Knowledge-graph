"""
Neo4j -> 本地导出，用于graph_only推理。

此模块仅导出推理所需的子图：
  1) ECGFinding -[RELATED_TO]-> FindingLabel (包含 relation_type)
  2) DiseaseLabel -[MAPPING_TO]-> Disease (包含 relation_type；我们捕获关系类型和属性)
  3) Disease <-> ECGFinding (任何用作证据的语义关系；我们捕获关系类型和属性)
  4) Disease <-> Disease (MAPPED_TO / IS_A / INSTANTIATES，或任何疾病间的关系)

默认输出为 Parquet（如果可用），否则为 CSV：
  exported_graph_v1/
    nodes_findinglabel.parquet|csv
    nodes_ecgfinding.parquet|csv
    nodes_diseaselabel.parquet|csv
    nodes_disease.parquet|csv
    edges_fl_to_ef.parquet|csv
    edges_ef_to_dis.parquet|csv
    edges_dl_to_dis.parquet|csv
    edges_dis_to_dis.parquet|csv
    id_maps.json

    {'n_findinglabel': 66, 
    'n_ecgfinding': 92, 
    'n_diseaselabel': 39, 
    'n_disease': 420, 
    'edges_ef_to_fl': 609, 
    'edges_ef_to_dis': 143, 
    'edges_dl_to_dis': 152, 
    'isa_edges': 0, 
    'syn_groups': 420, 
    'missing_vocab_labels': ['Left posterior fascicular block', 'Ventricular aneurysm', 'Aneurysm of heart'], 
    'relation_type_ef_fl_top': [('RELATED', 453), ('NARROWER', 81), ('EXACT', 53), ('BROADER', 22)], 
    'edge_type_ef_dis_top': [('INDICATES', 74), ('SUPPORTS', 52), ('IS_SPECIFIC_FOR', 17)], 
    'relation_type_dl_dis_top': [('RELATED', 106), ('EXACT', 25), ('NARROWER', 11), ('BROADER', 10)]}

用法 (CLI):
  python graph_cache.py --out_dir exported_graph_v1

设计说明：
- 我们使用 Neo4j 内部 id(n) 作为导出的稳定引用，
  然后为每种节点类型分配连续本地 id 以便快速索引。
- 关系属性保存在名为 "props_json" 的 JSON 列中。
- 此导出器有意设计为模式容忍：它尝试一些属性名
  (例如，relation_type) 并将其余所有属性捕获到 props_json 中。
"""

from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
from py2neo import Graph


# -------------------------
# IO helpers
# -------------------------

def _to_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _can_write_parquet() -> bool:
    try:
        import pyarrow  # noqa: F401
        return True
    except Exception:
        try:
            import fastparquet  # noqa: F401
            return True
        except Exception:
            return False


def _write_df(df: pd.DataFrame, out_path_no_ext: Path) -> Path:
    """
    Write df to parquet if possible, else CSV.
    Return the final written path.
    """
    if _can_write_parquet():
        out_path = out_path_no_ext.with_suffix(".parquet")
        df.to_parquet(out_path, index=False)
        return out_path
    out_path = out_path_no_ext.with_suffix(".csv")
    df.to_csv(out_path, index=False)
    return out_path


def _json_dumps_safe(x: Any) -> str:
    try:
        return json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        return json.dumps(str(x), ensure_ascii=False)


# -------------------------
# Neo4j access
# -------------------------

def connect_neo4j_from_env() -> Graph:
    uri = "bolt://localhost:7687"
    user = "neo4j"
    password = os.environ.get("NEO4J_PASSWORD", "")
    db = "neo4j"

    if not uri or not user or not password:
        raise RuntimeError(
            "Missing Neo4j connection env vars. Please set:\n"
            "  NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD (and optionally NEO4J_DATABASE)."
        )
    return Graph(uri, auth=(user, password), name=db)


def run_cypher_df(graph: Graph, cypher: str, params: Optional[Dict[str, Any]] = None) -> pd.DataFrame:
    params = params or {}
    return graph.run(cypher, parameters=params).to_data_frame()


# -------------------------
# Local id assignment
# -------------------------

def assign_local_ids(nodes_df: pd.DataFrame, neo4j_id_col: str = "neo4j_id") -> Tuple[pd.DataFrame, Dict[int, int]]:
    """
    Assign contiguous local_id based on neo4j_id ordering.
    Returns (nodes_df_with_local_id, map neo4j_id->local_id).
    """
    if neo4j_id_col not in nodes_df.columns:
        raise ValueError(f"nodes_df missing column '{neo4j_id_col}'")
    ids = nodes_df[neo4j_id_col].astype(int).tolist()
    ids_sorted = sorted(set(ids))
    mapping = {nid: i for i, nid in enumerate(ids_sorted)}
    nodes_df = nodes_df.copy()
    nodes_df["local_id"] = nodes_df[neo4j_id_col].astype(int).map(mapping).astype(int)
    # reorder columns
    cols = ["local_id"] + [c for c in nodes_df.columns if c != "local_id"]
    nodes_df = nodes_df[cols]
    return nodes_df, mapping


def remap_edge_endpoints(
    edges_df: pd.DataFrame,
    src_map: Dict[int, int],
    dst_map: Dict[int, int],
    src_col: str = "src_neo4j_id",
    dst_col: str = "dst_neo4j_id",
    ) -> pd.DataFrame:
    df = edges_df.copy()
    df["src"] = df[src_col].astype(int).map(src_map)
    df["dst"] = df[dst_col].astype(int).map(dst_map)
    missing_src = df["src"].isna().sum()
    missing_dst = df["dst"].isna().sum()
    if missing_src or missing_dst:
        # Drop edges that reference nodes outside exported node sets
        df = df.dropna(subset=["src", "dst"]).copy()
    df["src"] = df["src"].astype(int)
    df["dst"] = df["dst"].astype(int)
    # Put src/dst first
    cols = ["src", "dst"] + [c for c in df.columns if c not in ("src", "dst")]
    return df[cols]


# -------------------------
# Exporter
# -------------------------

@dataclass
class ExportedPaths:
    out_dir: Path
    nodes_findinglabel: Path
    nodes_ecgfinding: Path
    nodes_diseaselabel: Path
    nodes_disease: Path
    edges_fl_to_ef: Path
    edges_ef_to_dis: Path
    edges_dl_to_dis: Path
    edges_dis_to_dis: Path
    id_maps: Path


class GraphExporter:
    def __init__(self, graph: Graph):
        self.graph = graph

    # ---- Node queries ----

    def export_nodes_findinglabel(self) -> pd.DataFrame:
        cypher = """
        MATCH (n:FindingLabel)
        RETURN id(n) AS neo4j_id,
               n.name AS name,
               labels(n) AS labels,
               properties(n) AS props
        """
        df = run_cypher_df(self.graph, cypher)
        df["props_json"] = df["props"].apply(_json_dumps_safe)
        df = df.drop(columns=["props"])
        return df

    def export_nodes_ecgfinding(self) -> pd.DataFrame:
        cypher = """
        MATCH (n:ECGFinding)
        RETURN id(n) AS neo4j_id,
               n.name AS name,
               labels(n) AS labels,
               properties(n) AS props
        """
        df = run_cypher_df(self.graph, cypher)
        df["props_json"] = df["props"].apply(_json_dumps_safe)
        df = df.drop(columns=["props"])
        return df

    def export_nodes_diseaselabel(self) -> pd.DataFrame:
        # DiseaseLabel nodes have "label" property per your description
        cypher = """
        MATCH (n:DiseaseLabel)
        RETURN id(n) AS neo4j_id,
               coalesce(n.label, n.name) AS label,
               labels(n) AS labels,
               properties(n) AS props
        """
        df = run_cypher_df(self.graph, cypher)
        df["props_json"] = df["props"].apply(_json_dumps_safe)
        df = df.drop(columns=["props"])
        return df

    def export_nodes_disease(self) -> pd.DataFrame:
        cypher = """
        MATCH (n:Disease)
        RETURN id(n) AS neo4j_id,
               n.name AS name,
               n.ontology_id AS ontology_id,
               labels(n) AS labels,
               properties(n) AS props
        """
        df = run_cypher_df(self.graph, cypher)
        df["props_json"] = df["props"].apply(_json_dumps_safe)
        df = df.drop(columns=["props"])
        return df

    # ---- Edge queries ----

    def export_edges_fl_to_ef(self) -> pd.DataFrame:
        """
        Your description: direction is ECGFinding -> FindingLabel via RELATED_TO with relation_type.
        For reasoning we often want EF -> FL edges; we keep as-is (src=ECGFinding, dst=FindingLabel).
        Name is "edges_fl_to_ef" historically; contents are EF->FL.
        """
        cypher = """
        MATCH (ef:ECGFinding)-[r:RELATED_TO]->(fl:FindingLabel)
        RETURN id(ef) AS src_neo4j_id,
               id(fl) AS dst_neo4j_id,
               type(r) AS rel,
               coalesce(r.relation_type, r.type, r.kind) AS relation_type,
               properties(r) AS props
        """
        df = run_cypher_df(self.graph, cypher)
        df["props_json"] = df["props"].apply(_json_dumps_safe)
        df = df.drop(columns=["props"])
        return df

    def export_edges_dl_to_dis(self) -> pd.DataFrame:
        """
        DiseaseLabel -> Disease mapping edges (type may vary); capture all relationships between those.
        """
        cypher = """
        MATCH (dl:DiseaseLabel)-[r]->(d:Disease)
        RETURN id(dl) AS src_neo4j_id,
               id(d) AS dst_neo4j_id,
               type(r) AS rel,
               coalesce(r.relation_type, r.type, r.kind) AS relation_type,
               properties(r) AS props
        """
        df = run_cypher_df(self.graph, cypher)
        df["props_json"] = df["props"].apply(_json_dumps_safe)
        df = df.drop(columns=["props"])
        return df

    def export_edges_ef_to_dis(self) -> pd.DataFrame:
        """
        Evidence edges between ECGFinding and Disease.
        Since direction may vary in your graph (Disease->ECGFinding or vice versa),
        we export both directions and normalize to (ef -> disease) in output columns.
        """
        # EF -> Disease
        cypher_1 = """
        MATCH (ef:ECGFinding)-[r]->(d:Disease)
        RETURN id(ef) AS ef_neo4j_id,
            id(d) AS dis_neo4j_id,
            type(r) AS rel,
            properties(r) AS props
        """
        # Disease -> EF
        cypher_2 = """
        MATCH (d:Disease)-[r]->(ef:ECGFinding)
        RETURN id(ef) AS ef_neo4j_id,
            id(d) AS dis_neo4j_id,
            type(r) AS rel,
            properties(r) AS props
        """

        df1 = run_cypher_df(self.graph, cypher_1)
        df2 = run_cypher_df(self.graph, cypher_2)

        df = pd.concat([df1, df2], ignore_index=True)

        # IMPORTANT: deduplicate using hashable structural keys only
        # (props may contain dicts -> unhashable)
        if not df.empty:
            df = df.drop_duplicates(subset=["ef_neo4j_id", "dis_neo4j_id", "rel"]).reset_index(drop=True)

        # Try to preserve a compact "edge_type" if present; else use rel
        def _edge_type(props: Any, rel: str) -> str:
            if isinstance(props, dict):
                for k in ("edge_type", "relation", "predicate", "sem_type", "type"):
                    if k in props and props[k] is not None:
                        return str(props[k])
            return str(rel)

        df["edge_type"] = [
            _edge_type(p, r) for p, r in zip(df["props"], df["rel"])
        ]

        # Capture a likely evidence/source sentence field if present (kept inside props_json anyway)
        def _pick_evidence(props: Any) -> Optional[str]:
            if not isinstance(props, dict):
                return None
            for k in ("source_sentence", "evidence_sentence", "sentence", "source", "evidence_text"):
                if k in props and props[k]:
                    return str(props[k])
            return None

        df["evidence"] = df["props"].apply(_pick_evidence)
        df["props_json"] = df["props"].apply(_json_dumps_safe)

        df = df.rename(
            columns={
                "ef_neo4j_id": "src_neo4j_id",
                "dis_neo4j_id": "dst_neo4j_id",
            }
        )
        df = df.drop(columns=["props"])
        return df

    def export_edges_dis_to_dis(self) -> pd.DataFrame:
        """
        Disease <-> Disease relations:
        - If you explicitly used relationship types MAPPED_TO / IS_A / INSTANTIATES, they will appear in rel.
        - If you encoded them as a property (e.g., r.decision), they will appear inside props_json.
        We export all Disease->Disease edges and keep rel + props.
        """
        cypher = """
        MATCH (a:Disease)-[r]->(b:Disease)
        RETURN id(a) AS src_neo4j_id,
               id(b) AS dst_neo4j_id,
               type(r) AS rel,
               coalesce(r.decision, r.relation_type, r.type, r.kind) AS decision,
               properties(r) AS props
        """
        df = run_cypher_df(self.graph, cypher)
        df["props_json"] = df["props"].apply(_json_dumps_safe)
        df = df.drop(columns=["props"])
        return df

    # ---- Master export ----

    def export_all(self, out_dir: str | Path) -> ExportedPaths:
        out_dir = _to_path(out_dir)
        _ensure_dir(out_dir)

        # 1) nodes
        n_fl = self.export_nodes_findinglabel()
        n_ef = self.export_nodes_ecgfinding()
        n_dl = self.export_nodes_diseaselabel()
        n_dis = self.export_nodes_disease()

        # assign local ids per type
        n_fl, fl_map = assign_local_ids(n_fl, "neo4j_id")
        n_ef, ef_map = assign_local_ids(n_ef, "neo4j_id")
        n_dl, dl_map = assign_local_ids(n_dl, "neo4j_id")
        n_dis, dis_map = assign_local_ids(n_dis, "neo4j_id")

        # 2) edges
        e_ef_fl = self.export_edges_fl_to_ef()     # actually ef -> fl
        e_dl_dis = self.export_edges_dl_to_dis()   # dl -> dis
        e_ef_dis = self.export_edges_ef_to_dis()   # ef -> dis (normalized)
        e_dis_dis = self.export_edges_dis_to_dis() # dis -> dis

        # remap endpoints
        e_ef_fl = remap_edge_endpoints(e_ef_fl, ef_map, fl_map)
        e_dl_dis = remap_edge_endpoints(e_dl_dis, dl_map, dis_map)
        e_ef_dis = remap_edge_endpoints(e_ef_dis, ef_map, dis_map)
        e_dis_dis = remap_edge_endpoints(e_dis_dis, dis_map, dis_map)

        # 3) write outputs
        p_nodes_fl = _write_df(n_fl, out_dir / "nodes_findinglabel")
        p_nodes_ef = _write_df(n_ef, out_dir / "nodes_ecgfinding")
        p_nodes_dl = _write_df(n_dl, out_dir / "nodes_diseaselabel")
        p_nodes_dis = _write_df(n_dis, out_dir / "nodes_disease")

        p_edges_ef_fl = _write_df(e_ef_fl, out_dir / "edges_ef_to_findinglabel")
        p_edges_ef_dis = _write_df(e_ef_dis, out_dir / "edges_ecgfinding_to_disease")
        p_edges_dl_dis = _write_df(e_dl_dis, out_dir / "edges_diseaselabel_to_disease")
        p_edges_dis_dis = _write_df(e_dis_dis, out_dir / "edges_disease_to_disease")

        # 4) id maps (for debugging / later joins)
        id_maps = {
            "FindingLabel": {"neo4j_id_to_local_id": fl_map},
            "ECGFinding": {"neo4j_id_to_local_id": ef_map},
            "DiseaseLabel": {"neo4j_id_to_local_id": dl_map},
            "Disease": {"neo4j_id_to_local_id": dis_map},
        }
        id_maps_path = out_dir / "id_maps.json"
        with id_maps_path.open("w", encoding="utf-8") as f:
            json.dump(id_maps, f, ensure_ascii=False)

        return ExportedPaths(
            out_dir=out_dir,
            nodes_findinglabel=p_nodes_fl,
            nodes_ecgfinding=p_nodes_ef,
            nodes_diseaselabel=p_nodes_dl,
            nodes_disease=p_nodes_dis,
            edges_fl_to_ef=p_edges_ef_fl,
            edges_ef_to_dis=p_edges_ef_dis,
            edges_dl_to_dis=p_edges_dl_dis,
            edges_dis_to_dis=p_edges_dis_dis,
            id_maps=id_maps_path,
        )


# -------------------------
# CLI
# -------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Export Neo4j subgraph for graph-only inference.")
    parser.add_argument(
        "--out_dir",
        type=str,
        default="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/inference_systems/graph_only",
        help="Output directory, e.g., exported_graph_v1"
    )
    args = parser.parse_args()

    graph = connect_neo4j_from_env()
    exporter = GraphExporter(graph)

    paths = exporter.export_all(args.out_dir)
    print("Export complete. Wrote:")
    for k, v in paths.__dict__.items():
        print(f"  - {k}: {v}")


if __name__ == "__main__":
    main()
###################################################################################################################################
    
# -------------------------
# Load exported graph -> in-memory index
# -------------------------

from dataclasses import dataclass
from typing import DefaultDict, Set
from collections import defaultdict
import numpy as np

def _read_table(path_no_ext: Path) -> pd.DataFrame:
    """
    Read parquet if exists else csv. You may pass a concrete file path too.
    """
    p = _to_path(path_no_ext)
    if p.suffix in (".parquet", ".csv"):
        path = p
    else:
        pq = p.with_suffix(".parquet")
        cs = p.with_suffix(".csv")
        path = pq if pq.exists() else cs

    if not path.exists():
        raise FileNotFoundError(f"Missing exported table: {path}")

    if path.suffix == ".parquet":
        return pd.read_parquet(path)
    return pd.read_csv(path)


def _norm_label(s: str) -> str:
    # conservative normalization for matching
    return " ".join(str(s).strip().split()).lower()


class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1

    def groups(self) -> Dict[int, List[int]]:
        g: DefaultDict[int, List[int]] = defaultdict(list)
        for i in range(len(self.parent)):
            g[self.find(i)].append(i)
        return dict(g)


@dataclass
class GraphIndex:
    # Node tables (optional but useful for debugging / explain)
    nodes_findinglabel: pd.DataFrame
    nodes_ecgfinding: pd.DataFrame
    nodes_diseaselabel: pd.DataFrame
    nodes_disease: pd.DataFrame

    # Core adjacency
    # ef_id -> list[(fl_id, relation_type)]
    ef_to_fl: List[List[Tuple[int, str]]]
    # fl_id -> list[(ef_id, relation_type)]
    fl_to_ef: List[List[Tuple[int, str]]]

    # ef_id -> list[(dis_id, edge_type)]
    ef_to_dis: List[List[Tuple[int, str]]]
    # dis_id -> list[(ef_id, edge_type)]
    dis_to_ef: List[List[Tuple[int, str]]]

    # dl_id -> list[(dis_id, relation_type)]
    dl_to_dis: List[List[Tuple[int, str]]]
    # dis_id -> list[(dl_id, relation_type)]
    dis_to_dl: List[List[Tuple[int, str]]]

    # Disease-Disease
    # dis_id -> list[(parent_dis_id)] (IS_A edges: child -> parent)
    isa_parents: List[List[int]]
    # dis_id -> list[(child_dis_id)] (derived reverse)
    isa_children: List[List[int]]

    # synonym groups via MAPPED_TO
    syn_group_id: List[int]                 # dis_id -> group_id
    syn_groups: Dict[int, List[int]]        # group_id -> list[dis_id]

    # Label alignment helpers
    dl_label_to_id: Dict[str, int]          # exported DiseaseLabel label -> dl_id
    dl_id_by_vocab_order: Optional[np.ndarray]  # shape (n_vocab,), dl_id in disease_label.csv order
    
    missing_vocab_labels: List[str]         # vocab labels missing in graph (if any)


def load_graph_index(
        export_dir: str | Path,
        *,
        disease_label_csv: Optional[str | Path] = None,
        strict_label_alignment: bool = True,
        dedup_edges: bool = True,
    ) -> GraphIndex:
    """
    Load exported tables and build in-memory adjacency lists.

    Parameters
    ----------
    export_dir:
        Directory created by GraphExporter.export_all()
    disease_label_csv:
        If provided, we align DiseaseLabel order with disease_label.csv (recommended).
        This ensures p_disease[:, j] corresponds to the correct dl_id.
    strict_label_alignment:
        If True, require 1-1 correspondence between exported DiseaseLabel labels and disease_label.csv labels.
    dedup_edges:
        If True, de-duplicate adjacency entries.
    """
    export_dir = _to_path(export_dir)

    # ---- read node tables ----
    n_fl = _read_table(export_dir / "nodes_findinglabel")
    n_ef = _read_table(export_dir / "nodes_ecgfinding")
    n_dl = _read_table(export_dir / "nodes_diseaselabel")
    n_dis = _read_table(export_dir / "nodes_disease")

    # ---- read edge tables ----
    e_ef_fl = _read_table(export_dir / "edges_ef_to_findinglabel")  # src=ef, dst=fl
    e_ef_dis = _read_table(export_dir / "edges_ecgfinding_to_disease")  # src=ef, dst=dis
    e_dl_dis = _read_table(export_dir / "edges_diseaselabel_to_disease")  # src=dl, dst=dis
    e_dis_dis = _read_table(export_dir / "edges_disease_to_disease")  # src=dis, dst=dis

    # Basic sizes
    nF = int(n_fl["local_id"].max()) + 1 if len(n_fl) else 0
    nE = int(n_ef["local_id"].max()) + 1 if len(n_ef) else 0
    nL = int(n_dl["local_id"].max()) + 1 if len(n_dl) else 0
    nD = int(n_dis["local_id"].max()) + 1 if len(n_dis) else 0

    # ---- build DiseaseLabel mapping label->id ----
    if "label" not in n_dl.columns:
        raise ValueError("nodes_diseaselabel missing column 'label' (export should provide it).")

    dl_label_to_id: Dict[str, int] = {}
    dup_labels: Set[str] = set()
    for _, row in n_dl.iterrows():
        lab = str(row["label"])
        dlid = int(row["local_id"])
        key = _norm_label(lab)
        if key in dl_label_to_id:
            dup_labels.add(key)
        else:
            dl_label_to_id[key] = dlid
    if dup_labels:
        raise ValueError(f"Duplicate DiseaseLabel labels after normalization: {sorted(list(dup_labels))[:20]} ...")

    dl_id_by_vocab_order = None
    if disease_label_csv is not None:
        df_vocab = pd.read_csv(_to_path(disease_label_csv))
        if "label" not in df_vocab.columns:
            raise ValueError("disease_label.csv must contain column 'label'")

        vocab_labels = df_vocab["label"].astype(str).tolist()
        ids: List[int] = []
        missing: List[str] = []
        for lab in vocab_labels:
            key = _norm_label(lab)
            dlid = dl_label_to_id.get(key)
            if dlid is None:
                missing.append(lab)
                ids.append(-1)  # placeholder to keep vocab length
            else:
                ids.append(dlid)

        # Also check for extras in graph not in vocab
        vocab_set = {_norm_label(x) for x in vocab_labels}
        graph_extra = [k for k in dl_label_to_id.keys() if k not in vocab_set]

        if (missing or graph_extra) and strict_label_alignment:
            raise ValueError(...)

        dl_id_by_vocab_order = np.array(ids, dtype=np.int64)


    # ---- adjacency builders ----
    ef_to_fl: List[List[Tuple[int, str]]] = [[] for _ in range(nE)]
    fl_to_ef: List[List[Tuple[int, str]]] = [[] for _ in range(nF)]

    if len(e_ef_fl):
        # expected cols: src, dst, relation_type
        for _, r in e_ef_fl.iterrows():
            ef = int(r["src"])
            fl = int(r["dst"])
            rt = str(r.get("relation_type", "RELATED"))
            ef_to_fl[ef].append((fl, rt))
            fl_to_ef[fl].append((ef, rt))

    ef_to_dis: List[List[Tuple[int, str]]] = [[] for _ in range(nE)]
    dis_to_ef: List[List[Tuple[int, str]]] = [[] for _ in range(nD)]
    if len(e_ef_dis):
        # expected cols: src, dst, edge_type (fallback to rel)
        for _, r in e_ef_dis.iterrows():
            ef = int(r["src"])
            dis = int(r["dst"])
            et = str(r.get("edge_type", r.get("rel", "")))
            ef_to_dis[ef].append((dis, et))
            dis_to_ef[dis].append((ef, et))

    dl_to_dis: List[List[Tuple[int, str]]] = [[] for _ in range(nL)]
    dis_to_dl: List[List[Tuple[int, str]]] = [[] for _ in range(nD)]
    if len(e_dl_dis):
        for _, r in e_dl_dis.iterrows():
            dl = int(r["src"])
            dis = int(r["dst"])
            rt = str(r.get("relation_type", "RELATED"))
            dl_to_dis[dl].append((dis, rt))
            dis_to_dl[dis].append((dl, rt))

    # ---- Disease-Disease: build synonym (MAPPED_TO) + IS_A parent links ----
    uf = UnionFind(nD)
    isa_parents: List[List[int]] = [[] for _ in range(nD)]
    isa_children: List[List[int]] = [[] for _ in range(nD)]

    if len(e_dis_dis):
        for _, r in e_dis_dis.iterrows():
            a = int(r["src"])
            b = int(r["dst"])
            rel = str(r.get("rel", ""))
            decision = str(r.get("decision", ""))

            # Normalize decision: sometimes stored as rel, sometimes as property
            key = (decision or rel).upper()

            if key == "MAPPED_TO":
                uf.union(a, b)
            elif key == "IS_A":
                # a is subtype (child), b is parent (per your definition N0 IS_A N1)
                isa_parents[a].append(b)
                isa_children[b].append(a)
            elif key == "INSTANTIATES":
                # rarely used; ignore for now (or treat as child->parent)
                isa_parents[a].append(b)
                isa_children[b].append(a)

    groups = uf.groups()
    # Assign compact group ids 0..G-1
    root_to_gid = {root: i for i, root in enumerate(sorted(groups.keys()))}
    syn_group_id = [0] * nD
    syn_groups: Dict[int, List[int]] = {}
    for root, members in groups.items():
        gid = root_to_gid[root]
        syn_groups[gid] = members
        for m in members:
            syn_group_id[m] = gid

    # ---- optional edge dedup ----
    if dedup_edges:
        def _dedup_list_pairs(x: List[Tuple[int, str]]) -> List[Tuple[int, str]]:
            seen = set()
            out = []
            for a, b in x:
                k = (a, b)
                if k not in seen:
                    out.append((a, b))
                    seen.add(k)
            return out

        ef_to_fl = [_dedup_list_pairs(v) for v in ef_to_fl]
        fl_to_ef = [_dedup_list_pairs(v) for v in fl_to_ef]
        ef_to_dis = [_dedup_list_pairs(v) for v in ef_to_dis]
        dis_to_ef = [_dedup_list_pairs(v) for v in dis_to_ef]
        dl_to_dis = [_dedup_list_pairs(v) for v in dl_to_dis]
        dis_to_dl = [_dedup_list_pairs(v) for v in dis_to_dl]

        # ISA lists
        def _dedup_ints(lst: List[int]) -> List[int]:
            seen = set()
            out = []
            for t in lst:
                if t not in seen:
                    out.append(t)
                    seen.add(t)
            return out

        isa_parents = [_dedup_ints(v) for v in isa_parents]
        isa_children = [_dedup_ints(v) for v in isa_children]

    return GraphIndex(
        nodes_findinglabel=n_fl,
        nodes_ecgfinding=n_ef,
        nodes_diseaselabel=n_dl,
        nodes_disease=n_dis,
        ef_to_fl=ef_to_fl,
        fl_to_ef=fl_to_ef,
        ef_to_dis=ef_to_dis,
        dis_to_ef=dis_to_ef,
        dl_to_dis=dl_to_dis,
        dis_to_dl=dis_to_dl,
        isa_parents=isa_parents,
        isa_children=isa_children,
        syn_group_id=syn_group_id,
        syn_groups=syn_groups,
        dl_label_to_id=dl_label_to_id,
        dl_id_by_vocab_order=dl_id_by_vocab_order,
        missing_vocab_labels=missing,
    )


def summarize_graph_index(g: GraphIndex, topk: int = 10) -> Dict[str, Any]:
    """
    Lightweight summary for sanity checks (counts + relation type distributions).
    """
    out: Dict[str, Any] = {}
    out["n_findinglabel"] = len(g.nodes_findinglabel)
    out["n_ecgfinding"] = len(g.nodes_ecgfinding)
    out["n_diseaselabel"] = len(g.nodes_diseaselabel)
    out["n_disease"] = len(g.nodes_disease)

    out["edges_ef_to_fl"] = sum(len(v) for v in g.ef_to_fl)
    out["edges_ef_to_dis"] = sum(len(v) for v in g.ef_to_dis)
    out["edges_dl_to_dis"] = sum(len(v) for v in g.dl_to_dis)
    out["isa_edges"] = sum(len(v) for v in g.isa_parents)

    out["syn_groups"] = len(g.syn_groups)
    out["missing_vocab_labels"] = g.missing_vocab_labels

    # Relation type distributions
    rt_fl = defaultdict(int)
    for v in g.ef_to_fl:
        for _, rt in v:
            rt_fl[str(rt)] += 1
    out["relation_type_ef_fl_top"] = sorted(rt_fl.items(), key=lambda x: -x[1])[:topk]

    et_ef_dis = defaultdict(int)
    for v in g.ef_to_dis:
        for _, et in v:
            et_ef_dis[str(et)] += 1
    out["edge_type_ef_dis_top"] = sorted(et_ef_dis.items(), key=lambda x: -x[1])[:topk]

    rt_dl = defaultdict(int)
    for v in g.dl_to_dis:
        for _, rt in v:
            rt_dl[str(rt)] += 1
    out["relation_type_dl_dis_top"] = sorted(rt_dl.items(), key=lambda x: -x[1])[:topk]

    return out

# if __name__ == "__main__":
#     g = load_graph_index(
#         export_dir="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/inference_systems/graph_only",
#         disease_label_csv="/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/disease_label.csv",
#         strict_label_alignment=False,
#     )

#     print(summarize_graph_index(g))