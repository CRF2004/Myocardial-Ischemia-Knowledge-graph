# reasoning.py
"""
仅基于图谱的推理：证据聚合 + 概率重塑（无训练过程）。

实现三个阶段（最小可行）：
    A) FindingLabel 概率 -> ECGFinding 证据分数
    B) ECGFinding 证据 + DiseaseLabel 先验 -> 疾病后验概率
    C) 将疾病后验概率映射回 DiseaseLabel 空间

此处不进行阈值拟合；阈值/指标相关工作应单独处理。

假设 / 约定（匹配当前导出的图谱）：

GraphIndex 提供邻接关系列表：
ef_to_fl: EF -> [(FL, 关系类型)]
ef_to_dis: EF -> [(疾病, 边类型)] 边类型包含 {"INDICATES","SUPPORTS","IS_SPECIFIC_FOR",...}
dl_to_dis: DL -> [(疾病, 关系类型)] 关系类型包含 {"EXACT","BROADER","NARROWER","RELATED",...}

Disease-Disease 约束当前被跳过，因为在导出中其决策为空。

DiseaseLabel 对齐：
GraphIndex.dl_id_by_vocab_order 是一个长度为 n_vocab 的数组：
dl_id_by_vocab_order[j] = dl_本地标识符，如果该标签不在图谱中，则为 -1。
因此，我们可以在词汇表顺序（与 p_disease 的列相匹配）下安全地操作。

输出：

p_star_label: np.ndarray，形状为 (N, n_vocab)，值在 [0,1] 范围内。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

# Import GraphIndex from your graph_cache.py
# If you moved GraphIndex to another file, update this import accordingly.
from graph_cache import GraphIndex


# -----------------------------
# Weight configuration
# -----------------------------

@dataclass(frozen=True)
class WeightConfig:
    # relation_type for EF->FindingLabel (RELATED_TO mapping)
    w_rel_finding: Dict[str, float]
    # relation_type for DiseaseLabel->Disease mapping
    w_rel_label_to_disease: Dict[str, float]
    # edge_type for ECGFinding->Disease evidence
    w_edge_evidence: Dict[str, float]

    # fusion scalar for adding evidence to prior in logit space
    alpha: float = 1.0
    # numeric stability
    eps: float = 1e-6

    @staticmethod
    def default() -> "WeightConfig":
        return WeightConfig(
            w_rel_finding={
                "EXACT": 1.0,
                "NARROWER": 0.8,
                "BROADER": 0.6,
                "RELATED": 0.1,
                # fallback
                "": 0.4,
                "NONE": 0.0,
            },
            w_rel_label_to_disease={
                "EXACT": 1.0,
                "BROADER": 0.8,
                "NARROWER": 0.6,
                "RELATED": 0.1,
                "NONE": 0.0,
                "": 0.4,
            },
            w_edge_evidence={
                "INDICATES": 1.0,
                "IS_SPECIFIC_FOR": 0.9,
                "SUPPORTS": 0.7,
                # fallback
                "ASSOCIATED_WITH": 0.6,
                "": 0.5,
            },
            alpha=1.0,
            eps=1e-6,
        )


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _logit(p: np.ndarray, eps: float) -> np.ndarray:
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p) - np.log(1.0 - p)


# -----------------------------
# Stage A: p_finding -> e_ecgfinding
# -----------------------------

def infer_ecgfinding_evidence(
        g: GraphIndex,
        p_finding: np.ndarray,
        *,
        weights: WeightConfig,
    ) -> np.ndarray:
    """
    Compute ECGFinding evidence score e(ef) for each sample.

    Inputs
    ------
    p_finding: (N, n_findinglabel_vocab)
        Must match FindingLabel label order used in training.
        We assume that FindingLabel nodes in graph correspond to this same label set.
        (If not, you should add an explicit alignment similar to DiseaseLabel.)

    Output
    ------
    e_ef: (N, n_ecgfinding)
    """
    if p_finding.ndim != 2:
        raise ValueError(f"p_finding must be 2D (N,C). Got {p_finding.shape}")
    N = p_finding.shape[0]
    nE = len(g.ef_to_fl)

    # We compute noisy-OR:
    # e(ef) = 1 - Π_{fl in N(ef)} (1 - w(rel_type) * p_finding[fl])
    e_ef = np.zeros((N, nE), dtype=np.float32)

    # For efficiency, loop over ef and vectorize over N
    one = 1.0
    for ef_id in range(nE):
        neigh = g.ef_to_fl[ef_id]
        if not neigh:
            continue
        prod = np.ones(N, dtype=np.float32)
        for fl_id, rel_type in neigh:
            w = weights.w_rel_finding.get(str(rel_type).upper(), weights.w_rel_finding.get("", 0.4))
            if w <= 0:
                continue
            prod *= (one - w * p_finding[:, fl_id].astype(np.float32))
        e_ef[:, ef_id] = one - prod

    # numerical safety
    e_ef = np.clip(e_ef, 0.0, 1.0)
    return e_ef


# -----------------------------
# Stage B: prior (p_disease) -> P0(disease), then fuse with evidence -> P*(disease)
# -----------------------------

def aggregate_prior_to_disease(
        g: GraphIndex,
        p_disease_label_vocab: np.ndarray,
        *,
        weights: WeightConfig,
    ) -> np.ndarray:
    """
    Project DiseaseLabel priors (vocab order) into Disease concept space via mapping edges.

    p_disease_label_vocab: (N, n_vocab)
      Model2 probabilities in disease_label.csv order.

    Returns
    -------
    P0_dis: (N, n_disease_concepts)
      Noisy-OR aggregation over mapped labels.

    Important:
    - If a vocab label is missing in graph (dl_id == -1), it contributes nothing to P0_dis.
    """
    if p_disease_label_vocab.ndim != 2:
        raise ValueError(f"p_disease must be 2D (N,C). Got {p_disease_label_vocab.shape}")
    if g.dl_id_by_vocab_order is None:
        raise ValueError("GraphIndex.dl_id_by_vocab_order is required for label alignment.")

    N, n_vocab = p_disease_label_vocab.shape
    nD = len(g.dis_to_dl)

    # Initialize product terms for noisy-OR
    prod = np.ones((N, nD), dtype=np.float32)

    # For each vocab index j -> dl_id (or -1)
    dl_ids = g.dl_id_by_vocab_order
    if dl_ids.shape[0] != n_vocab:
        raise ValueError(
            f"dl_id_by_vocab_order length {dl_ids.shape[0]} != p_disease vocab {n_vocab}."
        )

    # iterate over vocab columns; map to diseases via dl_to_dis
    for j in range(n_vocab):
        dl_id = int(dl_ids[j])
        if dl_id < 0:
            continue
        neigh = g.dl_to_dis[dl_id]
        if not neigh:
            continue
        p_col = p_disease_label_vocab[:, j].astype(np.float32)
        for dis_id, rel_type in neigh:
            w = weights.w_rel_label_to_disease.get(str(rel_type).upper(), weights.w_rel_label_to_disease.get("", 0.4))
            if w <= 0:
                continue
            prod[:, dis_id] *= (1.0 - w * p_col)

    P0 = 1.0 - prod
    P0 = np.clip(P0, 0.0, 1.0)
    return P0


def aggregate_evidence_to_disease(
        g: GraphIndex,
        e_ef: np.ndarray,
        *,
        weights: WeightConfig,
    ) -> np.ndarray:
    """
    Aggregate ECGFinding evidence into Disease concept space.

    e_ef: (N, n_ecgfinding)

    Returns
    -------
    E_dis: (N, n_disease_concepts)
    """
    if e_ef.ndim != 2:
        raise ValueError(f"e_ef must be 2D (N,E). Got {e_ef.shape}")
    N = e_ef.shape[0]
    nD = len(g.dis_to_ef)

    prod = np.ones((N, nD), dtype=np.float32)

    # For each EF, push evidence to linked diseases
    nE = len(g.ef_to_dis)
    if e_ef.shape[1] != nE:
        raise ValueError(f"e_ef second dim {e_ef.shape[1]} != number of ECGFinding nodes {nE}")

    for ef_id in range(nE):
        neigh = g.ef_to_dis[ef_id]
        if not neigh:
            continue
        e_col = e_ef[:, ef_id].astype(np.float32)
        for dis_id, edge_type in neigh:
            w = weights.w_edge_evidence.get(str(edge_type).upper(), weights.w_edge_evidence.get("", 0.5))
            if w <= 0:
                continue
            prod[:, dis_id] *= (1.0 - w * e_col)

    E_dis = 1.0 - prod
    E_dis = np.clip(E_dis, 0.0, 1.0)
    return E_dis


def fuse_prior_and_evidence(
        P0_dis: np.ndarray,
        E_dis: np.ndarray,
        *,
        weights: WeightConfig,
    ) -> np.ndarray:
    """
    Fuse prior and evidence in logit space:
      logit(P*) = logit(P0) + alpha * logit(E + eps)

    Returns
    -------
    P_star_dis: (N, n_disease_concepts)
    """
    if P0_dis.shape != E_dis.shape:
        raise ValueError(f"P0_dis shape {P0_dis.shape} != E_dis shape {E_dis.shape}")

    eps = weights.eps
    alpha = weights.alpha

    z = _logit(P0_dis, eps) + alpha * _logit(np.clip(E_dis, 0.0, 1.0), eps)
    P_star = _sigmoid(z).astype(np.float32)
    P_star = np.clip(P_star, 0.0, 1.0)
    return P_star


# -----------------------------
# Stage C: project back to DiseaseLabel vocab
# -----------------------------

def project_disease_to_label_vocab(
        g: GraphIndex,
        P_star_dis: np.ndarray,
        p_disease_label_vocab: np.ndarray,
        *,
        weights: WeightConfig,
    ) -> np.ndarray:
    """
    Project Disease posteriors back to DiseaseLabel vocab space.

    For each vocab label j:
      - if dl_id == -1 (missing in graph): keep model probability unchanged
      - else: p*_label[j] = noisy-OR over mapped diseases:
            1 - Π (1 - w(map_rel) * P_star_dis[disease])

    Returns
    -------
    p_star_label_vocab: (N, n_vocab)
    """
    if g.dl_id_by_vocab_order is None:
        raise ValueError("GraphIndex.dl_id_by_vocab_order is required for label alignment.")
    N, n_vocab = p_disease_label_vocab.shape
    if P_star_dis.shape[0] != N:
        raise ValueError("Row count mismatch between P_star_dis and p_disease_label_vocab")
    nD = P_star_dis.shape[1]
    if nD != len(g.dis_to_dl):
        raise ValueError("Disease concept dimension mismatch with graph index")

    dl_ids = g.dl_id_by_vocab_order
    if dl_ids.shape[0] != n_vocab:
        raise ValueError("dl_id_by_vocab_order length mismatch with vocab size")

    p_star = p_disease_label_vocab.astype(np.float32).copy()

    for j in range(n_vocab):
        dl_id = int(dl_ids[j])
        if dl_id < 0:
            # Missing label in graph; keep original model prob
            continue
        neigh = g.dl_to_dis[dl_id]
        if not neigh:
            continue

        prod = np.ones(N, dtype=np.float32)
        for dis_id, rel_type in neigh:
            w = weights.w_rel_label_to_disease.get(str(rel_type).upper(), weights.w_rel_label_to_disease.get("", 0.4))
            if w <= 0:
                continue
            prod *= (1.0 - w * P_star_dis[:, dis_id].astype(np.float32))

        p_star[:, j] = 1.0 - prod

    p_star = np.clip(p_star, 0.0, 1.0)
    return p_star


# -----------------------------
# End-to-end API
# -----------------------------

def graph_only_infer_pstar(
        g: GraphIndex,
        p_finding: np.ndarray,
        p_disease: np.ndarray,
        *,
        weights: Optional[WeightConfig] = None,
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
    """
    End-to-end graph-only inference producing p*_disease in vocab order.

    Returns
    -------
    p_star_label: (N, n_vocab)
    extras: dict with intermediate matrices (optional for debugging)
      - e_ef
      - P0_dis
      - E_dis
      - P_star_dis
    """
    weights = weights or WeightConfig.default()

    e_ef = infer_ecgfinding_evidence(g, p_finding, weights=weights)
    P0_dis = aggregate_prior_to_disease(g, p_disease, weights=weights)
    E_dis = aggregate_evidence_to_disease(g, e_ef, weights=weights)
    P_star_dis = fuse_prior_and_evidence(P0_dis, E_dis, weights=weights)
    p_star_label = project_disease_to_label_vocab(g, P_star_dis, p_disease, weights=weights)

    extras = {
        "e_ef": e_ef,
        "P0_dis": P0_dis,
        "E_dis": E_dis,
        "P_star_dis": P_star_dis,
    }
    return p_star_label, extras

def blend_with_model2(
        p_model2: np.ndarray,
        p_graph: np.ndarray,
        lam: float,
    ) -> np.ndarray:
    """
    Simple convex blending: p_final = (1-lam)*p_model2 + lam*p_graph
    """
    if p_model2.shape != p_graph.shape:
        raise ValueError(f"Shape mismatch: {p_model2.shape} vs {p_graph.shape}")
    lam = float(lam)
    if not (0.0 <= lam <= 1.0):
        raise ValueError("lam must be in [0,1]")
    out = (1.0 - lam) * p_model2.astype(np.float32) + lam * p_graph.astype(np.float32)
    return np.clip(out, 0.0, 1.0)
