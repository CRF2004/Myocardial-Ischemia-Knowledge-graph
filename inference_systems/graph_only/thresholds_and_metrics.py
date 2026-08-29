"""
用于多标签疾病预测的阈值拟合与指标计算。
本模块是模型无关的：它处理
    概率 p: (N, C)
    真实标签 Y: (N, C) 二值化
    并产生每个标签的阈值 tau: (C,)
    硬预测 Yhat: (N, C)
汇总指标（微平均/宏平均 F1，AUROC/AUPRC）
设计目标：
    匹配常见的 PTB-XL 多标签评估模式
    稳健处理全 0 / 全 1 标签（AUROC/AUPRC 可能未定义的情况）
    保持依赖最小化（仅需 numpy + sklearn）
说明：
    对于您的设置，"已标注"样本是全量标注的，因此不需要掩码。
    如果后续需要掩码，可将其添加为可选参数，并忽略被掩码的标签。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Literal, Optional, Tuple

import numpy as np
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


FitObjective = Literal["per_label_f1", "macro_f1"]
TieBreak = Literal["higher_recall", "higher_precision", "lower_threshold"]


@dataclass(frozen=True)
class ThresholdFitConfig:
    objective: FitObjective = "per_label_f1"
    grid_size: int = 201          # thresholds in [0,1] inclusive
    tie_break: TieBreak = "higher_recall"
    min_pos: int = 1              # require at least this many positives in val to fit label-specific tau
    default_tau: float = 0.5      # used if label cannot be fit
    clip_eps: float = 1e-8


def apply_thresholds(p: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """
    Apply per-label thresholds.

    p: (N, C) probabilities
    tau: (C,) thresholds

    Returns:
      yhat: (N, C) binary int8
    """
    p = np.asarray(p)
    tau = np.asarray(tau)
    if p.ndim != 2:
        raise ValueError(f"p must be 2D. Got {p.shape}")
    if tau.ndim != 1 or tau.shape[0] != p.shape[1]:
        raise ValueError(f"tau must be (C,) with C={p.shape[1]}. Got {tau.shape}")
    return (p >= tau[None, :]).astype(np.int8)


def _safe_div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else 0.0


def _precision_recall_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Tuple[float, float, float]:
    """
    Compute precision/recall/F1 for binary arrays.
    """
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))

    precision = _safe_div(tp, tp + fp)
    recall = _safe_div(tp, tp + fn)
    f1 = _safe_div(2 * precision * recall, precision + recall) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def fit_thresholds_per_label(
    p_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: ThresholdFitConfig = ThresholdFitConfig(),
) -> np.ndarray:
    """
    Fit per-label thresholds on validation set.

    Strategy:
    - Evaluate a fixed grid of thresholds for each label independently.
    - Choose threshold maximizing F1 (or contributing to macro-F1 if objective="macro_f1").
      For objective="macro_f1", we still do coordinate-wise independent search;
      this is a strong baseline and keeps runtime predictable.

    Returns:
      tau: (C,) float32
    """
    p_val = np.asarray(p_val)
    y_val = np.asarray(y_val)

    if p_val.ndim != 2 or y_val.ndim != 2:
        raise ValueError(f"p_val and y_val must be 2D. Got {p_val.shape}, {y_val.shape}")
    if p_val.shape != y_val.shape:
        raise ValueError(f"Shape mismatch: p_val {p_val.shape} vs y_val {y_val.shape}")

    N, C = p_val.shape
    thresholds = np.linspace(0.0, 1.0, cfg.grid_size, dtype=np.float32)

    tau = np.full((C,), float(cfg.default_tau), dtype=np.float32)

    for c in range(C):
        y = y_val[:, c].astype(np.int8)
        p = p_val[:, c].astype(np.float32)

        pos = int(y.sum())
        neg = int(N - pos)
        if pos < cfg.min_pos or neg < 1:
            # Can't meaningfully fit; keep default
            continue

        best_f1 = -1.0
        best_tau = float(cfg.default_tau)
        best_prec = 0.0
        best_rec = 0.0

        # Evaluate thresholds
        for t in thresholds:
            yhat = (p >= t).astype(np.int8)
            prec, rec, f1 = _precision_recall_f1(y, yhat)

            if f1 > best_f1 + 1e-12:
                best_f1, best_tau, best_prec, best_rec = f1, float(t), prec, rec
            elif abs(f1 - best_f1) <= 1e-12:
                # tie-break
                if cfg.tie_break == "higher_recall" and rec > best_rec + 1e-12:
                    best_f1, best_tau, best_prec, best_rec = f1, float(t), prec, rec
                elif cfg.tie_break == "higher_precision" and prec > best_prec + 1e-12:
                    best_f1, best_tau, best_prec, best_rec = f1, float(t), prec, rec
                elif cfg.tie_break == "lower_threshold" and float(t) < best_tau:
                    best_f1, best_tau, best_prec, best_rec = f1, float(t), prec, rec

        tau[c] = np.float32(best_tau)

    return tau


# -----------------------------
# Metrics
# -----------------------------

@dataclass(frozen=True)
class MetricsResult:
    micro_f1: float
    macro_f1: float
    micro_auroc: Optional[float]
    macro_auroc: Optional[float]
    micro_auprc: Optional[float]
    macro_auprc: Optional[float]
    per_label_f1: np.ndarray          # (C,)
    per_label_auroc: np.ndarray       # (C,) with nan for undefined
    per_label_auprc: np.ndarray       # (C,) with nan for undefined


def compute_metrics(
    p: np.ndarray,
    y_true: np.ndarray,
    y_pred: Optional[np.ndarray] = None,
    *,
    clip_eps: float = 1e-8,
) -> MetricsResult:
    """
    Compute standard multilabel metrics.

    - F1 uses y_pred if provided; else uses default 0.5 threshold.
    - AUROC/AUPRC computed from probabilities p.
      For labels with all-0 or all-1 ground truth, AUROC/AUPRC are undefined -> nan.
    """
    p = np.asarray(p, dtype=np.float32)
    y_true = np.asarray(y_true, dtype=np.int8)

    if p.ndim != 2 or y_true.ndim != 2:
        raise ValueError(f"p and y_true must be 2D. Got {p.shape}, {y_true.shape}")
    if p.shape != y_true.shape:
        raise ValueError(f"Shape mismatch: p {p.shape} vs y_true {y_true.shape}")

    N, C = p.shape
    p = np.clip(p, clip_eps, 1.0 - clip_eps)

    if y_pred is None:
        y_pred = (p >= 0.5).astype(np.int8)
    else:
        y_pred = np.asarray(y_pred, dtype=np.int8)
        if y_pred.shape != y_true.shape:
            raise ValueError(f"y_pred shape {y_pred.shape} != y_true shape {y_true.shape}")

    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    per_label_f1 = np.zeros((C,), dtype=np.float32)
    for c in range(C):
        per_label_f1[c] = float(f1_score(y_true[:, c], y_pred[:, c], average="binary", zero_division=0))
    macro_f1 = float(np.mean(per_label_f1))

    per_label_auroc = np.full((C,), np.nan, dtype=np.float32)
    per_label_auprc = np.full((C,), np.nan, dtype=np.float32)

    # Compute per-label AUROC/AUPRC where defined
    for c in range(C):
        y = y_true[:, c]
        if y.min() == y.max():
            continue  # undefined
        try:
            per_label_auroc[c] = float(roc_auc_score(y, p[:, c]))
        except Exception:
            pass
        try:
            per_label_auprc[c] = float(average_precision_score(y, p[:, c]))
        except Exception:
            pass

    # Micro AUROC/AUPRC: defined if there is at least one pos and one neg overall
    micro_auroc = None
    micro_auprc = None
    try:
        if y_true.min() != y_true.max():
            micro_auroc = float(roc_auc_score(y_true, p, average="micro"))
    except Exception:
        micro_auroc = None
    try:
        micro_auprc = float(average_precision_score(y_true, p, average="micro"))
    except Exception:
        micro_auprc = None

    # Macro AUROC/AUPRC: average over defined labels only
    macro_auroc = None
    macro_auprc = None
    auroc_defined = per_label_auroc[~np.isnan(per_label_auroc)]
    auprc_defined = per_label_auprc[~np.isnan(per_label_auprc)]
    if auroc_defined.size > 0:
        macro_auroc = float(np.mean(auroc_defined))
    if auprc_defined.size > 0:
        macro_auprc = float(np.mean(auprc_defined))

    return MetricsResult(
        micro_f1=micro_f1,
        macro_f1=macro_f1,
        micro_auroc=micro_auroc,
        macro_auroc=macro_auroc,
        micro_auprc=micro_auprc,
        macro_auprc=macro_auprc,
        per_label_f1=per_label_f1,
        per_label_auroc=per_label_auroc,
        per_label_auprc=per_label_auprc,
    )


# -----------------------------
# End-to-end helper
# -----------------------------

def fit_and_eval(
    p_val: np.ndarray,
    y_val: np.ndarray,
    p_test: np.ndarray,
    y_test: np.ndarray,
    *,
    cfg: ThresholdFitConfig = ThresholdFitConfig(),
) -> Tuple[np.ndarray, MetricsResult, MetricsResult]:
    """
    Fit thresholds on val, apply to val/test, and return metrics for both.

    Returns:
      tau, metrics_val, metrics_test
    """
    tau = fit_thresholds_per_label(p_val, y_val, cfg=cfg)
    yhat_val = apply_thresholds(p_val, tau)
    yhat_test = apply_thresholds(p_test, tau)

    m_val = compute_metrics(p_val, y_val, yhat_val, clip_eps=cfg.clip_eps)
    m_test = compute_metrics(p_test, y_test, yhat_test, clip_eps=cfg.clip_eps)
    return tau, m_val, m_test