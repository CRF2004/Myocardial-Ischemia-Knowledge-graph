"""
用于仅基于图谱推理的端到端执行流程：
    io_inputs -> 加载模型输出 + 从 PTB-XL 诊断陈述构建 Y 真值
    graph_cache -> 加载导出的子图到 GraphIndex
    reasoning -> 仅基于图谱的概率重塑：p* (对应于 DiseaseLabel 词表)
    thresholds_and_metrics -> 在 val_annot 上拟合阈值，在 test_annot 上评估

此脚本支持：

在 model2 概率上的基线评估（不使用图谱）

仅基于图谱的评估，生成 p_star 并比较指标

假设已通过 graph_cache.py 的导出器导出 Neo4j 子图，并已使用 -1 占位符为缺失标签修复了 dl_id_by_vocab_order。

用法（示例）：
python /mnt/chengrongfeng_private/SRP心肌缺血知识图谱/inference_systems/graph_only/run_graph_only.py \
    --model2_dir /mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model/models/model2_z_oof_all_12sl \
    --export_dir /mnt/chengrongfeng_private/SRP心肌缺血知识图谱/inference_systems/graph_only \
    --ptbxl_statements_csv /mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv \
    --disease_label_csv /mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/disease_label.csv \
    --out_dir /mnt/chengrongfeng_private/SRP心肌缺血知识图谱/inference_systems/graph_only \
    --strict_label_alignment 0 \
    --use_fixed_threshold 1
    
python3 /mnt/c/Users/12879/Desktop/projects/SRP心肌缺血知识图谱/inference_systems/graph_only/run_graph_only.py \
    --model2_dir /mnt/c/Users/12879/Desktop/projects/SRP心肌缺血知识图谱/model/model2_z_oof_all_12sl \
    --export_dir /mnt/c/Users/12879/Desktop/projects/SRP心肌缺血知识图谱/inference_systems/graph_only \
    --ptbxl_statements_csv /mnt/c/Users/12879/Desktop/projects/SRP心肌缺血知识图谱/data/labels/ptbxl_statements.csv \
    --disease_label_csv /mnt/c/Users/12879/Desktop/projects/SRP心肌缺血知识图谱/data/labels/disease_label.csv \
    --out_dir /mnt/c/Users/12879/Desktop/projects/SRP心肌缺血知识图谱/inference_systems/graph_only \
    --strict_label_alignment 0
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import numpy as np

from io_inputs import build_y_loader, load_model2_batches
from graph_cache import load_graph_index, summarize_graph_index
from reasoning import WeightConfig, graph_only_infer_pstar, blend_with_model2
from thresholds_and_metrics import ThresholdFitConfig, apply_thresholds, compute_metrics, fit_and_eval

def _to_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def _ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def _metrics_to_dict(m) -> Dict[str, Any]:
    return {
        "micro_f1": float(m.micro_f1),
        "macro_f1": float(m.macro_f1),
        "micro_auroc": None if m.micro_auroc is None else float(m.micro_auroc),
        "macro_auroc": None if m.macro_auroc is None else float(m.macro_auroc),
        "micro_auprc": None if m.micro_auprc is None else float(m.micro_auprc),
        "macro_auprc": None if m.macro_auprc is None else float(m.macro_auprc),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="在PTB-XL数据集上运行graph_only+阈值拟合")
    parser.add_argument("--model2_dir", type=str, required=True, help="model2_z_oof_all_12sl目录路径")
    parser.add_argument("--export_dir", type=str, required=True, help="exported_graph_v1目录路径")
    parser.add_argument("--ptbxl_statements_csv", type=str, required=True, help="ptbxl_statements.csv文件路径")
    parser.add_argument("--disease_label_csv", type=str, required=True, help="disease_label.csv文件路径")
    parser.add_argument("--out_dir", type=str, required=True, help="输出结果目录")
    parser.add_argument("--strict_label_alignment", type=int, default=0, help="标签对齐严格度：1=严格模式，0=允许缺失标签")
    parser.add_argument("--grid_size", type=int, default=201, help="阈值网格大小，范围[0,1]")
    parser.add_argument("--default_tau", type=float, default=0.5, help="无法拟合阈值时的默认阈值")
    parser.add_argument("--min_pos", type=int, default=1, help="验证集中拟合标签阈值所需的最小正样本数")
    parser.add_argument("--tie_break", type=str, default="higher_recall",
                        choices=["higher_recall", "higher_precision", "lower_threshold"],
                        help="阈值选择平局打破策略：higher_recall=更高召回率, higher_precision=更高精确率, lower_threshold=更低阈值")
    parser.add_argument("--use_fixed_threshold", type=int, default=0,
                        help="1=use fixed 0.5 threshold (no per-label fitting).")
    # 权重覆盖参数（可选）
    parser.add_argument("--alpha", type=float, default=1.0, help="证据融合系数α（logit空间）")
    parser.add_argument("--eps", type=float, default=1e-6, help="数值稳定性epsilon")
    args = parser.parse_args()

    out_dir = _to_path(args.out_dir)
    _ensure_dir(out_dir)

    # 1) Load Y builder and model2 batches (val/test annotated)
    y_loader = build_y_loader(
        ptbxl_statements_csv=args.ptbxl_statements_csv,
        disease_label_csv=args.disease_label_csv,
    )

    batches = load_model2_batches(
        model2_dir=args.model2_dir,
        y_loader=y_loader,
        strict_alignment=True,
    )
    val = batches["val_annot"]
    test = batches["test_annot"]

    # Sanity checks
    if val.p_disease is None or test.p_disease is None:
        raise RuntimeError("Missing p_disease for val_annot/test_annot in model2_dir.")
    if val.p_finding is None or test.p_finding is None:
        raise RuntimeError("Missing p_finding for val_annot/test_annot in model2_dir.")
    if val.y_true is None or test.y_true is None:
        raise RuntimeError("Missing y_true. y_loader did not return matrices.")

    # 2) Load GraphIndex
    g = load_graph_index(
        export_dir=args.export_dir,
        disease_label_csv=args.disease_label_csv,
        strict_label_alignment=bool(args.strict_label_alignment),
    )
    g_summary = summarize_graph_index(g)
    with (out_dir / "graph_summary.json").open("w", encoding="utf-8") as f:
        json.dump(g_summary, f, ensure_ascii=False, indent=2)

    # 3) Baseline: model2 probabilities
    cfg = ThresholdFitConfig(
        grid_size=int(args.grid_size),
        tie_break=args.tie_break,  # type: ignore
        default_tau=float(args.default_tau),
        min_pos=int(args.min_pos),
    )

    use_fixed = bool(args.use_fixed_threshold)

    def _fixed_eval(
        p_val: np.ndarray, y_val: np.ndarray,
        p_test: np.ndarray, y_test: np.ndarray,
        threshold: float = 0.5,
    ):
        tau = np.full((p_val.shape[1],), float(threshold), dtype=np.float32)
        yhat_val = apply_thresholds(p_val, tau)
        yhat_test = apply_thresholds(p_test, tau)
        m_val = compute_metrics(p_val, y_val, yhat_val, clip_eps=cfg.clip_eps)
        m_test = compute_metrics(p_test, y_test, yhat_test, clip_eps=cfg.clip_eps)
        return tau, m_val, m_test

    if use_fixed:
        tau_m2, m2_val, m2_test = _fixed_eval(
            val.p_disease, val.y_true,
            test.p_disease, test.y_true,
            threshold=0.5,
        )
    else:
        tau_m2, m2_val, m2_test = fit_and_eval(
            val.p_disease, val.y_true,
            test.p_disease, test.y_true,
            cfg=cfg,
        )

    # Save baseline artifacts
    np.save(out_dir / "tau_model2.npy", tau_m2)
    with (out_dir / "metrics_model2.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"val": _metrics_to_dict(m2_val), "test": _metrics_to_dict(m2_test)},
            f, ensure_ascii=False, indent=2
        )

    # 4) Graph-only: infer p*
    weights = WeightConfig.default()
    weights = WeightConfig(
        w_rel_finding=weights.w_rel_finding,
        w_rel_label_to_disease=weights.w_rel_label_to_disease,
        w_edge_evidence=weights.w_edge_evidence,
        alpha=float(args.alpha),
        eps=float(args.eps),
    )

    p_star_val, extras_val = graph_only_infer_pstar(g, val.p_finding, val.p_disease, weights=weights)
    p_star_test, extras_test = graph_only_infer_pstar(g, test.p_finding, test.p_disease, weights=weights)

    np.save(out_dir / "p_star_val.npy", p_star_val)
    np.save(out_dir / "p_star_test.npy", p_star_test)

    lams = np.linspace(0.0, 0.15, 16)
    best = None

    for lam in lams:
        p_val_mix = blend_with_model2(val.p_disease, p_star_val, lam)
        p_test_mix = blend_with_model2(test.p_disease, p_star_test, lam)

        if use_fixed:
            tau, m_val, m_test = _fixed_eval(
                p_val_mix, val.y_true,
                p_test_mix, test.y_true,
                threshold=0.5,
            )
        else:
            tau, m_val, m_test = fit_and_eval(
                p_val_mix, val.y_true,
                p_test_mix, test.y_true,
                cfg=cfg,
            )
        score = m_val.macro_f1  # 或者你更关注 micro_f1

        record = {
            "lam": lam,
            "val": {"micro_f1": m_val.micro_f1, "macro_f1": m_val.macro_f1, "micro_auprc": m_val.micro_auprc},
            "test": {"micro_f1": m_test.micro_f1, "macro_f1": m_test.macro_f1, "micro_auprc": m_test.micro_auprc},
        }
        print(record)

        if best is None or score > best["score"]:
            best = {"score": score, "record": record, "tau": tau}

    # 保存 best
    with (out_dir / "best_blend.json").open("w", encoding="utf-8") as f:
        json.dump(best["record"], f, ensure_ascii=False, indent=2)
    np.save(out_dir / "tau_best_blend.npy", best["tau"])

    # Save a small set of intermediate stats (not full matrices, which can be huge)
    def _extras_stats(extras: Dict[str, np.ndarray]) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for k, v in extras.items():
            out[k] = {
                "shape": list(v.shape),
                "min": float(np.min(v)),
                "max": float(np.max(v)),
                "mean": float(np.mean(v)),
            }
        return out

    with (out_dir / "extras_stats_val.json").open("w", encoding="utf-8") as f:
        json.dump(_extras_stats(extras_val), f, ensure_ascii=False, indent=2)
    with (out_dir / "extras_stats_test.json").open("w", encoding="utf-8") as f:
        json.dump(_extras_stats(extras_test), f, ensure_ascii=False, indent=2)

    # 5) Fit thresholds on val_annot using p*, evaluate on test_annot
    if use_fixed:
        tau_g, g_val, g_test = _fixed_eval(
            p_star_val, val.y_true,
            p_star_test, test.y_true,
            threshold=0.5,
        )
    else:
        tau_g, g_val, g_test = fit_and_eval(
            p_star_val, val.y_true,
            p_star_test, test.y_true,
            cfg=cfg,
        )

    np.save(out_dir / "tau_graph.npy", tau_g)
    with (out_dir / "metrics_graph.json").open("w", encoding="utf-8") as f:
        json.dump(
            {"val": _metrics_to_dict(g_val), "test": _metrics_to_dict(g_test)},
            f, ensure_ascii=False, indent=2
        )

    # 6) Print key results
    print("=== Graph summary ===")
    print(json.dumps(g_summary, ensure_ascii=False, indent=2))

    print("\n=== Baseline (model2) ===")
    print("Val :", _metrics_to_dict(m2_val))
    print("Test:", _metrics_to_dict(m2_test))

    print("\n=== Graph-only (p*) ===")
    print("Val :", _metrics_to_dict(g_val))
    print("Test:", _metrics_to_dict(g_test))

    # 7) Quick delta
    def _delta(a: float | None, b: float | None) -> float | None:
        if a is None or b is None:
            return None
        return float(b - a)

    print("\n=== Delta (graph - model2) on TEST ===")
    print({
        "micro_f1": _delta(m2_test.micro_f1, g_test.micro_f1),
        "macro_f1": _delta(m2_test.macro_f1, g_test.macro_f1),
        "micro_auroc": _delta(m2_test.micro_auroc, g_test.micro_auroc),
        "micro_auprc": _delta(m2_test.micro_auprc, g_test.micro_auprc),
        "macro_auroc": _delta(m2_test.macro_auroc, g_test.macro_auroc),
        "macro_auprc": _delta(m2_test.macro_auprc, g_test.macro_auprc),
    })

    # 8) Save runner config for reproducibility
    runner_cfg = {
        "model2_dir": args.model2_dir,
        "export_dir": args.export_dir,
        "ptbxl_statements_csv": args.ptbxl_statements_csv,
        "disease_label_csv": args.disease_label_csv,
        "strict_label_alignment": bool(args.strict_label_alignment),
        "threshold_cfg": {
            "grid_size": int(args.grid_size),
            "default_tau": float(args.default_tau),
            "min_pos": int(args.min_pos),
            "tie_break": args.tie_break,
        },
        "use_fixed_threshold": bool(args.use_fixed_threshold),
        "weights": {
            "alpha": float(args.alpha),
            "eps": float(args.eps),
            "w_rel_finding": weights.w_rel_finding,
            "w_rel_label_to_disease": weights.w_rel_label_to_disease,
            "w_edge_evidence": weights.w_edge_evidence,
        },
    }
    with (out_dir / "run_config.json").open("w", encoding="utf-8") as f:
        json.dump(runner_cfg, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
