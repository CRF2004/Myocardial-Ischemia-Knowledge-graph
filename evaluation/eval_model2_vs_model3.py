import ast
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score


def _safe_parse_list_of_tuples(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return []
    if not isinstance(x, str):
        return []
    x = x.strip()
    if x == "" or x == "[]":
        return []
    try:
        val = ast.literal_eval(x)
        return val if isinstance(val, list) else []
    except Exception:
        return []


def _build_label_vocab(label_csv_path):
    df = pd.read_csv(label_csv_path)
    if "label" not in df.columns or "snomed_id" not in df.columns:
        raise ValueError(
            f"Label file must contain columns ['label','snomed_id']: {label_csv_path}"
        )
    labels = df["label"].astype(str).tolist()
    snomed_ids = df["snomed_id"].astype(int).tolist()
    snomed_to_index = {sid: i for i, sid in enumerate(snomed_ids)}
    return labels, snomed_ids, snomed_to_index


def _load_snomeds_for_ecg_ids(labels_path, ecg_ids, snomed_col):
    stmt_df = pd.read_csv(labels_path, usecols=["ecg_id", snomed_col])
    stmt_df["ecg_id"] = stmt_df["ecg_id"].astype(int)
    target_set = set(int(x) for x in ecg_ids)
    stmt_df = stmt_df[stmt_df["ecg_id"].isin(target_set)].copy()

    snomed_by_ecg = {}
    for _, row in stmt_df.iterrows():
        eid = int(row["ecg_id"])
        parsed = _safe_parse_list_of_tuples(row[snomed_col])
        snomeds = set()
        for item in parsed:
            if isinstance(item, (tuple, list)) and len(item) >= 1:
                try:
                    snomeds.add(int(item[0]))
                except Exception:
                    continue
        snomed_by_ecg[eid] = snomeds

    missing = [eid for eid in ecg_ids if eid not in snomed_by_ecg]
    if missing:
        raise ValueError(
            f"{len(missing)} ecg_ids missing in statements file. Example: {missing[:10]}"
        )
    return snomed_by_ecg


def _build_y_true(ecg_ids, labels_path, disease_label_path, snomed_col):
    labels, snomed_ids, disease_map = _build_label_vocab(disease_label_path)
    snomed_by_ecg = _load_snomeds_for_ecg_ids(labels_path, ecg_ids, snomed_col)

    y = np.zeros((len(ecg_ids), len(labels)), dtype=np.float32)
    for i, eid in enumerate(ecg_ids):
        snomeds = snomed_by_ecg[eid]
        for sid in snomeds:
            idx = disease_map.get(sid)
            if idx is not None:
                y[i, idx] = 1.0
    return y, labels, snomed_ids


def _per_label_auroc(y_true, y_prob):
    scores = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if np.unique(col).size < 2:
            scores.append(np.nan)
        else:
            scores.append(roc_auc_score(col, y_prob[:, i]))
    return np.array(scores, dtype=np.float64)


def _per_label_auprc(y_true, y_prob):
    scores = []
    for i in range(y_true.shape[1]):
        col = y_true[:, i]
        if np.sum(col) == 0:
            scores.append(np.nan)
        else:
            scores.append(average_precision_score(col, y_prob[:, i]))
    return np.array(scores, dtype=np.float64)


def _micro_metric(y_true, y_prob, fn):
    y_flat = y_true.ravel()
    p_flat = y_prob.ravel()
    if np.unique(y_flat).size < 2:
        return np.nan
    return fn(y_flat, p_flat)


def _macro_metric(per_label_scores):
    if np.all(np.isnan(per_label_scores)):
        return np.nan
    return float(np.nanmean(per_label_scores))


def _evaluate(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    per_f1 = f1_score(y_true, y_pred, average=None, zero_division=0)
    micro_f1 = f1_score(y_true, y_pred, average="micro", zero_division=0)
    macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    per_auroc = _per_label_auroc(y_true, y_prob)
    micro_auroc = _micro_metric(y_true, y_prob, roc_auc_score)
    macro_auroc = _macro_metric(per_auroc)

    per_auprc = _per_label_auprc(y_true, y_prob)
    micro_auprc = _micro_metric(y_true, y_prob, average_precision_score)
    macro_auprc = _macro_metric(per_auprc)

    return {
        "micro_f1": float(micro_f1),
        "macro_f1": float(macro_f1),
        "micro_auroc": float(micro_auroc) if not np.isnan(micro_auroc) else np.nan,
        "macro_auroc": float(macro_auroc) if not np.isnan(macro_auroc) else np.nan,
        "micro_auprc": float(micro_auprc) if not np.isnan(micro_auprc) else np.nan,
        "macro_auprc": float(macro_auprc) if not np.isnan(macro_auprc) else np.nan,
        "per_f1": per_f1,
        "per_auroc": per_auroc,
        "per_auprc": per_auprc,
    }


def _align_predictions(ids_a, p_a, ids_b, p_b):
    if ids_a.shape == ids_b.shape and np.all(ids_a == ids_b):
        return ids_a, p_a, p_b

    map_a = {int(eid): i for i, eid in enumerate(ids_a)}
    map_b = {int(eid): i for i, eid in enumerate(ids_b)}
    common = sorted(set(map_a.keys()) & set(map_b.keys()))
    if not common:
        raise ValueError("No common ecg_id between model2 and model3 test sets.")
    idx_a = [map_a[eid] for eid in common]
    idx_b = [map_b[eid] for eid in common]
    return np.array(common, dtype=int), p_a[idx_a], p_b[idx_b]


def main():
    BASE_PATH = Path("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model")
    DATA_PATH = Path("/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1")
    model2_dir = BASE_PATH / "model2_z_oof_all_12sl"
    model3_dir = BASE_PATH / "model3_x_all_12sl"
    labels_path = DATA_PATH / "labels" / "ptbxl_statements.csv"
    disease_label_path = DATA_PATH / "labels" / "disease_label.csv"
    snomed_col = "scp_codes_ext_snomed"
    threshold = 0.5     # 将概率输出转换为二值预测
    delta_metric = "f1"
    delta_threshold = 0.02
    out_dir = Path("train") / "eval_outputs"
    out_dir.mkdir(parents=True, exist_ok=True)

    ids2 = np.load(model2_dir / "test_ecg_id_annot.npy").astype(int)
    ids3 = np.load(model3_dir / "test_ecg_id_annot.npy").astype(int)
    p2 = np.load(model2_dir / "p_disease_test_annot.npy")
    p3 = np.load(model3_dir / "p_disease_test_annot.npy")

    ecg_ids, p2, p3 = _align_predictions(ids2, p2, ids3, p3)

    y_true, labels, snomed_ids = _build_y_true(
        ecg_ids,
        labels_path=labels_path,
        disease_label_path=disease_label_path,
        snomed_col=snomed_col,
    )

    if p2.shape != y_true.shape or p3.shape != y_true.shape:
        raise ValueError(
            f"Shape mismatch: y_true={y_true.shape}, "
            f"p2={p2.shape}, p3={p3.shape}."
        )

    res2 = _evaluate(y_true, p2, threshold=threshold)
    res3 = _evaluate(y_true, p3, threshold=threshold)

    print("\n== Overall (test annot only) ==")
    print(
        f"Model2 Micro-F1={res2['micro_f1']:.4f} Macro-F1={res2['macro_f1']:.4f} "
        f"Micro-AUROC={res2['micro_auroc']:.4f} Macro-AUROC={res2['macro_auroc']:.4f} "
        f"Micro-AUPRC={res2['micro_auprc']:.4f} Macro-AUPRC={res2['macro_auprc']:.4f}"
    )
    print(
        f"Model3 Micro-F1={res3['micro_f1']:.4f} Macro-F1={res3['macro_f1']:.4f} "
        f"Micro-AUROC={res3['micro_auroc']:.4f} Macro-AUROC={res3['macro_auroc']:.4f} "
        f"Micro-AUPRC={res3['micro_auprc']:.4f} Macro-AUPRC={res3['macro_auprc']:.4f}"
    )

    support = y_true.sum(axis=0).astype(int)
    df = pd.DataFrame(
        {
            "label": labels,
            "snomed_id": snomed_ids,
            "support": support,
            "model2_f1": res2["per_f1"],
            "model3_f1": res3["per_f1"],
            "delta_f1": res2["per_f1"] - res3["per_f1"],
            "model2_auroc": res2["per_auroc"],
            "model3_auroc": res3["per_auroc"],
            "delta_auroc": res2["per_auroc"] - res3["per_auroc"],
            "model2_auprc": res2["per_auprc"],
            "model3_auprc": res3["per_auprc"],
            "delta_auprc": res2["per_auprc"] - res3["per_auprc"],
        }
    )

    delta_col = f"delta_{delta_metric}"
    thr = float(delta_threshold)

    # --- NEW: A/B/C grouping ---
    # A: improved, B: neutral, C: declined
    df["group"] = np.where(
        df[delta_col] >= thr,
        "A_improved",
        np.where(df[delta_col] <= -thr, "C_declined", "B_neutral"),
    )

    # --- Keep your improved/declined lists for console display ---
    improved = df[df["group"] == "A_improved"].copy().sort_values(delta_col, ascending=False)
    declined = df[df["group"] == "C_declined"].copy().sort_values(delta_col, ascending=True)

    print(
        f"\n== Per-label delta by {delta_metric} (abs threshold {thr}) =="
    )
    print(
        f"A_improved: {int((df['group']=='A_improved').sum())} | "
        f"B_neutral: {int((df['group']=='B_neutral').sum())} | "
        f"C_declined: {int((df['group']=='C_declined').sum())}"
    )

    if len(improved) > 0:
        print("\nTop improved labels:")
        for _, row in improved.head(20).iterrows():
            print(
                f"  + {row['label']} (snomed {row['snomed_id']}): "
                f"{delta_col}={row[delta_col]:.4f}, support={row['support']}"
            )
    if len(declined) > 0:
        print("\nTop declined labels:")
        for _, row in declined.head(20).iterrows():
            print(
                f"  - {row['label']} (snomed {row['snomed_id']}): "
                f"{delta_col}={row[delta_col]:.4f}, support={row['support']}"
            )

    # --- NEW: group summary table (report-friendly) ---
    # Note: mean/support-weighted summaries are both useful in reports
    summary = (
        df.groupby("group", dropna=False)
        .agg(
            n_labels=("label", "count"),
            total_support=("support", "sum"),
            median_support=("support", "median"),
            mean_support=("support", "mean"),
            mean_model2_f1=("model2_f1", "mean"),
            mean_model3_f1=("model3_f1", "mean"),
            mean_delta=("delta_f1", "mean"),
            mean_delta_metric=(delta_col, "mean"),
        )
        .reset_index()
        .sort_values("group")
    )

    print("\n== Group summary ==")
    print(summary.to_string(index=False))

    # --- NEW: save grouped per-label report + summary ---
    out_csv = out_dir / "model2_vs_model3_per_label_with_group.csv"
    df.sort_values(["group", delta_col], ascending=[True, False]).to_csv(out_csv, index=False)

    out_sum = out_dir / "model2_vs_model3_group_summary.csv"
    summary.to_csv(out_sum, index=False)

    print(f"\nPer-label report (with group) saved to: {out_csv}")
    print(f"Group summary saved to: {out_sum}")



if __name__ == "__main__":
    main()
