import os
import json
import joblib
import numpy as np
import torch

from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import f1_score
from iterstrat.ml_stratifiers import MultilabelStratifiedKFold
from xgboost import XGBClassifier

from dataloader import make_dataloader


# =========================
# Paths
# =========================
LABELS_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv"
FEATURES_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/features/12sl_features.csv"
SELECTED_FEATURES_PATH = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model/selected_features.csv"
FINDING_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/finding_label.csv"
DISEASE_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/disease_label.csv"

SPLIT_DIR = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model/splits"
TRAIN_SPLIT = os.path.join(SPLIT_DIR, "train_ecg_ids.csv")
VAL_SPLIT   = os.path.join(SPLIT_DIR, "val_ecg_ids.csv")
TEST_SPLIT  = os.path.join(SPLIT_DIR, "test_ecg_ids.csv")

OUT_DIR = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model"
os.makedirs(OUT_DIR, exist_ok=True)


# =========================
# Helpers: extract numpy from DataLoader
# =========================
@torch.no_grad()
def loader_to_numpy_model1(loader):
    """
    Returns:
      X: (N, F)
      Y_find: (N, Lf)
      ecg_id: (N,)
    """
    xs, ys, eids = [], [], []
    for batch in loader:
        xs.append(batch["x"].cpu().numpy())
        ys.append(batch["y_finding"].cpu().numpy())
        eids.append(np.array(batch["ecg_id"]))
    X = np.vstack(xs).astype(np.float32)
    Y = np.vstack(ys).astype(np.float32)
    ecg_id = np.concatenate(eids).astype(int)
    return X, Y, ecg_id


@torch.no_grad()
def loader_to_numpy_model2(loader):
    """
    Returns:
      X: (N, F)
      Y_find: (N, Lf)
      Y_dis: (N, Ld)
      dis_mask: (N, Ld)  (all-ones row means disease annotated; all-zeros means unknown)
      ecg_id: (N,)
    """
    xs, yf, yd, m, eids = [], [], [], [], []
    for batch in loader:
        xs.append(batch["x"].cpu().numpy())
        yf.append(batch["y_finding"].cpu().numpy())
        yd.append(batch["y_disease"].cpu().numpy())
        m.append(batch["disease_mask"].cpu().numpy())
        eids.append(np.array(batch["ecg_id"]))
    X = np.vstack(xs).astype(np.float32)
    Yf = np.vstack(yf).astype(np.float32)
    Yd = np.vstack(yd).astype(np.float32)
    M = np.vstack(m).astype(np.float32)
    ecg_id = np.concatenate(eids).astype(int)
    return X, Yf, Yd, M, ecg_id


def build_xgb_multioutput(random_state=42, outer_n_jobs=None, n_labels=None):
    base = XGBClassifier(
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_lambda=1.0,
        min_child_weight=1.0,
        gamma=0.0,
        objective="binary:logistic",
        eval_metric="logloss",
        tree_method="hist",
        n_jobs=1,                 # 关键：每个标签模型内部单线程
        random_state=random_state,
    )
    # 关键：标签维度并行
    if outer_n_jobs is None:
        cpu = os.cpu_count() or 8
        if n_labels is None:
            outer_n_jobs = max(1, cpu // 2)
        else:
            outer_n_jobs = max(1, min(cpu, n_labels, cpu // 2))
    return MultiOutputClassifier(base, n_jobs=outer_n_jobs)

def predict_proba_multioutput(model, X):
    proba_list = model.predict_proba(X)  # list length = L
    cols = []
    for p in proba_list:
        p = np.asarray(p)
        # 二分类时通常是 (N,2)，取正类概率
        if p.ndim == 2 and p.shape[1] == 2:
            cols.append(p[:, 1])
        else:
            cols.append(p.reshape(-1))
    return np.stack(cols, axis=1)

def save_np(path, arr):
    np.save(path, arr, allow_pickle=False)


# =========================
# Build loaders and oof
# =========================
def make_loaders_for_model1(selected_features_path=None, batch_size=2048):
    train_loader = make_dataloader(
        split_ids_path=TRAIN_SPLIT,
        labels_path=LABELS_PATH,
        features_path=FEATURES_PATH,
        finding_label_path=FINDING_LABEL_PATH,
        selected_features_path=selected_features_path,
        task="model1",
        batch_size=batch_size,
        shuffle=False,      # IMPORTANT: keep stable order for saving ecg_id aligned outputs
        num_workers=2,
    )
    val_loader = make_dataloader(
        split_ids_path=VAL_SPLIT,
        labels_path=LABELS_PATH,
        features_path=FEATURES_PATH,
        finding_label_path=FINDING_LABEL_PATH,
        selected_features_path=selected_features_path,
        task="model1",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_loader = make_dataloader(
        split_ids_path=TEST_SPLIT,
        labels_path=LABELS_PATH,
        features_path=FEATURES_PATH,
        finding_label_path=FINDING_LABEL_PATH,
        selected_features_path=selected_features_path,
        task="model1",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, val_loader, test_loader

def make_loaders_for_model2(selected_features_path=None, batch_size=2048):
    train_loader = make_dataloader(
        split_ids_path=TRAIN_SPLIT,
        labels_path=LABELS_PATH,
        features_path=FEATURES_PATH,
        finding_label_path=FINDING_LABEL_PATH,
        disease_label_path=DISEASE_LABEL_PATH,
        selected_features_path=selected_features_path,
        task="model2",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    val_loader = make_dataloader(
        split_ids_path=VAL_SPLIT,
        labels_path=LABELS_PATH,
        features_path=FEATURES_PATH,
        finding_label_path=FINDING_LABEL_PATH,
        disease_label_path=DISEASE_LABEL_PATH,
        selected_features_path=selected_features_path,
        task="model2",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    test_loader = make_dataloader(
        split_ids_path=TEST_SPLIT,
        labels_path=LABELS_PATH,
        features_path=FEATURES_PATH,
        finding_label_path=FINDING_LABEL_PATH,
        disease_label_path=DISEASE_LABEL_PATH,
        selected_features_path=selected_features_path,
        task="model2",
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return train_loader, val_loader, test_loader

def build_oof_p_finding_with_kfold(
    X_train: np.ndarray,
    Yf_train: np.ndarray,
    ecg_id_train: np.ndarray,
    n_splits: int = 5,
    random_state: int = 42,
):
    """
    Build out-of-fold p_finding for the TRAIN split only.

    Returns:
      p_oof: (N_train, Lf) out-of-fold probabilities
      model1_final: model trained on full TRAIN (for val/test inference)
    """
    mskf = MultilabelStratifiedKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=random_state
    )

    n = X_train.shape[0]
    lf = Yf_train.shape[1]
    p_oof = np.zeros((n, lf), dtype=np.float32)

    # 注意：MultilabelStratifiedKFold 需要 y 参与 split
    # 你的 Yf_train 是 float32 的 0/1，最好显式转 int
    Y_split = (Yf_train > 0.5).astype(np.int32)

    for fold, (tr_idx, va_idx) in enumerate(mskf.split(X_train, Y_split), start=1):
        X_tr, Y_tr = X_train[tr_idx], Yf_train[tr_idx]
        X_va = X_train[va_idx]

        model_fold = build_xgb_multioutput(
            random_state=random_state + fold,
            outer_n_jobs=os.cpu_count() // 2,
            n_labels=Y_tr.shape[1],
        )
        model_fold.fit(X_tr, Y_tr)

        p_fold = predict_proba_multioutput(model_fold, X_va).astype(np.float32)
        p_oof[va_idx] = p_fold

        print(f"[OOF] Fold {fold}/{n_splits}: train={len(tr_idx)}, oof={len(va_idx)}")

    # train final model1 on full train for val/test inference
    model1_final = build_xgb_multioutput(
        random_state=random_state + 999,
        outer_n_jobs=os.cpu_count() // 2,
        n_labels=Yf_train.shape[1],
    )
    model1_final.fit(X_train, Yf_train)

    if np.any(np.isnan(p_oof)) or (p_oof.sum(axis=1) == 0).all():
        print("[OOF] Warning: p_oof looks suspicious (all zeros or NaNs). Please verify.")

    return p_oof, model1_final

# =========================
# Train model1: x -> p_finding(x)
# =========================
def train_model1(selected_features_path=None, run_name="model1_all"):
    train_loader, val_loader, test_loader = make_loaders_for_model1(selected_features_path)

    Xtr, Yf_tr, eid_tr = loader_to_numpy_model1(train_loader)
    Xva, Yf_va, eid_va = loader_to_numpy_model1(val_loader)
    Xte, Yf_te, eid_te = loader_to_numpy_model1(test_loader)

    print(f"[model1] Xtr={Xtr.shape}, Yf_tr={Yf_tr.shape}")

    clf = build_xgb_multioutput(random_state=42, outer_n_jobs=os.cpu_count() // 2, n_labels=Yf_tr.shape[1])
    clf.fit(Xtr, Yf_tr)

    # probabilities
    Ptr = predict_proba_multioutput(clf, Xtr)
    Pva = predict_proba_multioutput(clf, Xva).astype(np.float32)
    Pte = predict_proba_multioutput(clf, Xte).astype(np.float32)

    # quick sanity metric (threshold 0.5)
    pred_va = (Pva >= 0.5).astype(int)
    micro_f1 = f1_score(Yf_va.astype(int), pred_va, average="micro", zero_division=0)
    macro_f1 = f1_score(Yf_va.astype(int), pred_va, average="macro", zero_division=0)
    print(f"[model1] Val Micro-F1={micro_f1:.4f}, Macro-F1={macro_f1:.4f}")

    out = os.path.join(OUT_DIR, run_name)
    os.makedirs(out, exist_ok=True)

    joblib.dump(clf, os.path.join(out, "model1_xgb_ovr.joblib"))

    # save probabilities aligned with ecg_id (for model2)
    np.save(os.path.join(out, "train_ecg_id.npy"), eid_tr)
    np.save(os.path.join(out, "val_ecg_id.npy"), eid_va)
    np.save(os.path.join(out, "test_ecg_id.npy"), eid_te)

    np.save(os.path.join(out, "p_finding_train.npy"), Ptr)
    np.save(os.path.join(out, "p_finding_val.npy"), Pva)
    np.save(os.path.join(out, "p_finding_test.npy"), Pte)

    meta = {
        "run_name": run_name,
        "selected_features_path": selected_features_path,
        "num_features": int(Xtr.shape[1]),
        "num_finding_labels": int(Yf_tr.shape[1]),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[model1] saved to {out}")
    return out


# =========================
# Train model2: z(x)=[x, p_finding(x)] -> disease probabilities
# =========================
def train_model2_with_oof(
    selected_features_path=None,
    run_name="model2_z_oof_all_12sl",
    n_splits_oof: int = 5,
):
    """
    model2 training with OOF p_finding on TRAIN split:
      - train: z=[x, p_finding_oof]
      - val/test: z=[x, p_finding_pred_by_model1_trained_on_full_train]

    disease training/eval ONLY on disease-annotated subset (mask>0).
    """
    train_loader, val_loader, test_loader = make_loaders_for_model2(selected_features_path)

    Xtr, Yf_tr, Yd_tr, M_tr, eid_tr = loader_to_numpy_model2(train_loader)
    Xva, Yf_va, Yd_va, M_va, eid_va = loader_to_numpy_model2(val_loader)
    Xte, Yf_te, Yd_te, M_te, eid_te = loader_to_numpy_model2(test_loader)

    # --------- Build OOF p_finding for TRAIN and final model1 for val/test ---------
    print(f"[model2][OOF] building OOF p_finding on TRAIN with {n_splits_oof}-fold ...")
    Ptr_oof, model1_final = build_oof_p_finding_with_kfold(
        X_train=Xtr,
        Yf_train=Yf_tr,
        ecg_id_train=eid_tr,
        n_splits=n_splits_oof,
        random_state=42,
    )

    # val/test p_finding using final model1 trained on full TRAIN
    Pva = predict_proba_multioutput(model1_final, Xva).astype(np.float32)
    Pte = predict_proba_multioutput(model1_final, Xte).astype(np.float32)

    # z = [x, p_finding]
    Ztr = np.concatenate([Xtr, Ptr_oof], axis=1).astype(np.float32)
    Zva = np.concatenate([Xva, Pva], axis=1).astype(np.float32)
    Zte = np.concatenate([Xte, Pte], axis=1).astype(np.float32)

    # --------- Filter to disease-annotated samples ---------
    tr_has = (M_tr.sum(axis=1) > 0)
    va_has = (M_va.sum(axis=1) > 0)
    te_has = (M_te.sum(axis=1) > 0)

    Ztr2, Yd_tr2 = Ztr[tr_has], Yd_tr[tr_has]
    Zva2, Yd_va2 = Zva[va_has], Yd_va[va_has]
    Zte2, Yd_te2 = Zte[te_has], Yd_te[te_has]

    print(f"[model2] Ztr(all)={Ztr.shape}, Ztr(annot)={Ztr2.shape}, Yd_tr2={Yd_tr2.shape}")
    print(f"[model2] Val annot={Zva2.shape}, Test annot={Zte2.shape}")

    # --------- Train model2 ---------
    clf = build_xgb_multioutput(random_state=43, outer_n_jobs=os.cpu_count() // 2, n_labels=Yd_tr2.shape[1])
    clf.fit(Ztr2, Yd_tr2.astype(np.int32))

    # --------- Evaluate on VAL annotated subset ---------
    Pva_d = predict_proba_multioutput(clf, Zva2).astype(np.float32)
    pred_va = (Pva_d >= 0.5).astype(int)

    micro_f1 = f1_score(Yd_va2.astype(int), pred_va, average="micro", zero_division=0)
    macro_f1 = f1_score(Yd_va2.astype(int), pred_va, average="macro", zero_division=0)
    print(f"[model2][OOF] Val(annot only) Micro-F1={micro_f1:.4f}, Macro-F1={macro_f1:.4f}")

    # --------- Save ---------
    out = os.path.join(OUT_DIR, run_name)
    os.makedirs(out, exist_ok=True)

    joblib.dump(clf, os.path.join(out, "model2_xgb_ovr.joblib"))

    # Save final model1 used for val/test p_finding (for reproducibility)
    joblib.dump(model1_final, os.path.join(out, "model1_final_for_z.joblib"))

    # Save OOF p_finding for train + p_finding for val/test
    np.save(os.path.join(out, "train_ecg_id.npy"), eid_tr)
    np.save(os.path.join(out, "val_ecg_id.npy"), eid_va)
    np.save(os.path.join(out, "test_ecg_id.npy"), eid_te)

    np.save(os.path.join(out, "p_finding_train_oof.npy"), Ptr_oof)
    np.save(os.path.join(out, "p_finding_val.npy"), Pva)
    np.save(os.path.join(out, "p_finding_test.npy"), Pte)

    # Save disease predictions on annotated subsets (honest evaluation scope)
    np.save(os.path.join(out, "val_ecg_id_annot.npy"), eid_va[va_has])
    np.save(os.path.join(out, "test_ecg_id_annot.npy"), eid_te[te_has])

    np.save(os.path.join(out, "p_disease_val_annot.npy"), Pva_d)
    Pte_d = predict_proba_multioutput(clf, Zte2).astype(np.float32)
    np.save(os.path.join(out, "p_disease_test_annot.npy"), Pte_d)

    meta = {
        "run_name": run_name,
        "selected_features_path": selected_features_path,
        "n_splits_oof": int(n_splits_oof),
        "num_x_features": int(Xtr.shape[1]),
        "num_finding_labels": int(Yf_tr.shape[1]),
        "num_z_features": int(Ztr.shape[1]),
        "num_disease_labels": int(Yd_tr.shape[1]),
        "train_annotated_count": int(tr_has.sum()),
        "val_annotated_count": int(va_has.sum()),
        "test_annotated_count": int(te_has.sum()),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[model2][OOF] saved to {out}")
    return out

# =========================
# Train model3: x -> disease probabilities (baseline)
# =========================
def train_model3(selected_features_path=None, run_name="model3_x_all"):
    train_loader, val_loader, test_loader = make_loaders_for_model2(selected_features_path)

    Xtr, Yf_tr, Yd_tr, M_tr, eid_tr = loader_to_numpy_model2(train_loader)
    Xva, Yf_va, Yd_va, M_va, eid_va = loader_to_numpy_model2(val_loader)
    Xte, Yf_te, Yd_te, M_te, eid_te = loader_to_numpy_model2(test_loader)

    tr_has = (M_tr.sum(axis=1) > 0)
    va_has = (M_va.sum(axis=1) > 0)
    te_has = (M_te.sum(axis=1) > 0)

    Xtr2, Yd_tr2 = Xtr[tr_has], Yd_tr[tr_has]
    Xva2, Yd_va2 = Xva[va_has], Yd_va[va_has]
    Xte2, Yd_te2 = Xte[te_has], Yd_te[te_has]

    print(f"[model3] Xtr(all)={Xtr.shape}, Xtr(annot)={Xtr2.shape}, Yd_tr2={Yd_tr2.shape}")

    clf = build_xgb_multioutput(random_state=44, outer_n_jobs=os.cpu_count() // 2, n_labels=Yd_tr2.shape[1])
    clf.fit(Xtr2, Yd_tr2.astype(np.int32))

    Pva = predict_proba_multioutput(clf, Xva2).astype(np.float32)
    pred_va = (Pva >= 0.5).astype(int)

    micro_f1 = f1_score(Yd_va2.astype(int), pred_va, average="micro", zero_division=0)
    macro_f1 = f1_score(Yd_va2.astype(int), pred_va, average="macro", zero_division=0)
    print(f"[model3] Val(annot only) Micro-F1={micro_f1:.4f}, Macro-F1={macro_f1:.4f}")

    out = os.path.join(OUT_DIR, run_name)
    os.makedirs(out, exist_ok=True)

    joblib.dump(clf, os.path.join(out, "model3_xgb_ovr.joblib"))

    np.save(os.path.join(out, "val_ecg_id_annot.npy"), eid_va[va_has])
    np.save(os.path.join(out, "test_ecg_id_annot.npy"), eid_te[te_has])

    np.save(os.path.join(out, "p_disease_val_annot.npy"), Pva)
    Pte   = predict_proba_multioutput(clf, Xte2).astype(np.float32)
    np.save(os.path.join(out, "p_disease_test_annot.npy"), Pte)

    meta = {
        "run_name": run_name,
        "selected_features_path": selected_features_path,
        "num_x_features": int(Xtr.shape[1]),
        "num_disease_labels": int(Yd_tr.shape[1]),
        "train_annotated_count": int(tr_has.sum()),
        "val_annotated_count": int(va_has.sum()),
        "test_annotated_count": int(te_has.sum()),
    }
    with open(os.path.join(out, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

    print(f"[model3] saved to {out}")
    return out


# =========================
# Main
# =========================
if __name__ == "__main__":
    # 1) Train model1: x -> p_finding(x)
    m1_dir = train_model1(selected_features_path=None, run_name="model1_all_12sl")

    # 2) Train model2: z=[x, p_finding(x)] -> disease
    train_model2_with_oof(
        selected_features_path=None,
        run_name="model2_z_oof_all_12sl",
        n_splits_oof=5,
    )

    # 3) Train model3 baseline: x -> disease
    train_model3(selected_features_path=None, run_name="model3_x_all_12sl")
