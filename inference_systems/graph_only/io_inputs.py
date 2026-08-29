"""
职责：只做“加载与对齐”，不做推理逻辑
输入：你那些 *.npy/meta.json
输出：标准化 batch 对象（dict 或 dataclass）
包含：
    load_split_ids(model2_dir)
    load_probs(model2_dir, split="val_annot")
    assert_alignment(ecg_ids, p_finding, p_disease, Y_true)
    return {"ecg_id": ..., "p_finding": ..., "p_disease": ..., "Y": ...}
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Literal, Optional, Tuple

import numpy as np


SplitName = Literal["train", "val", "test", "val_annot", "test_annot"]


@dataclass(frozen=True)
class InferenceBatch:
    """Standardized batch for downstream reasoning/eval."""
    split: str
    ecg_id: np.ndarray              # shape: (N,)
    p_finding: Optional[np.ndarray] # shape: (N, n_finding) or None
    p_disease: Optional[np.ndarray] # shape: (N, n_disease) or None
    y_true: Optional[np.ndarray]    # shape: (N, n_disease) or None
    meta: Dict[str, Any]            # merged metadata from dirs (if present)


# -----------------------------
# Basic file helpers
# -----------------------------

def _to_path(p: str | Path) -> Path:
    return p if isinstance(p, Path) else Path(p)


def load_json(path: str | Path) -> Dict[str, Any]:
    path = _to_path(path)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_npy(path: str | Path) -> np.ndarray:
    path = _to_path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    arr = np.load(path, allow_pickle=False)
    return arr


def _ensure_1d_ids(ecg_id: np.ndarray) -> np.ndarray:
    ecg_id = np.asarray(ecg_id)
    if ecg_id.ndim != 1:
        ecg_id = ecg_id.reshape(-1)
    # PTB-XL ecg_id are ints; cast if possible
    if ecg_id.dtype.kind not in ("i", "u"):
        # handle strings that represent ints
        try:
            ecg_id = ecg_id.astype(np.int64)
        except Exception:
            pass
    return ecg_id


def _assert_probability_matrix(p: np.ndarray, name: str) -> None:
    p = np.asarray(p)
    if p.ndim != 2:
        raise ValueError(f"{name} must be 2D (N, C). Got shape={p.shape}")
    if not np.isfinite(p).all():
        raise ValueError(f"{name} contains non-finite values.")
    # permissive range check (allow slight numeric drift)
    if (p < -1e-6).any() or (p > 1 + 1e-6).any():
        raise ValueError(f"{name} contains values outside [0,1] (with tolerance).")


# -----------------------------
# Split loading
# -----------------------------
def finding_id_file_for_split(split: SplitName) -> str:
    # model2_z_oof_all_12sl conventions
    if split == "train":
        return "train_ecg_id.npy"
    if split in ("val", "val_annot"):
        return "val_ecg_id.npy"      # IMPORTANT: p_finding_val.npy aligns to val_ecg_id.npy
    if split in ("test", "test_annot"):
        return "test_ecg_id.npy"     # p_finding_test.npy aligns to test_ecg_id.npy
    raise ValueError(split)

def disease_id_file_for_split(split: SplitName) -> str:
    # p_disease_*_annot aligns to *_ecg_id_annot.npy
    if split == "val_annot":
        return "val_ecg_id_annot.npy"
    if split == "test_annot":
        return "test_ecg_id_annot.npy"
    # non-annot splits typically not present
    if split in ("train", "val", "test"):
        return f"{split}_ecg_id.npy"
    raise ValueError(split)

def split_files_for_model2(split: SplitName) -> Tuple[str, Optional[str]]:
    """
    Return (ecg_id_filename, p_disease_filename) for model2 directory.
    p_disease may be None for splits where you did not save it (e.g., train).
    """
    if split == "train":
        return "train_ecg_id.npy", None
    if split == "val":
        return "val_ecg_id.npy", None
    if split == "test":
        return "test_ecg_id.npy", None
    if split == "val_annot":
        return "val_ecg_id_annot.npy", "p_disease_val_annot.npy"
    if split == "test_annot":
        return "test_ecg_id_annot.npy", "p_disease_test_annot.npy"
    raise ValueError(f"Unknown split: {split}")


def load_split_ids(model_dir: str | Path, split: SplitName, *, prefer_model2: bool = True) -> np.ndarray:
    """
    Load ecg_id list for a split from a model directory.

    By convention your directories store e.g.:
      - train_ecg_id.npy
      - val_ecg_id.npy
      - test_ecg_id.npy
      - val_ecg_id_annot.npy
      - test_ecg_id_annot.npy
    """
    model_dir = _to_path(model_dir)
    if prefer_model2:
        ecg_id_file, _ = split_files_for_model2(split)
        ecg_id = load_npy(model_dir / ecg_id_file)
        return _ensure_1d_ids(ecg_id)

    # Generic fallback: try a few common names
    candidates = [
        f"{split}_ecg_id.npy",
        f"{split}.npy",
    ]
    for fn in candidates:
        p = model_dir / fn
        if p.exists():
            return _ensure_1d_ids(load_npy(p))
    raise FileNotFoundError(f"Could not find ecg_id file for split={split} in {model_dir}")


# -----------------------------
# Probability loading
# -----------------------------

def load_p_finding(model_dir: str | Path, split: SplitName) -> Optional[np.ndarray]:
    """
    Load p_finding from a directory. Supports your saved naming conventions.

    model2_z_oof_all_12sl:
      - train: p_finding_train_oof.npy
      - val / val_annot: p_finding_val.npy
      - test / test_annot: p_finding_test.npy

    model1_all_12sl:
      - train: p_finding_train.npy
      - val: p_finding_val.npy
      - test: p_finding_test.npy
    """
    model_dir = _to_path(model_dir)

    # Prefer model2 naming if available
    mapping = {
        "train": ["p_finding_train_oof.npy", "p_finding_train.npy"],
        "val": ["p_finding_val.npy"],
        "val_annot": ["p_finding_val.npy"],
        "test": ["p_finding_test.npy"],
        "test_annot": ["p_finding_test.npy"],
    }
    for fn in mapping[str(split)]:
        p = model_dir / fn
        if p.exists():
            arr = load_npy(p)
            _assert_probability_matrix(arr, f"p_finding({split})")
            return arr

    # If missing, return None (some workflows might not need p_finding)
    return None


def load_p_disease(model_dir: str | Path, split: SplitName) -> Optional[np.ndarray]:
    """
    Load p_disease from a directory. Supports your saved naming conventions.

    model2_z_oof_all_12sl:
      - val_annot: p_disease_val_annot.npy
      - test_annot: p_disease_test_annot.npy

    model3_x_all_12sl:
      - val_annot: p_disease_val_annot.npy
      - test_annot: p_disease_test_annot.npy
    """
    model_dir = _to_path(model_dir)

    # Typical naming
    mapping = {
        "val_annot": ["p_disease_val_annot.npy"],
        "test_annot": ["p_disease_test_annot.npy"],
        # non-annot splits often not saved; keep None
        "train": [],
        "val": [],
        "test": [],
    }
    for fn in mapping[str(split)]:
        p = model_dir / fn
        if p.exists():
            arr = load_npy(p)
            _assert_probability_matrix(arr, f"p_disease({split})")
            return arr
    return None


# -----------------------------
# Alignment by ecg_id
# -----------------------------

def align_by_ecg_id(
    ref_ecg_id: np.ndarray,
    other_ecg_id: np.ndarray,
    other_array: np.ndarray,
    *,
    name: str,
    strict: bool = True,
    ) -> np.ndarray:
    """
    Align rows of other_array to match ref_ecg_id order.

    - strict=True: require identical sets; error on missing/extra ids.
    - strict=False: use intersection; drop ids not in both sets.
      (Note: this changes N; you must apply the same intersection to all arrays.)
    """
    ref_ecg_id = _ensure_1d_ids(ref_ecg_id)
    other_ecg_id = _ensure_1d_ids(other_ecg_id)

    if other_array.shape[0] != other_ecg_id.shape[0]:
        raise ValueError(
            f"{name}: other_array first dim {other_array.shape[0]} "
            f"!= other_ecg_id length {other_ecg_id.shape[0]}"
        )

    ref_set = set(ref_ecg_id.tolist())
    other_set = set(other_ecg_id.tolist())

    if strict:
        missing = ref_set - other_set
        extra = other_set - ref_set
        if missing or extra:
            raise ValueError(
                f"{name}: ecg_id mismatch under strict alignment.\n"
                f"  missing_in_other={len(missing)} extra_in_other={len(extra)}"
            )

    # Build index map for other
    idx_map: Dict[int, int] = {int(eid): i for i, eid in enumerate(other_ecg_id.tolist())}

    if strict:
        take_idx = np.array([idx_map[int(eid)] for eid in ref_ecg_id.tolist()], dtype=np.int64)
        return other_array[take_idx]

    # Non-strict: intersect; return only ids present in other
    keep_mask = np.array([int(eid) in idx_map for eid in ref_ecg_id.tolist()], dtype=bool)
    keep_ids = ref_ecg_id[keep_mask]
    take_idx = np.array([idx_map[int(eid)] for eid in keep_ids.tolist()], dtype=np.int64)
    return other_array[take_idx]


def intersect_ids(*id_arrays: np.ndarray) -> np.ndarray:
    """Compute sorted intersection of multiple 1D id arrays."""
    sets = [set(_ensure_1d_ids(a).tolist()) for a in id_arrays]
    inter = set.intersection(*sets) if sets else set()
    return np.array(sorted(inter), dtype=np.int64)

# -----------------------------
# Ground-truth (Y) loader for PTB-XL disease labels
# -----------------------------

import ast
import pandas as pd

def _parse_snomed_list(cell: str) -> set[int]:
    """
    Parse scp_codes_ext_snomed cell like:
      "[(320536, 100.0), (441840, 100.0), ...]"
    Return a set of SNOMED IDs (ints).
    """
    if not isinstance(cell, str) or cell.strip() == "":
        return set()
    try:
        items = ast.literal_eval(cell)
        return {int(code) for code, _ in items}
    except Exception:
        return set()


def build_y_loader(
    *,
    ptbxl_statements_csv: str | Path,
    disease_label_csv: str | Path,
    ):
    """
    Build a y_loader(ecg_id, split) callable that returns
    Y_true with shape (N, n_disease) in disease_label.csv order.

    Assumptions (matching your model2 logic):
    - Only annotated samples (ecg_id_annot) are passed in.
    - A disease label is positive iff its SNOMED appears in
      scp_codes_ext_snomed for that ecg_id.
    - All annotated samples are fully labeled (no partial mask).
    """

    ptbxl_statements_csv = _to_path(ptbxl_statements_csv)
    disease_label_csv = _to_path(disease_label_csv)

    # ---- load disease label vocabulary ----
    df_label = pd.read_csv(disease_label_csv)
    if "label" not in df_label.columns or "snomed_id" not in df_label.columns:
        raise ValueError(
            "disease_label.csv must contain columns ['label', 'snomed_id']"
        )

    disease_snomeds = df_label["snomed_id"].astype(int).tolist()
    n_disease = len(disease_snomeds)

    snomed_to_col = {
        int(snomed): idx for idx, snomed in enumerate(disease_snomeds)
    }

    # ---- load ptbxl statements ----
    df_stmt = pd.read_csv(ptbxl_statements_csv)
    if "ecg_id" not in df_stmt.columns or "scp_codes_ext_snomed" not in df_stmt.columns:
        raise ValueError(
            "ptbxl_statements.csv must contain columns "
            "['ecg_id', 'scp_codes_ext_snomed']"
        )

    # Map ecg_id -> set of SNOMEDs
    ecg_to_snomeds: dict[int, set[int]] = {}
    for _, row in df_stmt.iterrows():
        ecg_id = int(row["ecg_id"])
        snomed_set = _parse_snomed_list(row["scp_codes_ext_snomed"])
        ecg_to_snomeds[ecg_id] = snomed_set

    # ---- the actual loader ----
    def y_loader(ecg_id: np.ndarray, split: SplitName) -> np.ndarray:
        ecg_id = _ensure_1d_ids(ecg_id)
        N = ecg_id.shape[0]
        Y = np.zeros((N, n_disease), dtype=np.int8)

        for i, eid in enumerate(ecg_id.tolist()):
            snomeds = ecg_to_snomeds.get(int(eid), set())
            # Only disease SNOMEDs matter
            for snomed in snomeds:
                col = snomed_to_col.get(int(snomed))
                if col is not None:
                    Y[i, col] = 1

        return Y

    return y_loader

# -----------------------------
# Batch assembly
# -----------------------------

def load_inference_batch(
    *,
    split: SplitName,
    # Authority split directory (recommended: model2_z_oof_all_12sl)
    split_dir: str | Path,
    # From where to load p_finding (recommended: same as split_dir for model2)
    finding_dir: Optional[str | Path] = None,
    # From where to load p_disease (recommended: model2_z_oof_all_12sl; or model3 for ablation)
    disease_dir: Optional[str | Path] = None,
    # How to load ground-truth labels:
    y_loader: Optional[Callable[[np.ndarray, SplitName], np.ndarray]] = None,
    strict_alignment: bool = True,
    ) -> InferenceBatch:
    """
    Load ecg_id for split, probabilities, optionally Y, and align everything by ecg_id.

    y_loader signature:
        y = y_loader(ecg_id, split)
    where y is (N, n_disease) in the SAME disease label order as p_disease.

    Notes:
    - If p_disease is None for a split, you can still build a batch (e.g., train debug).
    - If strict_alignment=False, this function will intersect ids across available arrays.
      (Use with care; recommended to keep strict=True to avoid silent leakage.)
    """
    split_dir = _to_path(split_dir)
    finding_dir = _to_path(finding_dir) if finding_dir is not None else split_dir
    disease_dir = _to_path(disease_dir) if disease_dir is not None else split_dir

    # IDs (authority)
    ecg_id = load_split_ids(split_dir, split, prefer_model2=True)

    # Load probabilities
    p_finding = load_p_finding(finding_dir, split)
    p_disease = load_p_disease(disease_dir, split)
    # IDs (authority)
    ecg_id = load_split_ids(split_dir, split, prefer_model2=True)

    # Load probabilities
    p_finding = load_p_finding(finding_dir, split)
    p_disease = load_p_disease(disease_dir, split)

    # --- Align p_finding to authority ecg_id if needed ---
    if p_finding is not None:
        fid_file = finding_id_file_for_split(split)
        finding_ecg_id = load_npy(finding_dir / fid_file)
        finding_ecg_id = _ensure_1d_ids(finding_ecg_id)

        if finding_ecg_id.shape[0] != p_finding.shape[0]:
            raise ValueError(
                f"p_finding rows {p_finding.shape[0]} != {fid_file} length {finding_ecg_id.shape[0]}"
            )

        # Align to authority ecg_id (val_annot/test_annot are subsets)
        p_finding = align_by_ecg_id(
            ecg_id, finding_ecg_id, p_finding, name=f"p_finding[{split}]", strict=False
        )
        # When strict=False, result length may shrink (intersection). Keep ecg_id consistent:
        # We need to intersect ecg_id to those present in finding_ecg_id.
        inter = intersect_ids(ecg_id, finding_ecg_id)
        ecg_id = inter

    # --- Align p_disease to authority ecg_id if needed ---
    if p_disease is not None:
        did_file = disease_id_file_for_split(split)
        disease_ecg_id = load_npy(disease_dir / did_file)
        disease_ecg_id = _ensure_1d_ids(disease_ecg_id)

        if disease_ecg_id.shape[0] != p_disease.shape[0]:
            raise ValueError(
                f"p_disease rows {p_disease.shape[0]} != {did_file} length {disease_ecg_id.shape[0]}"
            )

        # p_disease should already match ecg_id for *_annot, but align defensively
        p_disease = align_by_ecg_id(
            ecg_id, disease_ecg_id, p_disease, name=f"p_disease[{split}]", strict=False
        )
        inter = intersect_ids(ecg_id, disease_ecg_id)
        ecg_id = inter


    # If non-strict, intersect ids across loaded arrays (and then re-align)
    if not strict_alignment:
        ids_to_intersect = [ecg_id]
        # if a directory contains its own ecg_id list, user should pass it; we assume same as ecg_id.
        # So intersection is only meaningful when y_loader returns subset; handled below.
        inter = intersect_ids(*ids_to_intersect)
        ecg_id = inter

    # Load Y
    y_true = None
    if y_loader is not None:
        y_true = y_loader(ecg_id, split)
        y_true = np.asarray(y_true)
        if y_true.ndim != 2 or y_true.shape[0] != ecg_id.shape[0]:
            raise ValueError(
                f"y_true must be 2D (N, C) with N={ecg_id.shape[0]}. Got {y_true.shape}"
            )

    meta = {}
    # Merge meta.json if present
    meta.update(load_json(split_dir / "meta.json"))
    if finding_dir != split_dir:
        meta.update({f"finding_meta::{finding_dir.name}": load_json(finding_dir / "meta.json")})
    if disease_dir != split_dir:
        meta.update({f"disease_meta::{disease_dir.name}": load_json(disease_dir / "meta.json")})

    return InferenceBatch(
        split=str(split),
        ecg_id=ecg_id,
        p_finding=p_finding,
        p_disease=p_disease,
        y_true=y_true,
        meta=meta,
    )


# -----------------------------
# Convenience: model2 default batch
# -----------------------------

def load_model2_batches(
    model2_dir: str | Path,
    *,
    y_loader: Optional[Callable[[np.ndarray, SplitName], np.ndarray]] = None,
    strict_alignment: bool = True,
    ) -> Dict[str, InferenceBatch]:
    """
    Convenience loader for the typical workflow:
      - train: ids + p_finding_train_oof (no p_disease expected)
      - val_annot: ids + p_finding_val + p_disease_val_annot + y_true(optional)
      - test_annot: ids + p_finding_test + p_disease_test_annot + y_true(optional)
    """
    model2_dir = _to_path(model2_dir)
    out: Dict[str, InferenceBatch] = {}
    out["train"] = load_inference_batch(
        split="train",
        split_dir=model2_dir,
        finding_dir=model2_dir,
        disease_dir=model2_dir,
        y_loader=y_loader,
        strict_alignment=strict_alignment,
    )
    out["val_annot"] = load_inference_batch(
        split="val_annot",
        split_dir=model2_dir,
        finding_dir=model2_dir,
        disease_dir=model2_dir,
        y_loader=y_loader,
        strict_alignment=strict_alignment,
    )
    out["test_annot"] = load_inference_batch(
        split="test_annot",
        split_dir=model2_dir,
        finding_dir=model2_dir,
        disease_dir=model2_dir,
        y_loader=y_loader,
        strict_alignment=strict_alignment,
    )
    return out
