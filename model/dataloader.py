"""
LABELS_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv"
FEATURES_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/features/12sl_features.csv"
SELECTED_FEATURES_PATH = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model/selected_features.csv"
FINDING_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/finding_label.csv"
DISEASE_LABEL_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/disease_label.csv"

SPLIT_DIR = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/splits"
train_split = os.path.join(SPLIT_DIR, "train_ecg_ids.csv")
val_split   = os.path.join(SPLIT_DIR, "val_ecg_ids.csv")
test_split  = os.path.join(SPLIT_DIR, "test_ecg_ids.csv")

# -----------------------
# model_1 dataloader
# -----------------------
train_loader_m1 = make_dataloader(
    split_ids_path=train_split,
    labels_path=LABELS_PATH,
    features_path=FEATURES_PATH,
    finding_label_path=FINDING_LABEL_PATH,
    selected_features_path=None,  # or SELECTED_FEATURES_PATH
    task="model1",
    batch_size=256,
    shuffle=True,
)

# -----------------------
# model_2 dataloader
# -----------------------
train_loader_m2 = make_dataloader(
    split_ids_path=train_split,
    labels_path=LABELS_PATH,
    features_path=FEATURES_PATH,
    finding_label_path=FINDING_LABEL_PATH,
    disease_label_path=DISEASE_LABEL_PATH,
    selected_features_path=None,  # or SELECTED_FEATURES_PATH
    task="model2",
    batch_size=256,
    shuffle=True,
)

"""
import os
import json
import ast
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


def _safe_parse_list_of_tuples(x: str):
    """
    Parse strings like:
      "[('NORM', 100.0), ('LVOLT', 100.0)]"
      "[(320536, 100.0), (441840, 100.0)]"
    Return a Python list, or [] if empty/invalid.
    """
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


def _load_split_ids(split_path: str) -> List[int]:
    """
    split_path can be:
      - ecg_splits.json  (keys: train/val/test)
      - train_ecg_ids.csv / val_ecg_ids.csv / test_ecg_ids.csv
    """
    if split_path.endswith(".json"):
        with open(split_path, "r") as f:
            data = json.load(f)
        # Caller should select the key externally
        raise ValueError("For JSON split, please pass split_json_path and split_key to the factory.")
    else:
        df = pd.read_csv(split_path)
        if "ecg_id" not in df.columns:
            raise ValueError(f"Split CSV must contain 'ecg_id'. Got columns: {df.columns.tolist()}")
        return df["ecg_id"].astype(int).tolist()


def _build_label_vocab(label_csv_path: str) -> Tuple[List[str], List[int], Dict[int, int]]:
    """
    label csv format:
      label,snomed_id
      Acute myocardial infarction,312327
    Return:
      labels: List[str]
      snomed_ids: List[int]
      snomed_to_index: Dict[snomed_id -> idx]
    """
    df = pd.read_csv(label_csv_path)
    if "label" not in df.columns or "snomed_id" not in df.columns:
        raise ValueError(f"Label file must contain columns ['label','snomed_id']: {label_csv_path}")
    labels = df["label"].astype(str).tolist()
    snomed_ids = df["snomed_id"].astype(int).tolist()
    snomed_to_index = {sid: i for i, sid in enumerate(snomed_ids)}
    return labels, snomed_ids, snomed_to_index


class PTBXLFeatureDataset(Dataset):
    """
    Dataset that returns:
      For model_1:
        x, y_finding, ecg_id
      For model_2:
        x, y_finding, y_disease, disease_mask, ecg_id

    disease_mask:
      - shape [num_disease_labels]
      - if this ECG has ANY disease label present in statements: mask = 1 for all disease labels
        (meaning disease labels are considered "observed"; negatives are treated as 0)
      - else: mask = 0 for all disease labels (unknown => loss masked out)

    NOTE:
      This implements your stated training rule:
        "仅对 disease 有标注的样本计算 disease loss（缺失样本 disease loss mask 掉）"
    """

    def __init__(
        self,
        ecg_ids: List[int],
        labels_path: str,
        features_path: str,
        finding_label_path: str,
        disease_label_path: Optional[str] = None,
        selected_features_path: Optional[str] = None,
        task: str = "model1",  # "model1" or "model2"
        snomed_col: str = "scp_codes_ext_snomed",
        dtype: torch.dtype = torch.float32,
    ):
        self.task = task
        if self.task not in ("model1", "model2"):
            raise ValueError("task must be one of: 'model1', 'model2'")

        self.dtype = dtype
        self.ecg_ids = [int(x) for x in ecg_ids]
        self.ecg_id_set = set(self.ecg_ids)

        # --- load label vocab ---
        self.finding_labels, self.finding_snomeds, self.finding_map = _build_label_vocab(finding_label_path)

        self.disease_labels = []
        self.disease_snomeds = []
        self.disease_map = {}
        if self.task == "model2":
            if disease_label_path is None:
                raise ValueError("disease_label_path is required for task='model2'")
            self.disease_labels, self.disease_snomeds, self.disease_map = _build_label_vocab(disease_label_path)

        # --- load features ---
        feat_df = pd.read_csv(features_path)

        if "ecg_id" not in feat_df.columns:
            raise ValueError(f"features file must contain 'ecg_id' column: {features_path}")

        feat_df["ecg_id"] = feat_df["ecg_id"].astype(int)
        feat_df = feat_df[feat_df["ecg_id"].isin(self.ecg_id_set)].copy()

        # selected features option (optional)
        if selected_features_path is not None:
            sf = pd.read_csv(selected_features_path)
            if "Feature" not in sf.columns:
                raise ValueError(f"selected_features file must have column 'Feature': {selected_features_path}")
            selected = sf["Feature"].astype(str).tolist()
            # keep ecg_id + selected feature columns that exist
            missing = [c for c in selected if c not in feat_df.columns]
            if missing:
                raise ValueError(
                    f"Some selected features not found in features file. Missing: {missing[:10]} (total {len(missing)})"
                )
            use_cols = ["ecg_id"] + selected
            feat_df = feat_df[use_cols]

        # sort by ecg_id for stable indexing
        feat_df = feat_df.sort_values("ecg_id")

        # store ecg_id -> row index
        self._feat_ecg_ids = feat_df["ecg_id"].to_numpy()
        self._feat_values = feat_df.drop(columns=["ecg_id"]).to_numpy(dtype=np.float32)

        self.num_features = self._feat_values.shape[1]

        # build lookup dict: ecg_id -> local row
        self._feat_index = {int(eid): i for i, eid in enumerate(self._feat_ecg_ids)}

        # check all requested ecg_ids exist in features
        missing_feat = [eid for eid in self.ecg_ids if eid not in self._feat_index]
        if missing_feat:
            raise ValueError(
                f"{len(missing_feat)} ecg_ids missing in features file. Example: {missing_feat[:10]}"
            )

        # --- load statements (snomed codes) ---
        stmt_df = pd.read_csv(labels_path)
        if "ecg_id" not in stmt_df.columns:
            raise ValueError(f"labels file must contain 'ecg_id' column: {labels_path}")
        if snomed_col not in stmt_df.columns:
            raise ValueError(f"labels file must contain '{snomed_col}' column: {labels_path}")

        stmt_df["ecg_id"] = stmt_df["ecg_id"].astype(int)
        stmt_df = stmt_df[stmt_df["ecg_id"].isin(self.ecg_id_set)].copy()

        # ecg_id -> set(snomed_id) present in statements
        self._snomed_by_ecg: Dict[int, set] = {}
        for _, row in stmt_df.iterrows():
            eid = int(row["ecg_id"])
            parsed = _safe_parse_list_of_tuples(row[snomed_col])
            snomeds = set()
            for item in parsed:
                # item is (snomed_id, score) OR may be malformed
                if isinstance(item, (tuple, list)) and len(item) >= 1:
                    try:
                        sid = int(item[0])
                        snomeds.add(sid)
                    except Exception:
                        continue
            self._snomed_by_ecg[eid] = snomeds

        missing_stmt = [eid for eid in self.ecg_ids if eid not in self._snomed_by_ecg]
        if missing_stmt:
            raise ValueError(
                f"{len(missing_stmt)} ecg_ids missing in statements file. Example: {missing_stmt[:10]}"
            )

        # precompute multi-hot labels for speed
        self._y_finding = np.zeros((len(self.ecg_ids), len(self.finding_labels)), dtype=np.float32)

        self._y_disease = None
        self._disease_mask = None
        if self.task == "model2":
            self._y_disease = np.zeros((len(self.ecg_ids), len(self.disease_labels)), dtype=np.float32)
            self._disease_mask = np.zeros((len(self.ecg_ids), len(self.disease_labels)), dtype=np.float32)

        for i, eid in enumerate(self.ecg_ids):
            snomeds = self._snomed_by_ecg[eid]

            # findings multi-hot
            for sid in snomeds:
                idx = self.finding_map.get(sid)
                if idx is not None:
                    self._y_finding[i, idx] = 1.0

            # disease multi-hot + mask
            if self.task == "model2":
                any_disease = False
                for sid in snomeds:
                    idx = self.disease_map.get(sid)
                    if idx is not None:
                        self._y_disease[i, idx] = 1.0
                        any_disease = True

                # If at least one disease label present => treat disease vector as "observed"
                # Mask all disease dims as 1.0. Otherwise keep 0.0 (unknown)
                if any_disease:
                    self._disease_mask[i, :] = 1.0

    def __len__(self) -> int:
        return len(self.ecg_ids)

    def __getitem__(self, idx: int):
        eid = int(self.ecg_ids[idx])
        feat_row = self._feat_index[eid]
        x = torch.tensor(self._feat_values[feat_row], dtype=self.dtype)

        y_finding = torch.tensor(self._y_finding[idx], dtype=self.dtype)

        if self.task == "model1":
            return {
                "x": x,
                "y_finding": y_finding,
                "ecg_id": eid,
            }

        # model2
        y_disease = torch.tensor(self._y_disease[idx], dtype=self.dtype)
        disease_mask = torch.tensor(self._disease_mask[idx], dtype=self.dtype)

        return {
            "x": x,
            "y_finding": y_finding,
            "y_disease": y_disease,
            "disease_mask": disease_mask,
            "ecg_id": eid,
        }


def make_dataloader(
    split_ids_path: str,
    labels_path: str,
    features_path: str,
    finding_label_path: str,
    disease_label_path: Optional[str] = None,
    selected_features_path: Optional[str] = None,
    task: str = "model1",
    batch_size: int = 256,
    shuffle: bool = True,
    num_workers: int = 2,
    pin_memory: bool = True,
    drop_last: bool = False,
) -> DataLoader:
    """
    split_ids_path: CSV that contains ecg_id column (e.g. train_ecg_ids.csv)
    """
    ecg_ids = _load_split_ids(split_ids_path)

    ds = PTBXLFeatureDataset(
        ecg_ids=ecg_ids,
        labels_path=labels_path,
        features_path=features_path,
        finding_label_path=finding_label_path,
        disease_label_path=disease_label_path,
        selected_features_path=selected_features_path,
        task=task,
    )

    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=drop_last,
    )

