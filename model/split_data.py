"""
唯一划分单位：ecg_id
划分比例：Train / Val / Test = 0.7 / 0.1 / 0.2
ecg_id 来源：以 ptbxl_statements.csv 为准（权威标签源）
可复现性：固定 random_state
输出：三个 csv + 一个 json（便于程序与人工检查）
"""

import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

# =========================
# Paths
# =========================
LABELS_PATH = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/ptbxl_statements.csv"
OUTPUT_DIR = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/model/splits"

TRAIN_RATIO = 0.7
VAL_RATIO = 0.1
TEST_RATIO = 0.2

RANDOM_STATE = 42

os.makedirs(OUTPUT_DIR, exist_ok=True)

# =========================
# Load ecg_id list
# =========================
labels_df = pd.read_csv(LABELS_PATH)

if "ecg_id" not in labels_df.columns:
    raise ValueError("ptbxl_statements.csv must contain 'ecg_id' column")

ecg_ids = labels_df["ecg_id"].dropna().unique()
ecg_ids = np.array(sorted(ecg_ids))

print(f"Total ECG records: {len(ecg_ids)}")

# =========================
# First split: Train vs Temp (Val + Test)
# =========================
train_ids, temp_ids = train_test_split(
    ecg_ids,
    test_size=(1.0 - TRAIN_RATIO),
    random_state=RANDOM_STATE,
    shuffle=True,
)

# =========================
# Second split: Val vs Test
# =========================
val_ratio_adjusted = VAL_RATIO / (VAL_RATIO + TEST_RATIO)

val_ids, test_ids = train_test_split(
    temp_ids,
    test_size=(1.0 - val_ratio_adjusted),
    random_state=RANDOM_STATE,
    shuffle=True,
)

# =========================
# Sanity checks
# =========================
assert len(set(train_ids) & set(val_ids)) == 0
assert len(set(train_ids) & set(test_ids)) == 0
assert len(set(val_ids) & set(test_ids)) == 0
assert len(train_ids) + len(val_ids) + len(test_ids) == len(ecg_ids)

print(f"Train: {len(train_ids)}")
print(f"Val  : {len(val_ids)}")
print(f"Test : {len(test_ids)}")

# =========================
# Save splits
# =========================
pd.DataFrame({"ecg_id": train_ids}).to_csv(
    os.path.join(OUTPUT_DIR, "train_ecg_ids.csv"), index=False
)
pd.DataFrame({"ecg_id": val_ids}).to_csv(
    os.path.join(OUTPUT_DIR, "val_ecg_ids.csv"), index=False
)
pd.DataFrame({"ecg_id": test_ids}).to_csv(
    os.path.join(OUTPUT_DIR, "test_ecg_ids.csv"), index=False
)

# JSON version (convenient for config loading)
split_json = {
    "train": train_ids.tolist(),
    "val": val_ids.tolist(),
    "test": test_ids.tolist(),
}

with open(os.path.join(OUTPUT_DIR, "ecg_splits.json"), "w") as f:
    json.dump(split_json, f, indent=2)

print(f"Splits saved to: {OUTPUT_DIR}")

