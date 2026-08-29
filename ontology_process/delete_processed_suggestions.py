# delete_processed_suggestions.py
import json
import os
from pathlib import Path

# ================== 配置 ==================
PROJECT_ROOT = "SRP心肌缺血知识图谱"
REVIEW_STATUS_FILE = "opt/Project/data/review_status.json"
CANDIDATES_DIR = os.path.join(PROJECT_ROOT, "output")
BACKUP_DIR = os.path.join(PROJECT_ROOT, "output/backup_preprocessed")
DOMAIN_WHITELIST = [
    "diagnostic_test", "disease", "patient_characteristics", "symptoms", "treatment_intervention"
]
# ===========================================


def load_json(path):
    with open(path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data, path):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_reviewed_map(review_status_path: str):
    """
    返回: {
        "ontology_update_candidates_chapter_1": {
            "ontology_update_candidates_chapter_1_0": "rejected"
        }
    }
    """
    if not os.path.exists(review_status_path):
        print(f"Warning: {review_status_path} not found, skip filtering")
        return {}

    data = load_json(review_status_path)
    reviewed_map = {}

    for chapter_key, items in data.items():
        if not isinstance(items, list):
            continue
        # 清理 chapter_key: "ontology_update_candidates_chapter_1" 
        clean_key = chapter_key.replace(".json", "")
        reviewed_map[clean_key] = {}
        for item in items:
            sugg_id = item.get("suggestion_id")
            if sugg_id:
                reviewed_map[clean_key][sugg_id] = item.get("status", "unknown")

    return reviewed_map, data  # ← 修复：返回 data 供后续使用


def preprocess_candidates():
    # 1. 加载人工审核状态
    reviewed_map, review_data = build_reviewed_map(REVIEW_STATUS_FILE)  # ← 修复：接收 data
    if not reviewed_map:
        print("No review status found, skip preprocessing")
        return

    total_processed = 0
    total_kept = 0

    # 2. 遍历所有 candidate 文件
    for file_name in os.listdir(CANDIDATES_DIR):
        if not file_name.startswith("ontology_update_candidates_chapter_") or not file_name.endswith(".json"):
            continue

        file_path = os.path.join(CANDIDATES_DIR, file_name)
        chapter_key = file_name.replace(".json", "")

        # 备份原文件
        backup_path = os.path.join(BACKUP_DIR, file_name)
        save_json(load_json(file_path), backup_path)
        print(f"Backed up: {file_name}")

        # 加载数据
        candidates = load_json(file_path)
        filtered = []

        # 过滤
        reviewed_ids = reviewed_map.get(chapter_key, {})
        processed_count = 0
        
        for idx, cand in enumerate(candidates):
            # 生成 suggestion_id: ontology_update_candidates_chapter_1_0
            sugg_id = f"{chapter_key}_{idx}"
            
            if sugg_id in reviewed_ids:
                status = reviewed_ids[sugg_id]  # ← 修复：直接从 map 取，不用 next()
                print(f"  Skip [{status.upper()}]: {sugg_id} -> {cand.get('original_entity_text', '')[:50]}...")
                processed_count += 1
                total_processed += 1
                continue
            
            filtered.append(cand)
            total_kept += 1

        # 保存过滤后的文件（覆盖原文件）
        save_json(filtered, file_path)
        print(f"  Kept {len(filtered)}/{len(candidates)} items in {file_name} "
              f"(processed: {processed_count})\n")

    print(f"\n=== PREPROCESSING SUMMARY ===")
    print(f"Total processed (removed): {total_processed}")
    print(f"Total kept (for LLM): {total_kept}")
    print("Preprocessing completed! Only unprocessed ADD_ENTITY items remain.\n")


if __name__ == "__main__":
    os.makedirs(BACKUP_DIR, exist_ok=True)
    preprocess_candidates()