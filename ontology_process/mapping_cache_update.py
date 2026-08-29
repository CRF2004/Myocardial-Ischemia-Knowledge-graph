import json
import os
from pathlib import Path
from typing import Dict, Any

# ================== 配置区 ==================
PROCESSED_BASE_DIR = "SRP心肌缺血知识图谱/output/processed"
CACHE_FILE = "SRP心肌缺血知识图谱/output/entity_mapping_cache.json"
BACKUP_CACHE = "SRP心肌缺血知识图谱/output/entity_mapping_cache.backup.json"
# ===========================================


def load_json(file_path: str) -> Any:
    """加载 JSON 文件"""
    if not os.path.exists(file_path):
        return {}
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    """保存 JSON 文件"""
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def normalize_key(text: str) -> str:
    """标准化缓存键（与原系统保持一致）"""
    return text.lower().strip()


def update_cache_from_mapped(cache: Dict, mapped_item: Dict) -> int:
    """
    从 auto_mapped 条目更新缓存
    返回更新的条目数
    """
    updated_count = 0
    
    # 获取原始文本
    original_text = mapped_item.get("original_entity_text", "")
    if not original_text:
        return 0
    
    cache_key = normalize_key(original_text)
    llm_output = mapped_item.get("llm_output", {})
    
    # 处理单个映射 (map_to_exist)
    if llm_output.get("action") == "map_to_exist" and llm_output.get("target_id"):
        target_id = llm_output["target_id"]
        cache[cache_key] = {
            "ontology_id": target_id,
            "normalized_name": original_text.replace(" ", "_"),
            "match_method": "exact_match",  # 映射到现有实体
            "original_text": original_text
        }
        updated_count += 1
        print(f"  ✓ Mapped: '{original_text}' → {target_id}")
    
    # 处理多个映射 (map_to_multiple)
    elif llm_output.get("action") == "map_to_multiple" and llm_output.get("mapping_ids"):
        mapping_ids = llm_output["mapping_ids"]
        # 对于复合实体，映射到第一个主要ID（或者可以选择不同策略）
        primary_id = mapping_ids[0] if mapping_ids else None
        if primary_id:
            cache[cache_key] = {
                "ontology_id": primary_id,
                "normalized_name": original_text.replace(" ", "_"),
                "match_method": "composite_primary",  # 复合实体的主要映射
                "original_text": original_text,
                "all_mappings": mapping_ids  # 保留所有映射信息
            }
            updated_count += 1
            print(f"  ✓ Composite: '{original_text}' → {mapping_ids}")
    
    return updated_count


def process_all_auto_mapped():
    """遍历所有 auto_mapped.json 并更新缓存"""
    
    # 1. 备份原缓存
    cache = load_json(CACHE_FILE)
    if cache:
        save_json(cache, BACKUP_CACHE)
        print(f"✓ Backup created: {BACKUP_CACHE}")
    
    # 2. 统计信息
    total_updated = 0
    processed_files = 0
    
    # 3. 遍历所有 processed 子目录
    if not os.path.exists(PROCESSED_BASE_DIR):
        print(f"❌ Directory not found: {PROCESSED_BASE_DIR}")
        return
    
    for chapter_dir in os.listdir(PROCESSED_BASE_DIR):
        chapter_path = os.path.join(PROCESSED_BASE_DIR, chapter_dir)
        if not os.path.isdir(chapter_path):
            continue
        
        auto_mapped_file = os.path.join(chapter_path, "auto_mapped.json")
        if not os.path.exists(auto_mapped_file):
            continue
        
        print(f"\n📂 Processing: {chapter_dir}")
        mapped_items = load_json(auto_mapped_file)
        
        if not mapped_items:
            print("  ⚠ Empty auto_mapped.json")
            continue
        
        # 4. 更新缓存
        chapter_updates = 0
        for item in mapped_items:
            chapter_updates += update_cache_from_mapped(cache, item)
        
        print(f"  ✓ Updated {chapter_updates} entries from this chapter")
        total_updated += chapter_updates
        processed_files += 1
    
    # 5. 保存更新后的缓存
    if total_updated > 0:
        save_json(cache, CACHE_FILE)
        print(f"\n{'='*60}")
        print(f"✅ Cache update completed!")
        print(f"   Files processed: {processed_files}")
        print(f"   Total entries updated: {total_updated}")
        print(f"   Cache size: {len(cache)} entries")
        print(f"   Backup: {BACKUP_CACHE}")
    else:
        print(f"\n⚠ No updates made (0 mapped entries found)")


def verify_cache_integrity():
    """验证缓存完整性（可选）"""
    cache = load_json(CACHE_FILE)
    
    issues = []
    for key, value in cache.items():
        if not isinstance(value, dict):
            issues.append(f"Invalid value type for key: {key}")
        elif "ontology_id" not in value:
            issues.append(f"Missing ontology_id for key: {key}")
    
    if issues:
        print("\n⚠ Cache integrity issues found:")
        for issue in issues:
            print(f"  - {issue}")
    else:
        print("\n✓ Cache integrity verified: OK")


# ================== 主入口 ==================
if __name__ == "__main__":
    print("Starting entity_mapping_cache update...\n")
    
    process_all_auto_mapped()
    
    # 可选：验证缓存完整性
    verify_cache_integrity()
    
    print("\n" + "="*60)
    print("Note: If you need to rollback, restore from:")
    print(f"  {BACKUP_CACHE}")