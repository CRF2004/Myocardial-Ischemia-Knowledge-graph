import json
import os
import re
from typing import List, Dict, Any
from pathlib import Path

# 你的 LLM 调用接口
import LLM_CALL

# ================== 配置区 ==================
ONTOLOGY_DIR = "SRP心肌缺血知识图谱/ontology"
OUTPUT_CANDIDATES_DIR = "SRP心肌缺血知识图谱/output"
PROCESSED_BASE_DIR = "SRP心肌缺血知识图谱/output/processed"
DOMAIN_WHITELIST = [
    "diagnostic_test", "disease", "patient_characteristics", "symptoms", "treatment_intervention"
]
MODEL_NAME = "gpt-4o" 
CONFIDENCE_THRESHOLD = 0.8
# ================== 类型映射表 ==================
ONTOLOGY_TYPE_TO_FILE = {
    "disease": "disease",
    "diseases": "disease",           # LLM 可能输出复数
    "symptom": "symptoms",
    "symptoms": "symptoms",
    "diagnostic_test": "diagnostic_test",
    "diagnostic_tests": "diagnostic_test",
    "examinations": "diagnostic_test",
    "patient_characteristic": "patient_characteristics",
    "patient_characteristics": "patient_characteristics",
    "treatment": "treatment_intervention",
    "treatments": "treatment_intervention",
    "intervention": "treatment_intervention",
    "treatment_intervention": "treatment_intervention",
    # 可继续补充
}
# ===========================================


def load_json(file_path: str) -> Any:
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def save_json(data: Any, file_path: str):
    Path(file_path).parent.mkdir(parents=True, exist_ok=True)
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def parse_llm_response(response: str) -> Dict:
    """解析 LLM 返回的 JSON，支持容错处理"""
    try:
        # 提取 JSON 块（如果被 markdown 包裹）
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', response, re.DOTALL)
        if json_match:
            response = json_match.group(1)
        
        # 移除 JSON 注释（// 和 /* */ 格式）
        response = re.sub(r'//.*?$', '', response, flags=re.MULTILINE)  # 移除单行注释
        response = re.sub(r'/\*.*?\*/', '', response, flags=re.DOTALL)  # 移除多行注释
        
        # 解析 JSON
        return json.loads(response)
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"问题JSON: {response}")
        
        # 尝试更激进的清理
        try:
            # 移除所有注释和多余空白
            cleaned = re.sub(r'//.*', '', response)
            cleaned = re.sub(r'/\*.*?\*/', '', cleaned, flags=re.DOTALL)
            cleaned = re.sub(r',\s*([}\]])', r'\1', cleaned)  # 移除尾随逗号
            return json.loads(cleaned)
        except:
            print("简单修复也失败，返回默认拒绝决策")
            return {
                "action": "reject",
                "target_id": None,
                "mapping_ids": None,
                "suggested_parent_id": None,
                "split_rationale": None,
                "reason": "LLM response parsing failed",
                "confidence": 0.0
            }
            
def load_all_ontologies(onto_dir: str) -> Dict[str, List[Dict]]:
    ontologies = {}
    for file in os.listdir(onto_dir):
        if file.endswith(".json"):
            onto_type = file.replace(".json", "")
            if onto_type in DOMAIN_WHITELIST:
                path = os.path.join(onto_dir, file)
                raw_list = load_json(path)
                normalized = []
                for item in raw_list:
                    norm = item.copy()
                    # 添加兼容字段
                    norm['id'] = item['ontology_id']
                    norm['label'] = item.get('name_en') or item.get('name_cn') or 'Unknown'
                    normalized.append(norm)
                ontologies[onto_type] = normalized
    return ontologies

def save_all_ontologies(onto_dir: str, ontologies: Dict[str, List[Dict]]):
    for onto_type, entities in ontologies.items():
        path = os.path.join(onto_dir, f"{onto_type}.json")
        save_json(entities, path)

def format_flat_entities(flat_list: List[Dict]) -> str:
    lines = []
    for e in flat_list:
        if 'ontology_id' not in e:
            continue
        label = e.get('name_en') or e.get('name_cn') or 'Unknown'
        lines.append(f"- {e['ontology_id']}: {label}")
    return "\n".join(lines)

def generate_child_id(parent_id: str, flat_entities: List[Dict]) -> str:
    if not parent_id:
        raise ValueError("parent_id cannot be None")
    
    children = [
        e['id'] for e in flat_entities
        if e.get('id') and e['id'].startswith(parent_id + '.') 
        and e['id'].count('.') == parent_id.count('.') + 1
    ]
    numbers = [int(c.split('.')[-1]) for c in children if c.split('.')[-1].isdigit()]
    next_num = (max(numbers) if numbers else 0) + 1
    return f"{parent_id}.{next_num:03d}"

def build_prompt(entity: Dict, original_text: str, flat_entities: List[Dict]) -> str:
    # 直接使用 entity，不再从 candidate 取
    onto_type = entity.get("ontology_type", "unknown")
    name_en = entity.get("name_en", "Unknown")
    definition = entity.get("definition", "No definition")
    prompt = f"""You are an ontology curator for Myocardial Ischemia Diagnostic Knowledge Graph.
    Domain whitelist: {", ".join(DOMAIN_WHITELIST)}

    Existing entities (flat list):
    {format_flat_entities(flat_entities)}

    # 只增加这一行关键提示
    Parent Selection Tip: Choose parent_id carefully based on category (e.g., demographics under PC001, diseases under D001, etc.)

    Candidate:
    - original_text: "{original_text}"
    - ontology_type: {onto_type}
    - name_en: {name_en}
    - definition: {definition}

    Rules:
    1. If concept is EXACTLY covered by one existing entity → map_to_exist
    2. If it's a COMPOSITE (e.g., "CV death or MI") → map_to_multiple, list existing or suggest split
    3. If it's a NEW valid concept in domain → add_new_entity, suggest parent_id
    4. If irrelevant to myocardial ischemia diagnosis → reject

    NEVER invent new IDs. Use existing or suggest parent.

    Output JSON only:
    {{
    "action": "map_to_exist" | "map_to_multiple" | "add_new_entity" | "reject",
    "target_id": "D001.001" | null,
    "mapping_ids": ["D001.002", "D001.003"] | null,
    "suggested_parent_id": "D001" | "SS001" | null,
    "split_rationale": "..." | null,
    "reason": "Brief justification",
    "confidence": 0.XX
    }}
    """
    return prompt


def call_ontology_agent(entity: Dict, original_text: str, flat_entities: List[Dict]) -> Dict:
    prompt = build_prompt(entity, original_text, flat_entities)
    response = LLM_CALL.chat(MODEL_NAME, prompt)
    return parse_llm_response(response)

def process_chapter_file(input_file: str):
    print(f"Processing: {input_file}")

    # 检查输入文件是否存在
    if not os.path.exists(input_file):
        print(f"输入文件不存在，跳过: {input_file}")
        return

    # 检查输出目录是否已存在
    chapter_name = Path(input_file).stem
    out_dir = os.path.join(PROCESSED_BASE_DIR, chapter_name)
    
    if os.path.exists(out_dir):
        # 检查是否包含有效的审计日志文件
        audit_log_path = os.path.join(out_dir, "audit_log.json")
        if os.path.exists(audit_log_path):
            try:
                audit_log = load_json(audit_log_path)
                if isinstance(audit_log, list) and len(audit_log) > 0:
                    print(f"文件已处理，跳过: {input_file} (已存在 {len(audit_log)} 条记录)")
                    return
            except Exception as e:
                print(f"审计日志文件损坏，将重新处理: {audit_log_path} (错误: {e})")

    # 加载本体
    ontologies = load_all_ontologies(ONTOLOGY_DIR)
    flat_all = [e for lst in ontologies.values() for e in lst]

    # 加载候选
    candidates = load_json(input_file)
    add_entities = [c for c in candidates if c.get("operation") == "ADD_ENTITY"]

    # 结果容器
    results = {
        "auto_adopted": [],
        "auto_mapped": [],
        "auto_rejected": [],
        "manual_review": [],
        "audit_log": []
    }

    for cand in add_entities:
        entity = cand["suggested_entity"]
        text = cand["original_entity_text"]
        raw_type = entity.get("ontology_type", "").lower().strip()
        onto_type = ONTOLOGY_TYPE_TO_FILE.get(raw_type)
        
        if not onto_type or onto_type not in ontologies:
            # 降级处理：尝试模糊匹配或默认
            for key in ONTOLOGY_TYPE_TO_FILE:
                if key in raw_type or raw_type in key:
                    onto_type = ONTOLOGY_TYPE_TO_FILE[key]
                    break
            else:
                # 最终兜底：提前定义 log_entry
                log_entry = {
                    **cand,
                    "llm_output": None,
                    "error": f"Unknown ontology_type: {raw_type}"
                }
                results["manual_review"].append(log_entry)
                results["audit_log"].append(log_entry)  # 也记录到审计日志
                continue  # 跳过这轮

        # 正常流程继续
        llm_output = call_ontology_agent(entity, text, flat_all)
        
        action = llm_output.get("action", "reject")
        conf = llm_output.get("confidence", 0.0)
        log_entry = {**cand, "llm_output": llm_output}
        results["audit_log"].append(log_entry)

        # 高置信自动处理
        if conf > CONFIDENCE_THRESHOLD:
            if action == "map_to_exist" and llm_output.get("target_id"):
                results["auto_mapped"].append({**log_entry, "mapped_to": llm_output["target_id"]})

            elif action == "map_to_multiple" and llm_output.get("mapping_ids"):
                results["auto_mapped"].append({**log_entry, "mapped_to": llm_output["mapping_ids"]})

            elif action == "add_new_entity" and llm_output.get("suggested_parent_id"):
                parent_id = llm_output["suggested_parent_id"]
                new_id = generate_child_id(parent_id, flat_all)
                new_entity = {
                    "ontology_id": new_id,        # ← 保留原始字段
                    "id": new_id,                 # ← 兼容 flat_all
                    "name_en": entity["name_en"],
                    "name_cn": entity.get("name_cn", ""),
                    "definition": entity.get("definition", ""),
                    "type": onto_type,
                    "level": parent_id.count('.') + 1,
                    "parent_id": parent_id,
                    "alias": ""
                }
                ontologies[onto_type].append(new_entity)
                flat_all.append(new_entity)
                results["auto_adopted"].append({**log_entry, "new_id": new_id, "new_entity": new_entity})

            elif action == "reject":
                results["auto_rejected"].append(log_entry)

            else:
                results["manual_review"].append(log_entry)
        else:
            results["manual_review"].append(log_entry)

    # 保存更新后的 ontology
    save_all_ontologies(ONTOLOGY_DIR, ontologies)

    # 创建输出目录
    chapter_name = Path(input_file).stem
    out_dir = os.path.join(PROCESSED_BASE_DIR, chapter_name)
    os.makedirs(out_dir, exist_ok=True)

    # 保存结果
    save_json(results["auto_adopted"], f"{out_dir}/auto_adopted.json")
    save_json(results["auto_mapped"], f"{out_dir}/auto_mapped.json")
    save_json(results["auto_rejected"], f"{out_dir}/auto_rejected.json")
    save_json(results["manual_review"], f"{out_dir}/manual_review.json")
    save_json(results["audit_log"], f"{out_dir}/audit_log.json")

    print(f"Completed: {len(results['auto_adopted'])} adopted, "
          f"{len(results['auto_mapped'])} mapped, "
          f"{len(results['auto_rejected'])} rejected, "
          f"{len(results['manual_review'])} manual")


# ================== 主入口 ==================
if __name__ == "__main__":
    # 自动处理 output 目录下所有 ontology_update_candidates_*.json
    for file in os.listdir(OUTPUT_CANDIDATES_DIR):
        if file.startswith("ontology_update_candidates_chapter_") and file.endswith(".json"):
            input_path = os.path.join(OUTPUT_CANDIDATES_DIR, file)
            process_chapter_file(input_path)