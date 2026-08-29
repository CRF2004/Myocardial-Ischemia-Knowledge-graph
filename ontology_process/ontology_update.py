import json
import logging
import os
from rapidfuzz import fuzz, process

import LLM_CALL

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class EntityMappingCache:
    """实体映射缓存类"""
    
    def __init__(self, cache_file_path="entity_mapping_cache.json"):
        self.cache_file_path = cache_file_path
        self.mapping_cache = self.load_cache()
    
    def load_cache(self):
        """加载映射缓存"""
        if os.path.exists(self.cache_file_path):
            try:
                with open(self.cache_file_path, 'r', encoding='utf-8') as f:
                    cache_data = json.load(f)
                    logger.info(f"加载映射缓存成功，从 {self.cache_file_path} 加载了 {len(cache_data)} 条记录")
                    return cache_data
            except Exception as e:
                logger.warning(f"加载映射缓存失败: {e}")
        
        logger.info(f"创建新的映射缓存，将保存至: {self.cache_file_path}")
        return {}
    
    def save_cache(self):
        """保存映射缓存"""
        try:
            # 确保目录存在
            cache_dir = os.path.dirname(self.cache_file_path)
            if cache_dir:  # 如果有目录路径
                os.makedirs(cache_dir, exist_ok=True)
            
            with open(self.cache_file_path, 'w', encoding='utf-8') as f:
                json.dump(self.mapping_cache, f, ensure_ascii=False, indent=2)
            logger.info(f"映射缓存已保存至: {self.cache_file_path}，包含 {len(self.mapping_cache)} 条记录")
        except Exception as e:
            logger.error(f"保存映射缓存失败: {e}")
            logger.error(f"缓存文件路径: {self.cache_file_path}")
            logger.error(f"目录路径: {os.path.dirname(self.cache_file_path)}")
    
    def get_mapping(self, entity_text):
        """获取实体映射"""
        return self.mapping_cache.get(entity_text.lower())
    
    def add_mapping(self, entity_text, ontology_id, normalized_name, match_method):
        """添加实体映射"""
        self.mapping_cache[entity_text.lower()] = {
            "ontology_id": ontology_id,
            "normalized_name": normalized_name,
            "match_method": match_method,
            "original_text": entity_text  # 保留原始大小写
        }
        logger.debug(f"添加映射: {entity_text} -> {ontology_id} ({normalized_name})")
    
    def get_cache_stats(self):
        """获取缓存统计信息"""
        methods = {}
        for mapping in self.mapping_cache.values():
            method = mapping.get('match_method', 'unknown')
            methods[method] = methods.get(method, 0) + 1
        return methods

def load_ontologies(ontology_paths):
    """加载所有本体JSON文件"""
    ontologies = {}
    all_entities = []  # 存储所有实体用于匹配
    name_to_entity = {}  # 创建名称到实体的快速查找字典
    
    for ont_type, path in ontology_paths.items():
        with open(path, 'r', encoding='utf-8') as f:
            ontology_data = json.load(f)
            ontologies[ont_type] = ontology_data
            
            # 为每个实体添加来源标识
            for entity in ontology_data:
                entity['_source_ontology'] = ont_type
                all_entities.append(entity)
                
                # 建立名称映射（英文名和中文名都加入）
                if entity.get('name_en'):
                    name_to_entity[entity['name_en'].lower()] = entity
                if entity.get('name_cn'):
                    name_to_entity[entity['name_cn'].lower()] = entity
    
    return ontologies, all_entities, name_to_entity

def exact_name_match(entity_text, name_to_entity):
    """精确名称匹配"""
    return name_to_entity.get(entity_text.lower())

def rapidfuzz_match(entity_text, all_entities, threshold=80):
    """使用RapidFuzz进行快速匹配"""
    # 提取所有实体的英文名用于匹配
    choices = []
    entity_map = {}
    
    for i, entity in enumerate(all_entities):
        name_en = entity.get('name_en', '')
        if name_en:
            choices.append(name_en)
            entity_map[name_en] = entity
    
    # 使用RapidFuzz进行匹配
    result = process.extractOne(entity_text, choices, scorer=fuzz.ratio)
    
    if result and result[1] >= threshold:
        matched_entity = entity_map[result[0]]
        return matched_entity, result[1]
    
    return None, 0

def get_top_candidates(entity_text, all_entities, top_k=6):
    """获取top-k候选实体"""
    choices = []
    entity_map = {}
    
    for entity in all_entities:
        name_en = entity.get('name_en', '')
        if name_en:
            choices.append(name_en)
            entity_map[name_en] = entity
    
    results = process.extract(entity_text, choices, scorer=fuzz.ratio, limit=top_k)
    candidates = [entity_map[result[0]] for result in results]
    
    return candidates

def llm_semantic_match(entity_text, source_sentence, candidates):
    """使用LLM进行语义匹配"""
    
    # 构建候选实体信息，包含本体来源
    candidates_info = []
    for i, candidate in enumerate(candidates):
        ontology_type = candidate.get('_source_ontology', 'unknown')
        definition = candidate.get('definition', 'null')
        
        candidates_info.append({
            "index": i + 1,
            "ontology_id": candidate['ontology_id'],
            "ontology_type": ontology_type,  # 添加本体类型信息
            "name_en": candidate['name_en'],
            "name_cn": candidate['name_cn'],
            "definition": definition
        })
    
    prompt = f"""你是医学本体实体匹配专家。请分析以下信息并做出判断。

    【待匹配实体】
    原始文本: "{entity_text}"
    上下文句子: "{source_sentence}"

    【候选实体列表】
    """
        
    for cand in candidates_info:
        prompt += f"""
        {cand['index']}. [{cand['ontology_type']}] {cand['name_cn']} ({cand['name_en']})
        - ID: {cand['ontology_id']}
        - 定义: {cand['definition']}
        """
        
    prompt += """
    【任务说明】
    判断原始实体是否与某个候选实体匹配，或者需要创建新实体。

    【输出要求】
    1. 必须输出纯JSON格式，不含markdown标记、注释或其他文字
    2. 所有字符串值用双引号，内部引号用单引号
    3. 空值使用null（不带引号）
    4. 英文名称应尽量符合umls命名习惯
    5. 只有"name_cn"的值输出为中文，其余文本全部保持英文输出

    【JSON格式】
    {
    "match_type": "EXISTING 或 NEW_ENTITY",
    "matched_ontology_id": "匹配的实体ID或null",
    "normalized_name": "规范化名称（英文名）",
    "new_entity_suggestion": {
        "ontology_type": "本体类型(diseases/symptoms/treatments/examinations/patient_characteristics)或null",
        "suggested_parent_id": "建议的父节点ID或null",
        "name_en": "英文名或null",
        "name_cn": "中文名或null",
        "definition": "定义或null",
        "type": "实体类型或null"
    } 或 null,
    "rationale": "判断理由(简洁说明)"
    }

    【注意事项】
    - 如果match_type为EXISTING，则new_entity_suggestion必须为null
    - 如果match_type为NEW_ENTITY，则必须指定ontology_type以确定添加到哪个本体文件
    - ontology_type必须从以下选项中选择：diseases, symptoms, treatments, examinations, patient_characteristics
    """
    
    response = LLM_CALL.chat("DMXAPI-HuoShan-DeepSeek-V3", prompt)
    return LLM_CALL.parse_llm_response(response)

def normalize_entity(entity_text, source_sentence, all_entities, name_to_entity, mapping_cache):
    """规范化单个实体"""
    
    # 1. 首先检查映射缓存
    cached_mapping = mapping_cache.get_mapping(entity_text)
    if cached_mapping:
        logger.info(f"缓存命中: {entity_text} -> {cached_mapping['normalized_name']} (方法: {cached_mapping['match_method']})")
        return {
            "text": entity_text,
            "ontology_id": cached_mapping['ontology_id'],
            "normalized_name": cached_mapping['normalized_name'],
            "match_method": "cache_hit"
        }, None
    
    # 2. 精确名称匹配
    matched_entity = exact_name_match(entity_text, name_to_entity)
    if matched_entity:
        logger.info(f"精确匹配成功: {entity_text} -> {matched_entity['name_en']}")
        
        mapping_cache.add_mapping(entity_text, matched_entity['ontology_id'], 
                                 matched_entity['name_en'], "exact_match")
        
        return {
            "text": entity_text,
            "ontology_id": matched_entity['ontology_id'],
            "normalized_name": matched_entity['name_en'],
            "match_method": "exact_match"
        }, None
    
    # 3. RapidFuzz模糊匹配
    matched_entity, score = rapidfuzz_match(entity_text, all_entities)
    if matched_entity:
        logger.info(f"RapidFuzz匹配成功: {entity_text} -> {matched_entity['name_en']} (分数: {score})")
        
        mapping_cache.add_mapping(entity_text, matched_entity['ontology_id'], 
                                 matched_entity['name_en'], "rapidfuzz")
        
        return {
            "text": entity_text,
            "ontology_id": matched_entity['ontology_id'],
            "normalized_name": matched_entity['name_en'],
            "match_method": "rapidfuzz"
        }, None
    
    # 4. LLM语义匹配（最后的fallback）
    logger.info(f"所有自动匹配失败，使用LLM: {entity_text}")
    candidates = get_top_candidates(entity_text, all_entities, top_k=6)
    llm_result = llm_semantic_match(entity_text, source_sentence, candidates)
    
    if llm_result['match_type'] == 'EXISTING':
        # 找到匹配的现有实体
        matched_id = llm_result['matched_ontology_id']
        matched_entity = next((e for e in all_entities if e['ontology_id'] == matched_id), None)
        
        mapping_cache.add_mapping(entity_text, matched_id, 
                                 llm_result['normalized_name'], "llm_match")
        
        return {
            "text": entity_text,
            "ontology_id": matched_id,
            "normalized_name": llm_result['normalized_name'],
            "match_method": "llm_match"
        }, None
    
    else:
        # 需要新增实体
        new_entity_suggestion = llm_result['new_entity_suggestion']
        
        # 验证ontology_type是否存在
        if not new_entity_suggestion.get('ontology_type'):
            logger.warning(f"LLM未返回ontology_type，将使用'unknown': {entity_text}")
            new_entity_suggestion['ontology_type'] = 'unknown'
        
        new_entity_candidate = {
            "operation": "ADD_ENTITY",
            "source_sentence": source_sentence,
            "original_entity_text": entity_text,  # 添加原始实体文本
            "ontology_type": new_entity_suggestion['ontology_type'],  # 明确指定本体类型
            "suggested_parent_id": new_entity_suggestion['suggested_parent_id'],
            "suggested_entity": new_entity_suggestion,
            "rationale": llm_result['rationale']
        }
        
        # 为新实体生成临时ID（包含类型信息）
        temp_id = f"TEMP_{new_entity_suggestion['ontology_type']}_{hash(entity_text) % 10000:04d}"
        
        mapping_cache.add_mapping(entity_text, temp_id, 
                                 new_entity_suggestion['name_en'], "new_entity")
        
        return {
            "text": entity_text,
            "ontology_id": temp_id,
            "normalized_name": new_entity_suggestion['name_en'],
            "match_method": "new_entity"
        }, new_entity_candidate

def process_triple(triple, all_entities, name_to_entity, mapping_cache):
    """处理单个三元组"""
    head_result, head_candidate = normalize_entity(
        triple['head'], triple['source_sentences'], all_entities, name_to_entity, mapping_cache)
    tail_result, tail_candidate = normalize_entity(
        triple['tail'], triple['source_sentences'], all_entities, name_to_entity, mapping_cache)
    
    normalized_triple = {
        "head": head_result,
        "relation": triple['relation'],
        "tail": tail_result,
        "source_sentences": triple['source_sentences']
    }
    
    candidates = []
    if head_candidate:
        candidates.append(head_candidate)
    if tail_candidate:
        candidates.append(tail_candidate)
    
    return normalized_triple, candidates

def replace_sentence_ids_with_content(json_data):
    # 创建一个句子编号到内容的映射字典
    sentence_map = {item[0]: item[1] for item in json_data['sentences']}
    print(json.dumps(json_data, ensure_ascii=False, indent=2))
    # 遍历 extraction_result 中的 triplets
    for triplet in json_data['extraction_result']['triplets']:
        # 获取 source_sentences 中的编号列表
        source_ids = triplet['source_sentences']
        # 将编号替换为对应的句子内容
        triplet['source_sentences'] = [sentence_map.get(id, f"Sentence {id} not found") for id in source_ids]
    
    return json_data

def run_pipeline(input_triples_path, ontology_paths, output_dir, cache_file_path="entity_mapping_cache.json"):
    """运行整个pipeline"""
    # 检查输入文件是否存在
    if not os.path.exists(input_triples_path):
        logger.warning(f"输入文件不存在，跳过: {input_triples_path}")
        return 0
    
    # 读取输入文件以获取title
    with open(input_triples_path, 'r', encoding='utf-8') as f:
        input_json = json.load(f)
    
    # 检查输出文件是否已存在
    normalized_output_path = os.path.join(output_dir, f'normalized_triples_{input_json["title"]}.json')
    
    if os.path.exists(normalized_output_path):
        try:
            # 验证输出文件是否有效
            with open(normalized_output_path, 'r', encoding='utf-8') as f:
                normalized_data = json.load(f)
                if isinstance(normalized_data, list) and len(normalized_data) > 0:
                    logger.info(f"文件已处理，跳过: {input_triples_path} (已存在 {len(normalized_data)} 条记录)")
                    return len(normalized_data)
        except Exception as e:
            logger.warning(f"输出文件损坏，将重新处理: {normalized_output_path} (错误: {e})")
    
    logger.info("开始运行医学本体实体规范化Pipeline")
    
    # 初始化映射缓存
    mapping_cache = EntityMappingCache(cache_file_path)
    
    # 1. 加载数据
    logger.info("加载本体数据...")
    ontologies, all_entities, name_to_entity = load_ontologies(ontology_paths)
    logger.info(f"加载了 {len(all_entities)} 个本体实体，创建了 {len(name_to_entity)} 个名称映射")
    
    if input_json["extraction_result"] is None:
        print(f"{input_json['title']} 无有效三元组")
        return 0
    
    input_triples = replace_sentence_ids_with_content(input_json)["extraction_result"]["triplets"]
    
    # 2. 处理所有三元组
    normalized_triples = []
    new_entity_candidates = []
    
    logger.info(f"开始处理 {len(input_triples)} 个三元组...")
    
    # 统计匹配方法使用情况
    match_method_stats = {}
    
    for i, triple in enumerate(input_triples):
        logger.info(f"处理三元组 {i+1}/{len(input_triples)}: {triple['head']} - {triple['relation']} - {triple['tail']}")
        
        normalized_triple, candidates = process_triple(triple, all_entities, name_to_entity, mapping_cache)
        normalized_triples.append(normalized_triple)
        new_entity_candidates.extend(candidates)
        
        # 统计匹配方法
        head_method = normalized_triple['head']['match_method']
        tail_method = normalized_triple['tail']['match_method']
        match_method_stats[head_method] = match_method_stats.get(head_method, 0) + 1
        match_method_stats[tail_method] = match_method_stats.get(tail_method, 0) + 1
    
    # 3. 保存缓存
    mapping_cache.save_cache()
    
    # 4. 输出结果
    os.makedirs(output_dir, exist_ok=True)
    
    # 输出规范化三元组
    normalized_output_path = os.path.join(output_dir, f'normalized_triples_{input_json["title"]}.json')
    with open(normalized_output_path, 'w', encoding='utf-8') as f:
        json.dump(normalized_triples, f, ensure_ascii=False, indent=2)
    
    # 输出新增实体候选
    candidates_output_path = os.path.join(output_dir, f'ontology_update_candidates_{input_json["title"]}.json')
    with open(candidates_output_path, 'w', encoding='utf-8') as f:
        json.dump(new_entity_candidates, f, ensure_ascii=False, indent=2)
    
    # 输出统计信息
    logger.info(f"Pipeline完成! 处理了 {len(input_triples)} 个三元组")
    logger.info(f"规范化结果保存至: {normalized_output_path}")
    logger.info(f"新增实体候选保存至: {candidates_output_path}")
    logger.info(f"发现 {len(new_entity_candidates)} 个新增实体候选")
    
    logger.info("匹配方法统计:")
    for method, count in match_method_stats.items():
        logger.info(f"  {method}: {count} 次")
    
    cache_stats = mapping_cache.get_cache_stats()
    logger.info("缓存统计:")
    for method, count in cache_stats.items():
        logger.info(f"  {method}: {count} 条记录")

# 使用示例
if __name__ == "__main__":
    # 配置文件路径
    ontology_paths = {
        "diseases": "SRP心肌缺血知识图谱/ontology/disease.json",
        "examinations": "SRP心肌缺血知识图谱/ontology/diagnostic_test.json", 
        "symptoms": "SRP心肌缺血知识图谱/ontology/symptoms.json",
        "treatments": "SRP心肌缺血知识图谱/ontology/treatment_intervention.json",
        "patient_characteristics": "SRP心肌缺血知识图谱/ontology/patient_characteristics.json"
    }
    
    output_dir = "SRP心肌缺血知识图谱/output/"
    cache_file_path = "SRP心肌缺血知识图谱/output/entity_mapping_cache.json"  # 映射缓存文件路径
    log_file_path = "SRP心肌缺血知识图谱/output/entity_normalization.log"     # 日志文件路径
    
    target_files = [
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_2_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_2_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_2_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_2_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_1_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_1_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_1_2_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_1_2_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_2_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_2_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_2_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_2_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_2_5_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_1_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_1_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_1_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_2_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_2_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_2_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_2_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_2_5_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_3_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_3_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_3_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_3_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_3_5_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_3_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_2_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_2_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_5_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_5_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_5_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_6_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_1_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_2_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_5_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_6_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_7_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_8_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_9_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_10_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_11_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_3_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_5_4_cot.json",
        # "SRP心肌缺血知识图谱/triples/triple_data/output_chapter_05_cot.json",
        
        # 未处理的文件
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_13_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_2_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_2_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_2_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_2_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_2_5_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_2_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_2_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_2_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_1_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_1_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_1_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_5_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_5_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_6_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_7_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_5_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_6_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_7_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_4_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_1_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_1_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_1_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_1_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_1_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_1_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_2_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_2_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_3_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_3_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_3_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_3_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_4_1_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_4_2_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_4_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_5_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_6_6_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_64_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_66_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_67_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_68_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_69_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_7_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_70_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_72_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_73_cot.json",
        "SRP心肌缺血知识图谱/triples/triple_data_new/output_chapter_8_cot.json",
    ]
    from tqdm import tqdm
    for file in tqdm(target_files):
        input_triples_path = file
        # 运行Pipeline
        run_pipeline(input_triples_path, ontology_paths, output_dir, cache_file_path)