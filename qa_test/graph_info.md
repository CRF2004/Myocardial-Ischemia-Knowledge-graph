## 图谱连接方式
使用py2neo连接Neo4j数据库
- URI: bolt://localhost:7687
- 用户名: neo4j
- 密码: neoneo44j

## Node labels (3505):
- Entity, DiagnosticTest, Ontology, Disease, PatientCharacteristics, Symptoms, TreatmentIntervention, Knowledge, ECGFinding, Treatment, FindingLabel, DiseaseLabel

## Relationship types (3086):
- ASSOCIATED_WITH, PART_OF, CAUSES, AFFECTS, INDICATES, DIAGNOSES, PREVENTS, TREATS, MEASURES, PRECEDES, HAS_CHILD, HAS_ONTOLOGY, SUPPORTS, IS_SPECIFIC_FOR, MAPPED_TO, IS_A, RELATED_TO

## Property keys:
- ontology_id, normalized_name, match_method, text, source_sentences, level, parent_id, definition, id, label, type, name_cn, name_en, umls_definition, umls_all_sources, umls_cui, umls_score, umls_types, umls_sources, umls_canonical_name, umls_mapping_json, knowledge_source, name, mapping_confidence, best_label, best_snomed_id, mapping_rationale, evidence_text, chunk_index, section_title, source, paragraph_index, created_at, norm_status, reason, llm_model, method, updated_at, link_status, link_updated_at, sim_score, confidence, rank, relation_type, similarity, snomed_id

## 不同标签组合的节点属性：

### 主要标签组合：

**['Ontology', 'Symptoms']** 标签组合的节点属性：
发现 5 种不同的属性组合：
  组合1: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合2: definition, id, label, level, name_cn, name_en, ontology_id, parent_id
  组合3: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type
  组合4: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合5: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_types

**['Knowledge', 'Symptoms']** 标签组合的节点属性：
发现 2 种不同的属性组合：
  组合1: match_method, normalized_name, ontology_id, text
  组合2: name, ontology_id

**['Ontology', 'TreatmentIntervention']** 标签组合的节点属性：
发现 3 种不同的属性组合：
  组合1: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type
  组合2: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合3: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types

**['Knowledge', 'TreatmentIntervention']** 标签组合的节点属性：
- match_method, normalized_name, ontology_id, text

### 其他标签组合：

- **['DiagnosticTest', 'Knowledge']**: 2 种属性组合
  组合1: match_method, normalized_name, ontology_id, text
  组合2: name, ontology_id
- **['DiagnosticTest', 'Ontology']**: 4 种属性组合
  组合1: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type
  组合2: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合3: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合4: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_types
- **['Disease', 'Knowledge']**: 2 种属性组合
  组合1: match_method, normalized_name, ontology_id, text
  组合2: name, ontology_id
- **['Disease', 'Ontology']**: 5 种属性组合
  组合1: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合2: definition, id, label, level, name_cn, name_en, ontology_id, parent_id
  组合3: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type
  组合4: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合5: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_types
- **['DiseaseLabel']**: created_at, label, snomed_id, source, updated_at
- **['ECGFinding']**: name
- **['FindingLabel']**: name
- **['Knowledge', 'PatientCharacteristics']**: match_method, normalized_name, ontology_id, text
- **['Knowledge', 'Treatment']**: name, ontology_id
- **['Ontology', 'PatientCharacteristics']**: 3 种属性组合
  组合1: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type
  组合2: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_all_sources, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_sources, umls_types
  组合3: definition, id, label, level, name_cn, name_en, ontology_id, parent_id, type, umls_canonical_name, umls_cui, umls_definition, umls_mapping_json, umls_score, umls_types

## 关系属性组合：

**RELATED_TO** 关系的属性：
发现 2 种不同的属性组合：
  组合1: rank, reason, relation_type, similarity, updated_at
  组合2: rank, reason, relation_type, similarity

**HAS_CHILD** 关系的属性：
- 无属性

**INDICATES** 关系的属性：
发现 2 种不同的属性组合：
  组合1: chunk_index, evidence_text, knowledge_source, paragraph_index, section_title
  组合2: knowledge_source, source_sentences

**ASSOCIATED_WITH** 关系的属性：
- knowledge_source, source_sentences

**TREATS** 关系的属性：
- knowledge_source, source_sentences

**DIAGNOSES** 关系的属性：
- knowledge_source, source_sentences

**HAS_ONTOLOGY** 关系的属性：
发现 2 种不同的属性组合：
  组合1: 无属性
  组合2: reason

**MEASURES** 关系的属性：
- knowledge_source, source_sentences

**CAUSES** 关系的属性：
- knowledge_source, source_sentences

**SUPPORTS** 关系的属性：
- chunk_index, evidence_text, knowledge_source, paragraph_index, section_title

**AFFECTS** 关系的属性：
- knowledge_source, source_sentences

**MAPPED_TO** 关系的属性：
- reason, sim_score

**PREVENTS** 关系的属性：
- knowledge_source, source_sentences

**IS_SPECIFIC_FOR** 关系的属性：
- chunk_index, evidence_text, knowledge_source, paragraph_index, section_title

**IS_A** 关系的属性：
- reason, sim_score

**PART_OF** 关系的属性：
- knowledge_source, source_sentences

**PRECEDES** 关系的属性：
- knowledge_source, source_sentences

## 数量统计信息：

| 类别                    | 数量  |
| ----------------------- | ----- |
| 本体层总实体数          | 592   |
| 本体层-疾病             | 148   |
| 本体层-检验检查         | 228   |
| 本体层-症状体征         | 89    |
| 本体层-干预治疗         | 127   |
| 知识层总实体数          | 787   |
| 知识层-疾病             | 272   |
| 知识层-检验检查         | 294   |
| 知识层-症状体征         | 86    |
| 知识层-干预治疗         | 135   |
| 疾病-检验检查三元组     | 8   |
| 检验检查-症状体征三元组 | 8   |
| 疾病-干预治疗三元组     | 6   |

## 节点数量详情：
- DiagnosticTest: 522
- Disease: 420
- DiseaseLabel: 39
- ECGFinding: 92
- Entity: 0
- FindingLabel: 66
- Knowledge: 946
- Ontology: 708
- PatientCharacteristics: 274
- Symptoms: 175
- Treatment: 1
- TreatmentIntervention: 262

## 关系数量详情：
- AFFECTS: 54
- ASSOCIATED_WITH: 299
- CAUSES: 86
- DIAGNOSES: 160
- HAS_CHILD: 700
- HAS_ONTOLOGY: 115
- INDICATES: 358
- IS_A: 18
- IS_SPECIFIC_FOR: 18
- MAPPED_TO: 39
- MEASURES: 90
- PART_OF: 16
- PRECEDES: 9
- PREVENTS: 34
- RELATED_TO: 872
- SUPPORTS: 57
- TREATS: 161

## 平均度数：1.76091

## 本体最深层数：0

## 本体映射覆盖率：

| mapped | total | coverage |
| ------ | ----- | -------- |
| 470      | 708   | 66.4      |

## 一个疾病连接的检查种类数排名：

| disease                    | test_count |
| -------------------------- | ---------- |
| Unknown                    | 7       |