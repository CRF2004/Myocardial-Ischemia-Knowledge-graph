import json
import spacy
from scispacy.umls_linking import UmlsEntityLinker

# 加载模型和UMLS linker
nlp = spacy.load("/mnt/model/en_core_sci_md-0.5.4/en_core_sci_md/en_core_sci_md-0.5.4")
linker = UmlsEntityLinker(resolve_abbreviations=True, threshold=0.8)
nlp.add_pipe(linker)

print("加载完成")

# 允许的词表
ALLOWED_VOCABULARIES = {
    "disease": {"ICD10CM", "SNOMEDCT_US", "SNOMEDCT", "ICD10"},
}

# 输入输出路径
input_path = "SRP心肌缺血知识图谱/ontology/disease.json"
output_path = "SRP心肌缺血知识图谱/ontology/disease_mapped.json"

# 读取输入文件
with open(input_path, "r", encoding="utf-8") as f:
    data = json.load(f)

processed = []
for item in data:
    name = item.get("name_en", "").strip()
    if not name:
        processed.append(item)
        continue

    doc = nlp(name)
    align_results = []

    for ent in doc.ents:
        for umls_ent in ent._.umls_ents:
            cui = umls_ent[0]
            score = umls_ent[1]
            kb_ent = linker.kb.cui_to_entity[cui]

            # 检查词表来源
            vocabularies = {x.split(":")[0] for x in kb_ent.aliases}
            if not vocabularies & ALLOWED_VOCABULARIES["disease"]:
                continue

            # 提取ICD编码（若存在）
            icd_codes = [alias.split(":")[1] for alias in kb_ent.aliases
                         if alias.split(":")[0] in {"ICD10CM", "ICD10"}]

            align_results.append({
                "cui": cui,
                "concept_name": kb_ent.canonical_name,
                "similarity_score": score,
                "icd_codes": icd_codes
            })

    if align_results:
        item["align_result"] = align_results
    else:
        item["align_result"] = []

    processed.append(item)

# 保存结果
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(processed, f, ensure_ascii=False, indent=2)

print(f"完成映射，共处理 {len(processed)} 条实体，输出文件：{output_path}")
