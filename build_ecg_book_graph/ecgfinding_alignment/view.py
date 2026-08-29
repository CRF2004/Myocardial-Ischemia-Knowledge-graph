import json
with open("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph/ecgfinding_alignment/chapter_extraction_results_disambiguated.json", "r", encoding="utf-8") as f:
    data = json.load(f)
    
d = [str((item["ecg_finding"], item["relation"], item["knowledge_entity"])) for item in data]
print("\n".join(set(d)))