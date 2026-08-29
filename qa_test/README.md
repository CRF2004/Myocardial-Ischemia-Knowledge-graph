- MedQA： /mnt/data/medqa_clean_1_2/relevance_1_data.json、 /mnt/data/medqa_clean_1_2/relevance_2_data.json
- MedExpQA 首次包含了由医生撰写的参考金解释，以便与 LLM 的表现进行比较。



统一 Result Schema，每一条 QA 结果必须包含：
```json
{
  "q_id": "00017",
  "method": "n_hop",
  "model": "gpt-4o-mini",
  "answer": "C",
  "is_correct": null,
  "context": {  // 示例，这里内容视方法而定
    "type": "graph",
    "nodes": [...],
    "edges": [...],
    "linearized_text": "..."
  },
  "retrieval_meta": {   // 示例，这里内容视方法而定
    "hop": 2,
    "num_nodes": 37,
    "num_edges": 52,
    "entity_match_strategy": "label+embedding"
  },
  "generation_meta": {  
    "model": "gpt-4o-mini",
    "temperature": 0.2,
    "prompt_version": "v3"
  }
}
```

eval.py 只关心这些字段


可以考虑利用 Ontology 层级结构（HAS_CHILD 关系）进行知识上采样，例如问题提到具体的某种药物，检索时可以自动带入其所属的药物大类相关知识。