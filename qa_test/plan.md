Graph-guided Answer Validation（推荐）

流程：

LLM 先独立作答（only LLM，保留 reasoning）

抽取 LLM reasoning 中的：

ECG findings

symptom

disease assumption

用图谱做 consistency check：

是否存在 guideline-supported path？

是否违反明确禁忌 / 排除规则？

不改答案 or 触发 re-consider

评测：

accuracy

错误类型减少（contra-guideline errors）

可解释性：给出 violated / supported paths