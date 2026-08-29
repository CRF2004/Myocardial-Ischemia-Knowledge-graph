"""
对齐（反向）：PTB-XL心电图 finding 标签 → 书籍图谱提取的 ECGFinding 实体

处理流程：
    加载书籍图谱提取的 JSON 数据（字典列表），收集独立的 ECGFinding（实体集合，作为 candidates）。
    加载 PTB-XL 的 finding_scp_label.csv（作为 query：每条 finding label / description）。

为以下文本计算嵌入向量：
    query：PTB-XL 的 description（或你可改为 scp_code+description）
    candidates：书籍 ECGFinding 文本

对每个 query：通过余弦相似度检索前 k 个书籍 ECGFinding 候选项。
    使用大语言模型判断关系类型：精确匹配/上位概念/无关联。
    若为无关联：则不输出该映射条目。
按指定模式输出 JSON 格式结果。
"""
import os
import json
from tqdm import tqdm
from datetime import datetime, timezone

import pandas as pd
import numpy as np

from embedding import get_embedding
from LLM_CALL import chat


# =========================
# Paths (as provided)
# =========================
BOOK_GRAPH_JSON = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph/ecgfinding_alignment/chapter_extraction_results.json"
FINDING_LABEL_CSV = "/mnt/data/ptb-xl-a-comprehensive-electrocardiographic-feature-dataset-1.0.1/labels/finding_scp_label.csv"
OUTPUT_JSON = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/build_ecg_book_graph/ecgfinding_alignment/ptbxl_scp_to_book_ecgfinding.json"


# =========================
# Config
# =========================
EMBED_MODEL = "text-embedding-3-small"
LLM_MODEL = "gpt-4o-mini"
TOP_K = 8

# be stricter/looser on retrieval before LLM:
MIN_RETRIEVAL_SIM = None


# =========================
# Utils
# =========================
def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def cosine_sim_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """
    a: (n, d), b: (m, d) -> sims: (n, m)
    """
    a_norm = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_norm @ b_norm.T


def safe_json_loads(s: str):
    """
    Try parse JSON strictly; if LLM wraps with code fences or extra text, try to extract first JSON object/array.
    """
    s = (s or "").strip()
    # Remove code fences if present
    if s.startswith("```"):
        s = s.strip("`")
        lines = s.splitlines()
        if len(lines) >= 2 and lines[0].lower().startswith("json"):
            s = "\n".join(lines[1:])
    s = s.strip()

    try:
        return json.loads(s)
    except Exception:
        pass

    # Try to extract JSON array/object substring
    first_obj = s.find("{")
    first_arr = s.find("[")
    if first_obj == -1 and first_arr == -1:
        return None
    start = first_arr if (first_arr != -1 and (first_obj == -1 or first_arr < first_obj)) else first_obj

    # Find matching end by scanning
    stack = []
    for i in range(start, len(s)):
        ch = s[i]
        if ch in "{[":
            stack.append(ch)
        elif ch in "}]":
            if not stack:
                continue
            open_ch = stack.pop()
            if (open_ch == "{" and ch != "}") or (open_ch == "[" and ch != "]"):
                return None
            if not stack:
                candidate = s[start:i + 1]
                try:
                    return json.loads(candidate)
                except Exception:
                    return None
    return None


def build_llm_prompt(query_label: str, candidates: list) -> str:
    """
    query_label: PTB-XL finding label text (e.g., description)
    candidates: list of dicts:
      { "ecg_finding": ..., "similarity": float }
    LLM should classify each candidate into MAPS_TO / SUPERTYPE_OF / NONE
    """
    cand_lines = []
    for idx, c in enumerate(candidates, start=1):
        cand_lines.append(
            f'{idx}. ecg_finding="{c["ecg_finding"]}"'
        )
    cand_block = "\n".join(cand_lines)

    return f"""
    You are aligning PTB-XL ECG finding labels (SCP-coded descriptions)
    to ECG finding entities extracted from a textbook knowledge graph.

    The goal is STRICT SEMANTIC ALIGNMENT, not clinical reasoning.

    Given ONE query PTB-XL label and a ranked list of candidate textbook ECG findings,
    decide the correct alignment relation for each candidate.

    Allowed relation types (choose exactly one per candidate):

    - MAPS_TO:
    The candidate textbook finding is semantically equivalent to the query label
    (same ECG finding, same meaning, possible wording variation).

    - SUPERTYPE_OF:
    The candidate textbook finding is a more general supertype of the query label
    (query ⊂ candidate). Example: query="Anterior ST elevation" and candidate="ST segment elevation".

    - NONE:
    No valid semantic alignment. Do NOT map.

    Important rules (follow strictly):

    - MAPS_TO is ONLY for true equivalence or synonymy.
    Do NOT use MAPS_TO for findings that merely suggest, indicate,
    support, or are evidence for a disease.

    - SUPERTYPE_OF is mainly for clear taxonomic containment
    (query is a subtype of candidate). Do NOT infer beyond ECG morphology.

    - If the query is a waveform-level ECG pattern and the candidate is
    a higher-level clinical interpretation (ischemia, injury, infarction),
    the correct choice is NONE.

    - Do NOT infer causality, diagnostic meaning, or clinical implication.
    This task is alignment, NOT reasoning.

    - If unsure between MAPS_TO / SUPERTYPE_OF and NONE, choose NONE.

    Return JSON only (no code fences), with this schema:
    {{
    "decisions": [
        {{
        "rank": 1,
        "relation_type": "MAPS_TO|SUPERTYPE_OF|NONE",
        "reason": "short explanation (<=30 chars)"
        }},
        ...
    ]
    }}

    Query PTB-XL label:
    "{query_label}"

    Candidates:
    {cand_block}
        """.strip()


# =========================
# Load inputs
# =========================
with open(BOOK_GRAPH_JSON, "r", encoding="utf-8") as f:
    book_rows = json.load(f)

# Collect unique textbook ECGFinding entities (candidates)
book_findings = sorted({r.get("ecg_finding", "").strip() for r in book_rows if r.get("ecg_finding")})
if not book_findings:
    raise ValueError("No ecg_finding found in the input JSON.")

# Load PTB-XL finding labels (queries)
df = pd.read_csv(FINDING_LABEL_CSV)

# By default use description as query text; you can swap to:
# queries = [f'{c} | {d}' for c, d in zip(df["scp_code"].astype(str), df["description"].astype(str))]
queries = df["description"].astype(str).tolist()
scp_codes = df["scp_code"].astype(str).tolist()
statement_category = df["Statement Category"].astype(str).tolist()

# =========================
# Embeddings (batch)
# =========================
# Embedding for all queries (PTB-XL labels)
query_embs = get_embedding(EMBED_MODEL, queries)
query_embs = np.array(query_embs, dtype=np.float32)

# Embedding for all candidates (book findings)
cand_embs = get_embedding(EMBED_MODEL, book_findings)
cand_embs = np.array(cand_embs, dtype=np.float32)

# Compute similarity matrix (n_queries x n_candidates)
sims = cosine_sim_matrix(query_embs, cand_embs)


# =========================
# Align each query(label) -> topK book findings -> LLM decision -> output
# =========================
items = []
generated_at = _now_iso_utc()

for i, query_text in tqdm(list(enumerate(queries))):
    sim_row = sims[i]  # (n_candidates,)
    top_idx = np.argsort(-sim_row)[:TOP_K]

    candidates = []
    for cand_idx in top_idx:
        simv = float(sim_row[cand_idx])
        if MIN_RETRIEVAL_SIM is not None and simv < MIN_RETRIEVAL_SIM:
            continue
        candidates.append({
            "rank": len(candidates) + 1,
            "ecg_finding": book_findings[cand_idx],
            "similarity": simv
        })

    if not candidates:
        continue

    prompt = build_llm_prompt(query_text, candidates)
    llm_resp = chat(LLM_MODEL, prompt)
    parsed = safe_json_loads(llm_resp)
    print(query_text)
    print(candidates)
    print(json.dumps(parsed, indent=2, ensure_ascii=False))

    # If parsing fails, fall back to marking all as NONE (conservative)
    decisions = None
    if isinstance(parsed, dict) and isinstance(parsed.get("decisions"), list):
        decisions = parsed["decisions"]

    mappings = []
    if decisions is not None:
        dec_by_rank = {}
        for d in decisions:
            try:
                rnk = int(d.get("rank"))
                rel = str(d.get("relation_type", "NONE")).strip().upper()
                reason = str(d.get("reason", "")).strip()
                if rel not in {"MAPS_TO", "SUPERTYPE_OF", "NONE"}:
                    rel = "NONE"
                dec_by_rank[rnk] = (rel, reason)
            except Exception:
                continue

        for c in candidates:
            rnk = c["rank"]
            rel, reason = dec_by_rank.get(rnk, ("NONE", "LLM无有效输出"))
            if rel == "NONE":
                continue
            mappings.append({
                "edge_type": "RELATED_TO",
                "relation_type": rel,
                "reason": reason,
                "rank": rnk,
                "similarity": c["similarity"],
                "node_text": c["ecg_finding"]
            })

    if not mappings:
        continue

    items.append({
        # 这里把“PTB-XL 标签”作为主键
        "scp_code": str(scp_codes[i]),
        "label_text": query_text,
        "statement_category": str(statement_category[i]),
        "top_k": TOP_K,
        "generated_at": generated_at,
        "mappings": mappings
    })


output = {
    "meta": {
        "task": "align_ptbxl_scp_labels_to_book_ecg_findings",
        "embedding_model": EMBED_MODEL,
        "llm_model": LLM_MODEL,
        "top_k": TOP_K,
        "generated_at": generated_at,
        "input_book_graph": BOOK_GRAPH_JSON,
        "input_label_csv": FINDING_LABEL_CSV,
        "output_path": OUTPUT_JSON
    },
    "items": items
}

os.makedirs(os.path.dirname(OUTPUT_JSON), exist_ok=True)
with open(OUTPUT_JSON, "w", encoding="utf-8") as f:
    json.dump(output, f, ensure_ascii=False, indent=2)

print(f"Done. Wrote {len(items)} aligned PTB-XL labels to: {OUTPUT_JSON}")
