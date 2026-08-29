# Myocardial Ischemia Knowledge Graph

心肌缺血临床知识图谱研究原型。本项目围绕临床指南知识的结构化、实体对齐、图谱构建与可解释推理展开，目标是探索如何把分散的医学证据组织为可追溯、可查询的知识结构。

## 项目概览

- 以 2025 AHA 相关临床指南为主要知识来源，设计心肌缺血领域本体。
- 构建知识抽取与实体融合流程，累计形成 1,000 余条三元组。
- 通过 UMLS linking 和多人协作维护流程处理术语对齐与本体更新。
- 探索 XGBoost 黑盒预测结果与知识图谱路径之间的映射，用于辅助解释模型输出。
- 项目由五人跨学科团队完成，并形成软件著作权成果。

## 仓库结构

| 目录 | 内容 |
| --- | --- |
| `ontology/` | 疾病、症状、检查、患者特征与治疗干预本体 |
| `triples/` | 指南知识抽取、提示词与实体对齐代码 |
| `ontology_process/` | UMLS linking、本体维护与人工审核流程 |
| `upload_graph/` | Neo4j 图谱构建与更新脚本 |
| `export_graph/` | 可展示的节点、边数据与导出脚本 |
| `graph_embedding/` | 图谱向量化与检索代码 |
| `inference_systems/` | 图谱推理及模型—图谱联合实验 |
| `model/` | 特征选择、数据划分与传统机器学习代码 |
| `qa_test/` | RAG、路径检索与问答实验代码 |
| `evaluation/` | 上游、中游与下游评估脚本 |

## 快速开始

建议使用 Python 3.10 或更新版本：

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

如需调用 LLM 或连接 Neo4j，请通过环境变量提供配置：

```bash
export DMX_API_KEY="your-api-key"
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your-password"
```

不同实验脚本的数据入口仍保留原型阶段的目录约定，运行前请根据本地数据位置调整参数。

## 公开版本说明

本仓库是用于研究交流与面试展示的整理版本。出于版权、隐私和仓库体积考虑，以下内容未上传：临床指南 PDF 与解析后的大段原文、训练数据、模型权重、向量索引、批量 LLM 响应、运行日志和访问凭据。仓库保留了核心本体、图谱样例、方法代码和实验设计。

## 研究反思

这个项目验证了从临床指南到知识图谱原型的完整链路，也暴露了两个重要限制：图谱质量尚缺少系统的形式化评估，下游问答与 RAG 效果尚未达到稳定应用水平。这些问题进一步推动了我对医学 AI 中证据审计、可追溯性和可靠评估的关注。

## License

当前未添加开源许可证。未经许可，请勿将本仓库内容用于商业用途。
