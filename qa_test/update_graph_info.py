#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
"""
Neo4j图谱信息收集脚本
用于收集图谱的统计信息并更新graph_info.md文件
"""

from py2neo import Graph
from typing import Dict, List, Tuple, Any, Set
from collections import defaultdict


def connect_graph() -> Graph:
    """连接到Neo4j图谱数据库"""
    graph = Graph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password=os.environ.get("NEO4J_PASSWORD", "")
    )
    return graph


def get_node_labels(graph: Graph) -> List[str]:
    """获取所有节点标签"""
    result = graph.run("CALL db.labels() YIELD label RETURN label")
    return [record["label"] for record in result]


def get_relationship_types(graph: Graph) -> List[str]:
    """获取所有关系类型"""
    result = graph.run("CALL db.relationshipTypes() YIELD relationshipType RETURN relationshipType")
    return [record["relationshipType"] for record in result]


def get_property_keys(graph: Graph) -> List[str]:
    """获取所有属性键"""
    result = graph.run("CALL db.propertyKeys() YIELD propertyKey RETURN propertyKey")
    return [record["propertyKey"] for record in result]


def count_nodes_by_label(graph: Graph, label: str) -> int:
    """统计指定标签的节点数量"""
    result = graph.run(f"MATCH (n:{label}) RETURN count(n) as count")
    for record in result:
        return record["count"]
    return 0


def count_relationships_by_type(graph: Graph, rel_type: str) -> int:
    """统计指定类型的关系数量"""
    result = graph.run(f"MATCH ()-[r:{rel_type}]->() RETURN count(r) as count")
    for record in result:
        return record["count"]
    return 0


def get_ontology_layer_counts(graph: Graph) -> Dict[str, int]:
    """获取本体层实体数量统计"""
    # 本体层实体具有Ontology标签或通过HAS_ONTOLOGY关系连接
    stats = {
        "总实体数": 0,
        "疾病": 0,
        "检验检查": 0,
        "症状体征": 0,
        "干预治疗": 0
    }

    # 统计本体层各类型
    result1 = graph.run("MATCH (n:Ontology:Disease) RETURN count(n) as count")
    for record in result1:
        ontology_disease = record["count"]

    result2 = graph.run("MATCH (n:Ontology:DiagnosticTest) RETURN count(n) as count")
    for record in result2:
        ontology_test = record["count"]

    result3 = graph.run("MATCH (n:Ontology:Symptoms) RETURN count(n) as count")
    for record in result3:
        ontology_symptom = record["count"]

    result4 = graph.run("MATCH (n:Ontology:TreatmentIntervention) RETURN count(n) as count")
    for record in result4:
        ontology_treatment = record["count"]

    stats["疾病"] = ontology_disease
    stats["检验检查"] = ontology_test
    stats["症状体征"] = ontology_symptom
    stats["干预治疗"] = ontology_treatment
    stats["总实体数"] = ontology_disease + ontology_test + ontology_symptom + ontology_treatment

    return stats


def get_knowledge_layer_counts(graph: Graph) -> Dict[str, int]:
    """获取知识层实体数量统计"""
    stats = {
        "总实体数": 0,
        "疾病": 0,
        "检验检查": 0,
        "症状体征": 0,
        "干预治疗": 0
    }

    # 统计知识层各类型（非本体层的节点）
    result1 = graph.run("MATCH (n:Knowledge:Disease) WHERE NOT n:Ontology RETURN count(n) as count")
    for record in result1:
        knowledge_disease = record["count"]

    result2 = graph.run("MATCH (n:Knowledge:DiagnosticTest) WHERE NOT n:Ontology RETURN count(n) as count")
    for record in result2:
        knowledge_test = record["count"]

    result3 = graph.run("MATCH (n:Knowledge:Symptoms) WHERE NOT n:Ontology RETURN count(n) as count")
    for record in result3:
        knowledge_symptom = record["count"]

    result4 = graph.run("MATCH (n:Knowledge:TreatmentIntervention) WHERE NOT n:Ontology RETURN count(n) as count")
    for record in result4:
        knowledge_treatment = record["count"]

    stats["疾病"] = knowledge_disease
    stats["检验检查"] = knowledge_test
    stats["症状体征"] = knowledge_symptom
    stats["干预治疗"] = knowledge_treatment
    stats["总实体数"] = knowledge_disease + knowledge_test + knowledge_symptom + knowledge_treatment

    return stats


def count_triples(graph: Graph) -> Dict[str, int]:
    """统计三元组数量"""
    triples = {
        "疾病-检验检查三元组": 0,
        "检验检查-症状体征三元组": 0,
        "疾病-干预治疗三元组": 0
    }

    # 统计疾病-检验检查三元组（Disease -> INDICATES -> DiagnosticTest）
    result1 = graph.run("""
        MATCH (d:Disease)-[r:INDICATES]->(t:DiagnosticTest)
        RETURN count(r) as count
    """)
    for record in result1:
        disease_test = record["count"]
    triples["疾病-检验检查三元组"] = disease_test

    # 统计检验检查-症状体征三元组
    result2 = graph.run("""
        MATCH (t:DiagnosticTest)-[r:INDICATES]->(s:Symptoms)
        RETURN count(r) as count
    """)
    for record in result2:
        test_symptom = record["count"]
    triples["检验检查-症状体征三元组"] = test_symptom

    # 统计疾病-干预治疗三元组
    result3 = graph.run("""
        MATCH (d:Disease)-[r:TREATS]->(t:TreatmentIntervention)
        RETURN count(r) as count
    """)
    for record in result3:
        disease_treatment = record["count"]
    triples["疾病-干预治疗三元组"] = disease_treatment

    return triples


def calculate_avg_degree(graph: Graph, total_nodes: int, total_relationships: int) -> float:
    """计算平均度数"""
    if total_nodes == 0:
        return 0.0
    # 平均度数 = 总关系数 * 2 / 总节点数（无向图每个边算两次）
    return round(total_relationships * 2 / total_nodes, 5)


def get_max_ontology_depth(graph: Graph) -> int:
    """计算本体最深层数（基于HAS_CHILD关系）"""
    result = graph.run("""
        MATCH (n:Ontology)
        WHERE NOT (n)<-[:HAS_CHILD]-()
        MATCH path = (n)<-[:HAS_CHILD*]-(m)
        RETURN max(length(path)) as max_depth
    """)
    for record in result:
        max_depth = record["max_depth"]
        return max_depth if max_depth is not None else 0
    return 0


def get_ontology_mapping_coverage(graph: Graph) -> Dict[str, Any]:
    """计算本体映射覆盖率"""
    # 检查有多少Ontology节点有umls_cui属性
    result = graph.run("""
        MATCH (n:Ontology)
        RETURN count(n) as total, count(n.umls_cui) as mapped
    """)
    for record in result:
        total = record["total"]
        mapped = record["mapped"]
        coverage = round(mapped / total * 100, 1) if total > 0 else 0.0

        return {
            "mapped": mapped,
            "total": total,
            "coverage": coverage
        }
    return {"mapped": 0, "total": 0, "coverage": 0.0}


def get_disease_test_ranking(graph: Graph, top_k: int = 10) -> List[Tuple[str, int]]:
    """获取疾病连接的检查种类数排名"""
    result = graph.run(f"""
        MATCH (d:Disease)-[:INDICATES]->(t:DiagnosticTest)
        RETURN d.name_cn as disease, count(DISTINCT t) as test_count
        ORDER BY test_count DESC
        LIMIT {top_k}
    """)
    return [(record["disease"] or "Unknown", record["test_count"]) for record in result]


def get_label_property_combinations(graph: Graph) -> Dict[str, List[Set[str]]]:
    """
    获取不同标签组合的节点属性组合

    对于每个节点的标签组合（如[Ontology, Disease]），收集该标签组合下所有节点的属性集合，
    并返回去重后的属性组合列表

    返回格式：
    {
        "[Ontology]": [{"属性1", "属性2", ...}],
        "[Knowledge, Disease]": [{"属性1", "属性2"}, {"属性3", "属性4"}],  # 可能有多种组合
        ...
    }
    """
    from collections import defaultdict

    label_comb_property_sets = defaultdict(set)

    # 获取所有节点及其标签和属性
    result = graph.run("""
        MATCH (n)
        RETURN labels(n) as node_labels, keys(n) as properties
        LIMIT 10000
    """)

    # 按标签组合收集属性
    for record in result:
        node_labels = tuple(sorted(record["node_labels"]))  # 排序确保标签组合一致
        props = frozenset(record["properties"])

        if node_labels:
            label_comb_property_sets[node_labels].add(props)

    # 转换为列表格式
    label_comb_property_sets_list = {}
    for label_comb, prop_sets in label_comb_property_sets.items():
        label_comb_property_sets_list[str(list(label_comb))] = [set(props) for props in prop_sets]

    return label_comb_property_sets_list


def get_relationship_property_combinations(graph: Graph) -> Dict[str, List[Set[str]]]:
    """
    获取不同关系类型的属性组合

    对于每种关系类型，收集该类型下所有关系的属性集合，并返回去重后的属性组合列表

    返回格式：
    {
        "INDICATES": [{"属性1", "属性2"}, {"属性1", "属性2", "属性3"}],  # 可能有多种组合
        "TREATS": [{"属性1"}],
        ...
    }
    """
    from collections import defaultdict

    rel_type_property_sets = defaultdict(set)

    # 获取所有关系及其类型和属性
    result = graph.run("""
        MATCH ()-[r]->()
        RETURN type(r) as rel_type, keys(r) as properties
        LIMIT 50000
    """)

    # 按关系类型收集属性
    for record in result:
        rel_type = record["rel_type"]
        props = frozenset(record["properties"])

        if rel_type:
            rel_type_property_sets[rel_type].add(props)

    # 转换为列表格式
    rel_type_property_sets_list = {}
    for rel_type, prop_sets in rel_type_property_sets.items():
        rel_type_property_sets_list[rel_type] = [set(props) for props in prop_sets]

    return rel_type_property_sets_list


def generate_markdown(graph_info: Dict[str, Any]) -> str:
    """生成Markdown格式的内容"""
    md_lines = []

    # 图谱连接方式
    md_lines.append("## 图谱连接方式")
    md_lines.append("使用py2neo连接Neo4j数据库")
    md_lines.append("- URI: bolt://localhost:7687")
    md_lines.append("- 用户名: neo4j")
    md_lines.append("- 密码: neoneo44j")
    md_lines.append("")

    # Node labels
    total_nodes = sum(graph_info["node_counts"].values())
    node_labels = ", ".join(graph_info["node_labels"])
    md_lines.append(f"## Node labels ({total_nodes}):")
    md_lines.append(f"- {node_labels}")
    md_lines.append("")

    # Relationship types
    total_rels = sum(graph_info["relationship_counts"].values())
    rel_types = ", ".join(graph_info["relationship_types"])
    md_lines.append(f"## Relationship types ({total_rels}):")
    md_lines.append(f"- {rel_types}")
    md_lines.append("")

    # Property keys
    prop_keys = ", ".join(graph_info["property_keys"])
    md_lines.append("## Property keys:")
    md_lines.append(f"- {prop_keys}")
    md_lines.append("")

    # 标签组合属性
    md_lines.append("## 不同标签组合的节点属性：")
    md_lines.append("")
    label_props = graph_info["label_property_combinations"]

    # 定义优先展示的标签组合（本体层）
    priority_combinations = [
        "['Ontology']",
        "['Knowledge']",
        "['Ontology', 'Disease']",
        "['Knowledge', 'Disease']",
        "['Ontology', 'DiagnosticTest']",
        "['Knowledge', 'DiagnosticTest']",
        "['Ontology', 'Symptoms']",
        "['Knowledge', 'Symptoms']",
        "['Ontology', 'TreatmentIntervention']",
        "['Knowledge', 'TreatmentIntervention']"
    ]

    # 展示优先级标签组合
    md_lines.append("### 主要标签组合：")
    md_lines.append("")
    for label_comb in priority_combinations:
        if label_comb in label_props:
            combinations = label_props[label_comb]
            md_lines.append(f"**{label_comb}** 标签组合的节点属性：")
            if len(combinations) == 1:
                props = ", ".join(sorted(combinations[0]))
                md_lines.append(f"- {props}")
            else:
                md_lines.append(f"发现 {len(combinations)} 种不同的属性组合：")
                for i, props in enumerate(combinations, 1):
                    props_str = ", ".join(sorted(props))
                    md_lines.append(f"  组合{i}: {props_str}")
            md_lines.append("")

    # 展示其他标签组合
    other_combinations = [comb for comb in label_props.keys() if comb not in priority_combinations]
    if other_combinations:
        md_lines.append("### 其他标签组合：")
        md_lines.append("")
        for label_comb in sorted(other_combinations):
            combinations = label_props[label_comb]
            if len(combinations) == 1:
                props = ", ".join(sorted(combinations[0]))
                md_lines.append(f"- **{label_comb}**: {props}")
            else:
                md_lines.append(f"- **{label_comb}**: {len(combinations)} 种属性组合")
                for i, props in enumerate(combinations, 1):
                    props_str = ", ".join(sorted(props))
                    md_lines.append(f"  组合{i}: {props_str}")
        md_lines.append("")

    # 关系属性组合
    md_lines.append("## 关系属性组合：")
    md_lines.append("")
    rel_props = graph_info["relationship_property_combinations"]

    # 按关系数量排序展示
    sorted_rel_types = sorted(rel_props.items(), key=lambda x: graph_info["relationship_counts"].get(x[0], 0), reverse=True)

    for rel_type, combinations in sorted_rel_types:
        md_lines.append(f"**{rel_type}** 关系的属性：")
        if len(combinations) == 0:
            md_lines.append("- 无属性")
        elif len(combinations) == 1:
            if len(combinations[0]) == 0:
                md_lines.append("- 无属性")
            else:
                props = ", ".join(sorted(combinations[0]))
                md_lines.append(f"- {props}")
        else:
            md_lines.append(f"发现 {len(combinations)} 种不同的属性组合：")
            for i, props in enumerate(combinations, 1):
                if len(props) == 0:
                    props_str = "无属性"
                else:
                    props_str = ", ".join(sorted(props))
                md_lines.append(f"  组合{i}: {props_str}")
        md_lines.append("")

    # 数量统计信息
    md_lines.append("## 数量统计信息：")
    md_lines.append("")
    md_lines.append("| 类别                    | 数量  |")
    md_lines.append("| ----------------------- | ----- |")

    ontology_stats = graph_info["ontology_layer"]
    md_lines.append(f"| 本体层总实体数          | {ontology_stats['总实体数']}   |")
    md_lines.append(f"| 本体层-疾病             | {ontology_stats['疾病']}   |")
    md_lines.append(f"| 本体层-检验检查         | {ontology_stats['检验检查']}   |")
    md_lines.append(f"| 本体层-症状体征         | {ontology_stats['症状体征']}    |")
    md_lines.append(f"| 本体层-干预治疗         | {ontology_stats['干预治疗']}   |")

    knowledge_stats = graph_info["knowledge_layer"]
    md_lines.append(f"| 知识层总实体数          | {knowledge_stats['总实体数']}   |")
    md_lines.append(f"| 知识层-疾病             | {knowledge_stats['疾病']}   |")
    md_lines.append(f"| 知识层-检验检查         | {knowledge_stats['检验检查']}   |")
    md_lines.append(f"| 知识层-症状体征         | {knowledge_stats['症状体征']}    |")
    md_lines.append(f"| 知识层-干预治疗         | {knowledge_stats['干预治疗']}   |")

    triples = graph_info["triples"]
    md_lines.append(f"| 疾病-检验检查三元组     | {triples['疾病-检验检查三元组']}   |")
    md_lines.append(f"| 检验检查-症状体征三元组 | {triples['检验检查-症状体征三元组']}   |")
    md_lines.append(f"| 疾病-干预治疗三元组     | {triples['疾病-干预治疗三元组']}   |")
    md_lines.append("")

    # 节点数量详情
    md_lines.append("## 节点数量详情：")
    for label, count in sorted(graph_info["node_counts"].items()):
        md_lines.append(f"- {label}: {count}")
    md_lines.append("")

    # 关系数量详情
    md_lines.append("## 关系数量详情：")
    for rel_type, count in sorted(graph_info["relationship_counts"].items()):
        md_lines.append(f"- {rel_type}: {count}")
    md_lines.append("")

    # 平均度数
    md_lines.append(f"## 平均度数：{graph_info['avg_degree']}")
    md_lines.append("")

    # 本体最深层数
    md_lines.append(f"## 本体最深层数：{graph_info['max_ontology_depth']}")
    md_lines.append("")

    # 本体映射覆盖率
    mapping = graph_info["ontology_mapping"]
    md_lines.append("## 本体映射覆盖率：")
    md_lines.append("")
    md_lines.append("| mapped | total | coverage |")
    md_lines.append("| ------ | ----- | -------- |")
    md_lines.append(f"| {mapping['mapped']}      | {mapping['total']}   | {mapping['coverage']}      |")
    md_lines.append("")

    # 疾病连接的检查种类数排名
    md_lines.append("## 一个疾病连接的检查种类数排名：")
    md_lines.append("")
    md_lines.append("| disease                    | test_count |")
    md_lines.append("| -------------------------- | ---------- |")
    for disease, count in graph_info["disease_test_ranking"]:
        md_lines.append(f"| {disease}                    | {count}       |")

    return "\n".join(md_lines)


def main():
    """主函数"""
    print("正在连接Neo4j数据库...")
    graph = connect_graph()
    print("连接成功！")

    print("\n正在收集图谱信息...")

    # 获取基础信息
    node_labels = get_node_labels(graph)
    relationship_types = get_relationship_types(graph)
    property_keys = get_property_keys(graph)

    # 统计节点数量
    print("统计节点数量...")
    node_counts = {label: count_nodes_by_label(graph, label) for label in node_labels}

    # 统计关系数量
    print("统计关系数量...")
    relationship_counts = {rel_type: count_relationships_by_type(graph, rel_type) for rel_type in relationship_types}

    # 获取本体层和知识层统计
    print("统计本体层和知识层...")
    ontology_layer = get_ontology_layer_counts(graph)
    knowledge_layer = get_knowledge_layer_counts(graph)

    # 统计三元组
    print("统计三元组...")
    triples = count_triples(graph)

    # 计算平均度数
    total_nodes = sum(node_counts.values())
    total_relationships = sum(relationship_counts.values())
    avg_degree = calculate_avg_degree(graph, total_nodes, total_relationships)

    # 获取本体最深层数
    print("计算本体最深层数...")
    max_ontology_depth = get_max_ontology_depth(graph)

    # 获取本体映射覆盖率
    print("计算本体映射覆盖率...")
    ontology_mapping = get_ontology_mapping_coverage(graph)

    # 获取疾病连接的检查种类数排名
    print("生成疾病-检查排名...")
    disease_test_ranking = get_disease_test_ranking(graph, top_k=10)

    # 获取标签属性组合
    print("分析标签属性组合...")
    label_property_combinations = get_label_property_combinations(graph)

    # 获取关系属性组合
    print("分析关系属性组合...")
    relationship_property_combinations = get_relationship_property_combinations(graph)

    # 整合所有信息
    graph_info = {
        "node_labels": node_labels,
        "relationship_types": relationship_types,
        "property_keys": property_keys,
        "node_counts": node_counts,
        "relationship_counts": relationship_counts,
        "ontology_layer": ontology_layer,
        "knowledge_layer": knowledge_layer,
        "triples": triples,
        "avg_degree": avg_degree,
        "max_ontology_depth": max_ontology_depth,
        "ontology_mapping": ontology_mapping,
        "disease_test_ranking": disease_test_ranking,
        "label_property_combinations": label_property_combinations,
        "relationship_property_combinations": relationship_property_combinations
    }

    print("\n正在生成Markdown内容...")
    markdown_content = generate_markdown(graph_info)

    # 写入文件
    output_file = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/graph_info.md"
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write(markdown_content)

    print(f"\n✓ 图谱信息已成功更新到 {output_file}")
    print("\n统计摘要：")
    print(f"  - 总节点数: {total_nodes}")
    print(f"  - 总关系数: {total_relationships}")
    print(f"  - 节点标签数: {len(node_labels)}")
    print(f"  - 关系类型数: {len(relationship_types)}")
    print(f"  - 平均度数: {avg_degree}")
    print(f"  - 本体最深层数: {max_ontology_depth}")


if __name__ == "__main__":
    main()