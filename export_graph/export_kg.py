#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
"""
导出Neo4j知识图谱数据到CSV文件
导出指定标签的节点和相关边
"""

from py2neo import Graph
import pandas as pd
import json
from datetime import datetime


def serialize_props(props):
    """
    序列化属性，处理DateTime等非JSON可序列化的对象
    :param props: 属性字典
    :return: 可JSON序列化的字典
    """
    serialized = {}
    for key, value in props.items():
        if isinstance(value, datetime):
            serialized[key] = value.isoformat()
        elif isinstance(value, list):
            # 处理列表中的非JSON可序列化对象
            serialized[key] = [item.isoformat() if isinstance(item, datetime) else item for item in value]
        else:
            try:
                # 尝试直接序列化，如果失败则转换为字符串
                json.dumps(value)
                serialized[key] = value
            except (TypeError, ValueError):
                serialized[key] = str(value)
    return serialized


def connect_to_neo4j():
    """连接到Neo4j数据库"""
    graph = Graph(
        "bolt://localhost:7687",
        user="neo4j",
        password=os.environ.get("NEO4J_PASSWORD", "")
    )
    return graph


def export_nodes(graph, target_labels):
    """
    导出节点数据
    :param graph: Neo4j图对象
    :param target_labels: 目标节点标签列表
    :return: DataFrame包含节点数据
    """
    nodes_data = []

    for label in target_labels:
        # 查询指定标签的所有节点
        query = f"""
        MATCH (n:{label})
        RETURN n
        """
        result = graph.run(query)

        for record in result:
            node = record['n']
            node_id = node.identity
            node_type = label

            # 获取节点属性
            props = dict(node)

            # 尝试获取name字段
            name = props.get('name', props.get('label', props.get('name_en', props.get('name_cn', ''))))

            # 获取source字段
            source = props.get('source', props.get('knowledge_source', ''))

            # 将所有属性存入meta字段（处理DateTime等非JSON序列化对象）
            serialized_props = serialize_props(props)
            meta = json.dumps(serialized_props, ensure_ascii=False)

            nodes_data.append({
                'node_id': node_id,
                'node_type': node_type,
                'name': name,
                'source': source,
                'meta': meta
            })

    return pd.DataFrame(nodes_data)


def export_edges(graph, node_ids):
    """
    导出边数据
    :param graph: Neo4j图对象
    :param node_ids: 目标节点ID列表
    :return: DataFrame包含边数据
    """
    edges_data = []

    # 将node_ids转换为字符串列表用于查询
    node_ids_str = ','.join(map(str, node_ids))

    # 查询与这些节点相关的所有边（作为源节点或目标节点）
    query = f"""
    MATCH (a)-[r]->(b)
    WHERE ID(a) IN [{node_ids_str}] OR ID(b) IN [{node_ids_str}]
    RETURN ID(a) AS src_id, type(r) AS rel_type, ID(b) AS tgt_id, r
    """
    result = graph.run(query)

    for record in result:
        src_id = record['src_id']
        rel_type = record['rel_type']
        tgt_id = record['tgt_id']
        relationship = record['r']

        # 获取关系属性
        props = dict(relationship)

        # 获取confidence字段
        confidence = props.get('confidence', props.get('sim_score', props.get('similarity', 1.0)))

        # 将所有属性存入meta字段（处理DateTime等非JSON序列化对象）
        serialized_props = serialize_props(props)
        meta = json.dumps(serialized_props, ensure_ascii=False)

        edges_data.append({
            'src_id': src_id,
            'rel_type': rel_type,
            'tgt_id': tgt_id,
            'confidence': confidence,
            'meta': meta
        })

    return pd.DataFrame(edges_data)


def main():
    """主函数"""
    print("开始导出知识图谱数据...")

    # 连接到Neo4j
    print("连接到Neo4j数据库...")
    graph = connect_to_neo4j()

    # 定义要导出的节点标签
    target_labels = ["FindingLabel", "ECGFinding", "Disease", "DiseaseLabel"]

    # 导出节点
    print(f"导出节点: {target_labels}")
    nodes_df = export_nodes(graph, target_labels)
    nodes_df.to_csv('nodes.csv', index=False, encoding='utf-8-sig')
    print(f"已导出 {len(nodes_df)} 个节点到 nodes.csv")

    # 获取所有节点ID用于查询相关边
    node_ids = nodes_df['node_id'].tolist()

    # 导出边
    print("导出相关边...")
    edges_df = export_edges(graph, node_ids)
    edges_df.to_csv('kg_edges.csv', index=False, encoding='utf-8-sig')
    print(f"已导出 {len(edges_df)} 条边到 kg_edges.csv")

    # 打印统计信息
    print("\n导出统计:")
    print(f"- 总节点数: {len(nodes_df)}")
    print(f"- 总边数: {len(edges_df)}")
    print("\n节点类型分布:")
    print(nodes_df['node_type'].value_counts())
    print("\n关系类型分布:")
    print(edges_df['rel_type'].value_counts())

    print("\n导出完成!")


if __name__ == "__main__":
    main()