#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
从JSON文件导出知识图谱数据到CSV文件
导出指定JSON文件中的节点和相关边，输出格式与export_kg.py一致
图谱结构: FindingLabel -> ECGFinding -> Disease -> DiseaseLabel
"""

import pandas as pd
import json
from collections import defaultdict


def load_json_file(filepath):
    """
    加载JSON文件
    :param filepath: JSON文件路径
    :return: JSON数据
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return json.load(f)


def extract_unique_entities(chapter_data, disease_data, ecgfinding_data):
    """
    从所有数据中提取唯一的实体作为节点
    :param chapter_data: chapter_extraction_results.json数据
    :param disease_data: ptbxl_scp_to_book_disease.json数据
    :param ecgfinding_data: ptbxl_scp_to_book_ecgfinding.json数据
    :return: 节点数据列表
    """
    nodes = {}  # name -> node info
    node_id_counter = 0

    # 1. 从chapter_data中提取ECGFinding和Disease节点
    if isinstance(chapter_data, list):
        for item in chapter_data:
            ecg_finding = item.get('ecg_finding')
            knowledge_entity = item.get('knowledge_entity')

            # ECGFinding节点
            if ecg_finding and ecg_finding not in nodes:
                node_id_counter += 1
                nodes[ecg_finding] = {
                    'node_id': node_id_counter,
                    'node_type': 'ECGFinding',
                    'name': ecg_finding,
                    'source': 'book_chapter',
                    'meta': {
                        'original_text': ecg_finding,
                        'data_source': 'chapter_extraction_results.json'
                    }
                }

            # Disease节点
            if knowledge_entity and knowledge_entity not in nodes:
                node_id_counter += 1
                nodes[knowledge_entity] = {
                    'node_id': node_id_counter,
                    'node_type': 'Disease',
                    'name': knowledge_entity,
                    'source': 'book_chapter',
                    'meta': {
                        'original_text': knowledge_entity,
                        'data_source': 'chapter_extraction_results.json'
                    }
                }

    # 2. 从disease_data中提取FindingLabel和DiseaseLabel节点
    if isinstance(disease_data, dict) and 'items' in disease_data:
        for item in disease_data['items']:
            scp_code = item.get('scp_code')
            label_text = item.get('label_text')

            # FindingLabel节点（SCP code）
            if scp_code and scp_code not in nodes:
                node_id_counter += 1
                nodes[scp_code] = {
                    'node_id': node_id_counter,
                    'node_type': 'FindingLabel',
                    'name': scp_code,
                    'source': 'ptbxl_scp',
                    'meta': {
                        'scp_code': scp_code,
                        'label_text': label_text,
                        'statement_category': item.get('statement_category'),
                        'data_source': 'ptbxl_scp_to_book_disease.json'
                    }
                }

            # DiseaseLabel节点（SCP label_text）
            if label_text and label_text not in nodes:
                node_id_counter += 1
                nodes[label_text] = {
                    'node_id': node_id_counter,
                    'node_type': 'DiseaseLabel',
                    'name': label_text,
                    'source': 'ptbxl_scp',
                    'meta': {
                        'scp_code': scp_code,
                        'label_text': label_text,
                        'statement_category': item.get('statement_category'),
                        'data_source': 'ptbxl_scp_to_book_disease.json'
                    }
                }

            # 从mappings中提取Disease节点（book中的disease实体）
            if 'mappings' in item:
                for mapping in item['mappings']:
                    node_text = mapping.get('node_text')
                    if node_text and node_text not in nodes:
                        node_id_counter += 1
                        nodes[node_text] = {
                            'node_id': node_id_counter,
                            'node_type': 'Disease',
                            'name': node_text,
                            'source': 'book_disease',
                            'meta': {
                                'original_text': node_text,
                                'data_source': 'ptbxl_scp_to_book_disease.json'
                            }
                        }

    # 3. 从ecgfinding_data中提取FindingLabel和ECGFinding节点
    if isinstance(ecgfinding_data, dict) and 'items' in ecgfinding_data:
        for item in ecgfinding_data['items']:
            scp_code = item.get('scp_code')
            label_text = item.get('label_text')

            # FindingLabel节点（SCP code）
            if scp_code and scp_code not in nodes:
                node_id_counter += 1
                nodes[scp_code] = {
                    'node_id': node_id_counter,
                    'node_type': 'FindingLabel',
                    'name': scp_code,
                    'source': 'ptbxl_scp',
                    'meta': {
                        'scp_code': scp_code,
                        'label_text': label_text,
                        'statement_category': item.get('statement_category'),
                        'data_source': 'ptbxl_scp_to_book_ecgfinding.json'
                    }
                }

            # ECGFinding节点（SCP label_text）
            if label_text and label_text not in nodes:
                node_id_counter += 1
                nodes[label_text] = {
                    'node_id': node_id_counter,
                    'node_type': 'ECGFinding',
                    'name': label_text,
                    'source': 'ptbxl_scp',
                    'meta': {
                        'scp_code': scp_code,
                        'label_text': label_text,
                        'statement_category': item.get('statement_category'),
                        'data_source': 'ptbxl_scp_to_book_ecgfinding.json'
                    }
                }

            # 从mappings中提取ECGFinding节点（book中的ecg_finding实体）
            if 'mappings' in item:
                for mapping in item['mappings']:
                    node_text = mapping.get('node_text')
                    if node_text and node_text not in nodes:
                        node_id_counter += 1
                        nodes[node_text] = {
                            'node_id': node_id_counter,
                            'node_type': 'ECGFinding',
                            'name': node_text,
                            'source': 'book_ecgfinding',
                            'meta': {
                                'original_text': node_text,
                                'data_source': 'ptbxl_scp_to_book_ecgfinding.json'
                            }
                        }

    return list(nodes.values())


def extract_edges(chapter_data, disease_data, ecgfinding_data, nodes_dict):
    """
    从所有数据中提取边
    图谱结构: FindingLabel -> ECGFinding -> Disease -> DiseaseLabel

    :param chapter_data: chapter_extraction_results.json数据
    :param disease_data: ptbxl_scp_to_book_disease.json数据
    :param ecgfinding_data: ptbxl_scp_to_book_ecgfinding.json数据
    :param nodes_dict: 节点字典（name到node_id的映射）
    :return: 边数据列表
    """
    edges = []
    edge_id_counter = 0

    # 创建名称到node_id和node_type的映射
    name_to_info = {node['name']: {'id': node['node_id'], 'type': node['node_type']} for node in nodes_dict}

    # 1. 从chapter_data中提取边: ECGFinding -> Disease
    if isinstance(chapter_data, list):
        for item in chapter_data:
            ecg_finding = item.get('ecg_finding')
            knowledge_entity = item.get('knowledge_entity')
            relation = item.get('relation')
            evidence_text = item.get('evidence_text', '')
            source = item.get('source', '')

            if ecg_finding in name_to_info and knowledge_entity in name_to_info:
                src_type = name_to_info[ecg_finding]['type']
                tgt_type = name_to_info[knowledge_entity]['type']

                # 确保是 ECGFinding -> Disease
                if src_type == 'ECGFinding' and tgt_type == 'Disease':
                    edge_id_counter += 1
                    edges.append({
                        'src_id': name_to_info[ecg_finding]['id'],
                        'rel_type': relation,
                        'tgt_id': name_to_info[knowledge_entity]['id'],
                        'confidence': 1.0,
                        'meta': {
                            'evidence_text': evidence_text,
                            'source': source,
                            'section_title': item.get('section_title'),
                            'paragraph_index': item.get('paragraph_index'),
                            'chunk_index': item.get('chunk_index'),
                            'data_source': 'chapter_extraction_results.json'
                        }
                    })

    # 2. 从disease_data中提取边
    if isinstance(disease_data, dict) and 'items' in disease_data:
        for item in disease_data['items']:
            scp_code = item.get('scp_code')
            label_text = item.get('label_text')

            # FindingLabel -> DiseaseLabel边（通过HAS_LABEL关系）
            if scp_code in name_to_info and label_text in name_to_info:
                src_type = name_to_info[scp_code]['type']
                tgt_type = name_to_info[label_text]['type']

                if src_type == 'FindingLabel' and tgt_type == 'DiseaseLabel':
                    edge_id_counter += 1
                    edges.append({
                        'src_id': name_to_info[scp_code]['id'],
                        'rel_type': 'HAS_LABEL',
                        'tgt_id': name_to_info[label_text]['id'],
                        'confidence': 1.0,
                        'meta': {
                            'scp_code': scp_code,
                            'label_text': label_text,
                            'statement_category': item.get('statement_category'),
                            'data_source': 'ptbxl_scp_to_book_disease.json'
                        }
                    })

            # FindingLabel -> Disease边（来自mappings）
            if 'mappings' in item:
                for mapping in item['mappings']:
                    node_text = mapping.get('node_text')
                    relation_type = mapping.get('relation_type', 'MAPS_TO')
                    similarity = mapping.get('similarity', 1.0)
                    reason = mapping.get('reason', '')
                    rank = mapping.get('rank', 0)

                    if scp_code in name_to_info and node_text in name_to_info:
                        src_type = name_to_info[scp_code]['type']
                        tgt_type = name_to_info[node_text]['type']

                        # FindingLabel -> Disease
                        if src_type == 'FindingLabel' and tgt_type == 'Disease':
                            edge_id_counter += 1
                            edges.append({
                                'src_id': name_to_info[scp_code]['id'],
                                'rel_type': relation_type,
                                'tgt_id': name_to_info[node_text]['id'],
                                'confidence': similarity,
                                'meta': {
                                    'scp_code': scp_code,
                                    'node_text': node_text,
                                    'reason': reason,
                                    'rank': rank,
                                    'similarity': similarity,
                                    'edge_type': mapping.get('edge_type'),
                                    'data_source': 'ptbxl_scp_to_book_disease.json'
                                }
                            })

    # 3. 从ecgfinding_data中提取边
    if isinstance(ecgfinding_data, dict) and 'items' in ecgfinding_data:
        for item in ecgfinding_data['items']:
            scp_code = item.get('scp_code')
            label_text = item.get('label_text')

            # FindingLabel -> ECGFinding边（通过HAS_LABEL关系）
            if scp_code in name_to_info and label_text in name_to_info:
                src_type = name_to_info[scp_code]['type']
                tgt_type = name_to_info[label_text]['type']

                if src_type == 'FindingLabel' and tgt_type == 'ECGFinding':
                    edge_id_counter += 1
                    edges.append({
                        'src_id': name_to_info[scp_code]['id'],
                        'rel_type': 'HAS_LABEL',
                        'tgt_id': name_to_info[label_text]['id'],
                        'confidence': 1.0,
                        'meta': {
                            'scp_code': scp_code,
                            'label_text': label_text,
                            'statement_category': item.get('statement_category'),
                            'data_source': 'ptbxl_scp_to_book_ecgfinding.json'
                        }
                    })

            # FindingLabel -> ECGFinding边（来自mappings）
            if 'mappings' in item:
                for mapping in item['mappings']:
                    node_text = mapping.get('node_text')
                    relation_type = mapping.get('relation_type', 'MAPS_TO')
                    similarity = mapping.get('similarity', 1.0)
                    reason = mapping.get('reason', '')
                    rank = mapping.get('rank', 0)

                    if scp_code in name_to_info and node_text in name_to_info:
                        src_type = name_to_info[scp_code]['type']
                        tgt_type = name_to_info[node_text]['type']

                        # FindingLabel -> ECGFinding
                        if src_type == 'FindingLabel' and tgt_type == 'ECGFinding':
                            edge_id_counter += 1
                            edges.append({
                                'src_id': name_to_info[scp_code]['id'],
                                'rel_type': relation_type,
                                'tgt_id': name_to_info[node_text]['id'],
                                'confidence': similarity,
                                'meta': {
                                    'scp_code': scp_code,
                                    'node_text': node_text,
                                    'reason': reason,
                                    'rank': rank,
                                    'similarity': similarity,
                                    'edge_type': mapping.get('edge_type'),
                                    'data_source': 'ptbxl_scp_to_book_ecgfinding.json'
                                }
                            })

    return edges


def main():
    """主函数"""
    print("开始从JSON文件导出知识图谱数据...")

    # 定义输入文件路径
    chapter_file = '../build_ecg_book_graph/ecgfinding_alignment/chapter_extraction_results.json'
    disease_file = '../build_ecg_book_graph/disease_alignment/ptbxl_scp_to_book_disease.json'
    ecgfinding_file = '../build_ecg_book_graph/ecgfinding_alignment/ptbxl_scp_to_book_ecgfinding.json'

    # 加载数据
    print(f"加载数据文件...")
    print(f"  - {chapter_file}")
    chapter_data = load_json_file(chapter_file)
    print(f"  - {disease_file}")
    disease_data = load_json_file(disease_file)
    print(f"  - {ecgfinding_file}")
    ecgfinding_data = load_json_file(ecgfinding_file)

    # 提取节点
    print("提取节点数据...")
    nodes = extract_unique_entities(chapter_data, disease_data, ecgfinding_data)
    nodes_dict = {node['name']: node for node in nodes}

    # 准备节点DataFrame
    nodes_data = []
    for node in nodes:
        meta = json.dumps(node['meta'], ensure_ascii=False)
        nodes_data.append({
            'node_id': node['node_id'],
            'node_type': node['node_type'],
            'name': node['name'],
            'source': node['source'],
            'meta': meta
        })

    nodes_df = pd.DataFrame(nodes_data)
    nodes_df.to_csv('nodes.csv', index=False, encoding='utf-8-sig')
    print(f"已导出 {len(nodes_df)} 个节点到 nodes.csv")

    # 提取边
    print("提取边数据...")
    edges = extract_edges(chapter_data, disease_data, ecgfinding_data, nodes_dict)

    # 准备边DataFrame
    edges_data = []
    for edge in edges:
        meta = json.dumps(edge['meta'], ensure_ascii=False)
        edges_data.append({
            'src_id': edge['src_id'],
            'rel_type': edge['rel_type'],
            'tgt_id': edge['tgt_id'],
            'confidence': edge['confidence'],
            'meta': meta
        })

    edges_df = pd.DataFrame(edges_data)
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