from py2neo import Graph
import re

# 连接到Neo4j数据库
# 请根据实际情况修改连接参数
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neoneo44j"))

# 定义映射规则
LABEL_MAPPING = {
    # 完整形式映射（小写匹配）
    'diseases': 'Disease',
    'disease': 'Disease',
    'symptoms': 'Symptoms',
    'symptom': 'Symptoms',
    'diagnostic_test': 'DiagnosticTest',
    'diagnostictest': 'DiagnosticTest',
    'examinations': 'DiagnosticTest',
    'examination': 'DiagnosticTest',
    'patient_characteristics': 'PatientCharacteristics',
    'patientcharacteristics': 'PatientCharacteristics',
    'treatments': 'TreatmentIntervention',
    'treatment': 'TreatmentIntervention',
    'treatmentintervention': 'TreatmentIntervention',
    'intervention': 'TreatmentIntervention',
    
    # 缩写形式映射
    'D': 'Disease',
    'SS': 'Symptoms',
    'DT': 'DiagnosticTest',
    'PC': 'PatientCharacteristics',
    'TI': 'TreatmentIntervention'
}

def parse_ontology_id(ontology_id):
    """
    解析ontology_id并返回对应的标签
    
    Args:
        ontology_id: 节点的ontology_id属性值
        
    Returns:
        对应的标签名称，如果无法匹配则返回None
    """
    if not ontology_id:
        return None
    
    # 模式1: TEMP_{category}_{number}
    temp_pattern = r'TEMP_([a-zA-Z_]+)_\d+'
    match = re.match(temp_pattern, ontology_id)
    if match:
        category = match.group(1).lower()
        return LABEL_MAPPING.get(category)
    
    # 模式2: 缩写代码开头 (D, SS, DT, PC, TI)
    # 匹配以这些字母开头的编码
    for abbr in ['SS', 'DT', 'PC', 'TI', 'D']:  # SS要在D前面匹配
        if ontology_id.startswith(abbr):
            return LABEL_MAPPING.get(abbr)
    
    return None

def add_labels_to_knowledge_nodes():
    """
    为所有Knowledge节点根据ontology_id添加相应的标签
    """
    # 获取所有标签为Knowledge的节点
    query = "MATCH (n:Knowledge) RETURN n"
    results = graph.run(query).data()
    
    print(f"找到 {len(results)} 个Knowledge节点")
    
    updated_count = 0
    skipped_count = 0
    error_count = 0
    
    for record in results:
        node = record['n']
        ontology_id = node.get('ontology_id')
        
        if not ontology_id:
            print(f"节点 {node.identity} 没有ontology_id属性，跳过")
            skipped_count += 1
            continue
        
        # 解析ontology_id获取标签
        new_label = parse_ontology_id(ontology_id)
        
        if new_label:
            try:
                # 检查节点是否已有该标签
                existing_labels = list(node.labels)
                if new_label not in existing_labels:
                    # 添加新标签
                    update_query = f"""
                    MATCH (n:Knowledge)
                    WHERE id(n) = $node_id
                    SET n:{new_label}
                    """
                    graph.run(update_query, node_id=node.identity)
                    print(f"✓ 节点 {node.identity} (ontology_id: {ontology_id}) 添加标签: {new_label}")
                    updated_count += 1
                else:
                    print(f"- 节点 {node.identity} (ontology_id: {ontology_id}) 已有标签: {new_label}")
                    skipped_count += 1
            except Exception as e:
                print(f"✗ 更新节点 {node.identity} 时出错: {e}")
                error_count += 1
        else:
            print(f"? 无法匹配节点 {node.identity} 的ontology_id: {ontology_id}")
            skipped_count += 1
    
    # 输出统计信息
    print("\n" + "="*50)
    print(f"处理完成！")
    print(f"成功更新: {updated_count} 个节点")
    print(f"跳过: {skipped_count} 个节点")
    print(f"错误: {error_count} 个节点")
    print("="*50)

if __name__ == "__main__":
    try:
        add_labels_to_knowledge_nodes()
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()