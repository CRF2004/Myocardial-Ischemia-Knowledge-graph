from py2neo import Graph
import re

# 连接到Neo4j数据库
# 请根据实际情况修改连接参数
graph = Graph("bolt://localhost:7687", auth=("neo4j", "neoneo44j"))

# 定义简写代码的模式
ABBREVIATION_CODES = ['D', 'SS', 'DT', 'PC', 'TI']

def is_abbreviation_code(ontology_id):
    """
    判断ontology_id是否为简写形式
    
    Args:
        ontology_id: 节点的ontology_id属性值
        
    Returns:
        True if 是简写形式, False otherwise
    """
    if not ontology_id:
        return False
    
    # 检查是否以任何简写代码开头
    for abbr in ['SS', 'DT', 'PC', 'TI', 'D']:  # SS要在D前面检查
        if ontology_id.startswith(abbr):
            return True
    
    return False

def link_knowledge_to_ontology():
    """
    为简写形式的Knowledge节点创建到对应Ontology节点的关系
    """
    # 获取所有简写形式的Knowledge节点
    # 使用正则表达式匹配简写模式
    query = """
    MATCH (k:Knowledge)
    WHERE k.ontology_id =~ '^(D|SS|DT|PC|TI).*'
    RETURN k, k.ontology_id as ontology_id
    """
    
    results = graph.run(query).data()
    print(f"找到 {len(results)} 个简写形式的Knowledge节点")
    
    success_count = 0
    not_found_count = 0
    already_linked_count = 0
    error_count = 0
    
    for record in results:
        k_node = record['k']
        ontology_id = record['ontology_id']
        
        try:
            # 检查是否已经存在关系
            check_query = """
            MATCH (k:Knowledge)-[r:HAS_ONTOLOGY]->(o:Ontology)
            WHERE id(k) = $k_id AND o.ontology_id = $ontology_id
            RETURN count(r) as rel_count
            """
            check_result = graph.run(check_query, 
                                    k_id=k_node.identity, 
                                    ontology_id=ontology_id).data()
            
            if check_result[0]['rel_count'] > 0:
                print(f"- Knowledge节点 (ontology_id: {ontology_id}) 已存在关系，跳过")
                already_linked_count += 1
                continue
            
            # 查找对应的Ontology节点
            find_ontology_query = """
            MATCH (o:Ontology)
            WHERE o.ontology_id = $ontology_id
            RETURN o
            """
            ontology_results = graph.run(find_ontology_query, 
                                        ontology_id=ontology_id).data()
            
            if not ontology_results:
                print(f"✗ 未找到对应的Ontology节点: {ontology_id}")
                not_found_count += 1
                continue
            
            # 创建关系
            create_rel_query = """
            MATCH (k:Knowledge), (o:Ontology)
            WHERE id(k) = $k_id AND o.ontology_id = $ontology_id
            MERGE (k)-[r:HAS_ONTOLOGY]->(o)
            RETURN r
            """
            graph.run(create_rel_query, 
                     k_id=k_node.identity, 
                     ontology_id=ontology_id)
            
            print(f"✓ 成功创建关系: Knowledge (ontology_id: {ontology_id}) -> Ontology")
            success_count += 1
            
        except Exception as e:
            print(f"✗ 处理节点 (ontology_id: {ontology_id}) 时出错: {e}")
            error_count += 1
    
    # 输出统计信息
    print("\n" + "="*60)
    print(f"关系创建完成！")
    print(f"成功创建: {success_count} 个关系")
    print(f"已存在关系: {already_linked_count} 个")
    print(f"未找到对应Ontology: {not_found_count} 个")
    print(f"错误: {error_count} 个")
    print("="*60)
    
    # 验证结果
    verify_query = """
    MATCH (k:Knowledge)-[r:HAS_ONTOLOGY]->(o:Ontology)
    RETURN count(r) as total_relations
    """
    verify_result = graph.run(verify_query).data()
    print(f"\n当前数据库中 HAS_ONTOLOGY 关系总数: {verify_result[0]['total_relations']}")

def preview_matches():
    """
    预览将要创建的匹配关系（可选，用于验证）
    """
    query = """
    MATCH (k:Knowledge), (o:Ontology)
    WHERE k.ontology_id =~ '^(D|SS|DT|PC|TI).*'
      AND k.ontology_id = o.ontology_id
    RETURN k.ontology_id as ontology_id, 
           labels(k) as k_labels, 
           labels(o) as o_labels
    LIMIT 10
    """
    
    results = graph.run(query).data()
    print("预览前10个匹配的节点对：")
    print("-" * 60)
    for r in results:
        print(f"Ontology_id: {r['ontology_id']}")
        print(f"  Knowledge标签: {r['k_labels']}")
        print(f"  Ontology标签: {r['o_labels']}")
        print()

if __name__ == "__main__":
    try:
        # 可选：先预览匹配情况
        print("=== 预览匹配情况 ===\n")
        preview_matches()
        
        # 确认后执行
        print("\n=== 开始创建关系 ===\n")
        user_input = input("是否继续创建关系? (yes/no): ").strip().lower()
        
        if user_input in ['yes', 'y']:
            link_knowledge_to_ontology()
        else:
            print("操作已取消")
            
    except Exception as e:
        print(f"程序执行出错: {e}")
        import traceback
        traceback.print_exc()