import json
import os
from py2neo import Graph, Node, Relationship
from pathlib import Path

class OntologyUploader:
    def __init__(self, uri="bolt://localhost:7687", auth=("neo4j", "password")):
        """初始化Neo4j连接"""
        self.graph = Graph(uri, auth=auth)
        self.nodes_map = {}  # 用于存储 ontology_id -> Node 的映射
        self.orphan_nodes = []  # 记录找不到父节点的实体
        
    def extract_category_from_filename(self, filename):
        """从文件名提取分类标签"""
        basename = os.path.basename(filename)
        # 移除 "SRP心肌缺血知识图谱/ontology/" 前缀和 "_mapped.json" 后缀
        category = basename.replace("_mapped.json", "")
        # 转换为驼峰命名: diagnostic_test -> DiagnosticTest
        parts = category.split("_")
        category_label = "".join([p.capitalize() for p in parts])
        return category_label
    
    def parse_umls_mapping(self, umls_data):
        """解析UMLS映射数据,提取为独立属性"""
        if umls_data is None:
            return {
                "umls_cui": None,
                "umls_canonical_name": None,
                "umls_score": None,
                "umls_types": None,
                "umls_definition": None,
                "umls_sources": None,
                "umls_all_sources": None,
                "umls_mapping_json": None
            }
        
        return {
            "umls_cui": umls_data.get("cui"),
            "umls_canonical_name": umls_data.get("canonical_name"),
            "umls_score": umls_data.get("score"),
            "umls_types": json.dumps(umls_data.get("types")) if umls_data.get("types") else None,
            "umls_definition": umls_data.get("definition"),
            "umls_sources": json.dumps(umls_data.get("sources")) if umls_data.get("sources") else None,
            "umls_all_sources": json.dumps(umls_data.get("all_sources")) if umls_data.get("all_sources") else None,
            "umls_mapping_json": json.dumps(umls_data)  # 完整的UMLS数据作为JSON字符串
        }
    
    def create_node(self, entity_data, category_label):
        """创建节点"""
        # 提取UMLS属性
        umls_props = self.parse_umls_mapping(entity_data.get("umls_mapping"))
        
        # 构建节点属性
        props = {
            "ontology_id": entity_data.get("ontology_id"),
            "level": entity_data.get("level"),
            "parent_id": entity_data.get("parent_id"),
            "definition": entity_data.get("definition"),
            "type": entity_data.get("type"),
            "name_cn": entity_data.get("name_cn"),
            "name_en": entity_data.get("name_en"),
            "id": entity_data.get("id"),
            "label": entity_data.get("label"),
        }
        
        # 合并UMLS属性
        props.update(umls_props)
        
        # 创建节点,带有两个标签: Ontology 和 分类标签
        node = Node("Ontology", category_label, **props)
        return node
    
    def load_json_file(self, filepath):
        """加载JSON文件"""
        with open(filepath, 'r', encoding='utf-8') as f:
            return json.load(f)
    
    def upload_entities(self, json_files):
        """上传所有实体到Neo4j"""
        print("=" * 60)
        print("开始上传实体到Neo4j...")
        print("=" * 60)
        
        # 第一阶段：创建所有节点
        all_entities = []
        for json_file in json_files:
            print(f"\n处理文件: {json_file}")
            category_label = self.extract_category_from_filename(json_file)
            print(f"分类标签: {category_label}")
            
            entities = self.load_json_file(json_file)
            print(f"实体数量: {len(entities)}")
            
            for entity in entities:
                entity['_category_label'] = category_label
                all_entities.append(entity)
        
        print(f"\n总计实体数量: {len(all_entities)}")
        print("\n第一阶段: 创建所有节点...")
        
        tx = self.graph.begin()
        for i, entity in enumerate(all_entities, 1):
            node = self.create_node(entity, entity['_category_label'])
            tx.create(node)
            self.nodes_map[entity['ontology_id']] = node
            
            if i % 100 == 0:
                print(f"已创建 {i}/{len(all_entities)} 个节点...")
        
        tx.commit()
        print(f"✓ 所有 {len(all_entities)} 个节点创建完成!")
        
        # 第二阶段：创建关系
        print("\n第二阶段: 创建层级关系...")
        self.create_relationships(all_entities)
        
        # 输出统计信息
        self.print_statistics()
    
    def create_relationships(self, all_entities):
        """创建父子关系"""
        tx = self.graph.begin()
        relationship_count = 0
        
        for entity in all_entities:
            parent_id = entity.get('parent_id')
            
            # 跳过根节点(parent_id为null)
            if parent_id is None:
                continue
            
            child_id = entity['ontology_id']
            
            # 检查父节点是否存在
            if parent_id not in self.nodes_map:
                self.orphan_nodes.append({
                    'ontology_id': child_id,
                    'parent_id': parent_id,
                    'name_cn': entity.get('name_cn'),
                    'category': entity['_category_label']
                })
                print(f"⚠ 警告: 实体 {child_id} ({entity.get('name_cn')}) 的父节点 {parent_id} 不存在")
                continue
            
            # 创建关系: parent -[HAS_CHILD]-> child
            parent_node = self.nodes_map[parent_id]
            child_node = self.nodes_map[child_id]
            rel = Relationship(parent_node, "HAS_CHILD", child_node)
            tx.create(rel)
            relationship_count += 1
        
        tx.commit()
        print(f"✓ 创建了 {relationship_count} 个 HAS_CHILD 关系")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "=" * 60)
        print("上传统计信息")
        print("=" * 60)
        print(f"总节点数: {len(self.nodes_map)}")
        print(f"孤儿节点数(找不到父节点): {len(self.orphan_nodes)}")
        
        if self.orphan_nodes:
            print("\n孤儿节点详情:")
            print("-" * 60)
            for orphan in self.orphan_nodes:
                print(f"  ID: {orphan['ontology_id']}")
                print(f"  名称: {orphan['name_cn']}")
                print(f"  缺失的父ID: {orphan['parent_id']}")
                print(f"  分类: {orphan['category']}")
                print("-" * 60)
        
        print("\n✓ 上传完成!")


def main():
    """主函数"""
    # 定义JSON文件路径
    json_files = [
        "SRP心肌缺血知识图谱/ontology/diagnostic_test_mapped.json",
        "SRP心肌缺血知识图谱/ontology/disease_mapped.json",
        "SRP心肌缺血知识图谱/ontology/patient_characteristics_mapped.json",
        "SRP心肌缺血知识图谱/ontology/symptoms_mapped.json",
        "SRP心肌缺血知识图谱/ontology/treatment_intervention_mapped.json"
    ]
    
    # 创建上传器实例
    # 请根据实际情况修改用户名和密码
    uploader = OntologyUploader(
        uri="bolt://localhost:7687",
        auth=("neo4j", "neoneo44j")  # 修改为您的实际密码
    )
    
    # 执行上传
    uploader.upload_entities(json_files)


if __name__ == "__main__":
    main()