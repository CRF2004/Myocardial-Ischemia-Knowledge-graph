import json
import os
from pathlib import Path
from py2neo import Graph, Node, Relationship
from typing import List, Dict
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraphBuilder:
    def __init__(self, neo4j_uri="bolt://localhost:7687", 
                 neo4j_user="neo4j", 
                 neo4j_password="neoneo44j"):
        """初始化知识图谱构建器"""
        try:
            self.graph = Graph(neo4j_uri, auth=(neo4j_user, neo4j_password))
            logger.info("成功连接到Neo4j数据库")
        except Exception as e:
            logger.error(f"连接Neo4j失败: {e}")
            raise
        
        # 批处理配置
        self.batch_size = 100  # 每批处理100个三元组
        self.entity_cache = {}  # 实体缓存，避免重复创建
        
    def create_indexes(self):
        """创建索引以提高查询性能"""
        try:
            # 为实体的ontology_id创建唯一约束
            self.graph.run("CREATE CONSTRAINT IF NOT EXISTS FOR (n:Knowledge) REQUIRE n.ontology_id IS UNIQUE")
            logger.info("索引创建成功")
        except Exception as e:
            logger.warning(f"创建索引时出现警告: {e}")
    
    def get_or_create_entity(self, entity_data: Dict) -> Node:
        """获取或创建实体节点（带缓存）"""
        ontology_id = entity_data.get("ontology_id")
        
        # 检查缓存
        if ontology_id in self.entity_cache:
            return self.entity_cache[ontology_id]
        
        # 查询数据库
        node = self.graph.nodes.match("Knowledge", ontology_id=ontology_id).first()
        
        if node is None:
            # 创建新节点
            node = Node(
                "Knowledge",
                text=entity_data.get("text", ""),
                ontology_id=ontology_id,
                normalized_name=entity_data.get("normalized_name", ""),
                match_method=entity_data.get("match_method", "")
            )
            self.graph.create(node)
        
        # 加入缓存
        self.entity_cache[ontology_id] = node
        return node
    
    def create_relationship_batch(self, triples: List[Dict]):
        """批量创建关系（带去重检查）"""
        relationships = []
        skipped_count = 0
        
        for triple in triples:
            try:
                # 获取或创建头实体和尾实体
                head_node = self.get_or_create_entity(triple["head"])
                tail_node = self.get_or_create_entity(triple["tail"])
                
                # 创建关系
                relation_type = triple["relation"]
                source_sentences = triple.get("source_sentences", [])
                
                # 检查关系是否已存在（避免重复创建）
                existing_rel = self.graph.match((head_node, tail_node), relation_type).first()
                if existing_rel is not None:
                    skipped_count += 1
                    logger.debug(f"跳过已存在的关系: {head_node['ontology_id']} -> {relation_type} -> {tail_node['ontology_id']}")
                    continue
                
                # 将句子列表转换为字符串（如果句子太长，可以截断）
                source_text = " | ".join(source_sentences[:3])  # 最多保留3个句子
                if len(source_text) > 5000:  # 限制属性长度
                    source_text = source_text[:5000] + "..."
                
                rel = Relationship(
                    head_node,
                    relation_type,
                    tail_node,
                    source_sentences=source_text
                )
                relationships.append(rel)
                
            except Exception as e:
                logger.error(f"处理三元组时出错: {e}")
                logger.error(f"问题三元组: {triple}")
                continue
        
        # 批量提交关系
        if relationships:
            try:
                tx = self.graph.begin()
                for rel in relationships:
                    tx.create(rel)
                tx.commit()
                logger.info(f"成功创建 {len(relationships)} 个新关系，跳过 {skipped_count} 个已存在的关系")
            except Exception as e:
                logger.error(f"批量提交关系时出错: {e}")
    
    def process_json_file(self, file_path: str):
        """处理单个JSON文件"""
        logger.info(f"开始处理文件: {file_path}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                triples = json.load(f)
            
            logger.info(f"文件包含 {len(triples)} 个三元组")
            
            # 分批处理
            for i in range(0, len(triples), self.batch_size):
                batch = triples[i:i + self.batch_size]
                self.create_relationship_batch(batch)
                logger.info(f"已处理 {min(i + self.batch_size, len(triples))}/{len(triples)} 个三元组")
            
            logger.info(f"文件处理完成: {file_path}")
            
        except Exception as e:
            logger.error(f"处理文件时出错 {file_path}: {e}")
    
    def process_directory(self, directory: str):
        """处理目录下所有符合条件的JSON文件"""
        output_dir = Path(directory)
        
        if not output_dir.exists():
            logger.error(f"目录不存在: {directory}")
            return
        
        # 查找所有符合命名规则的文件
        json_files = sorted(
            output_dir.glob("normalized_triples_chapter_*.json"),
            key=lambda x: x.name
        )
        
        if not json_files:
            logger.warning(f"在 {directory} 目录下未找到符合命名规则的JSON文件")
            return
        
        logger.info(f"找到 {len(json_files)} 个JSON文件")
        
        # 创建索引
        self.create_indexes()
        
        # 逐个处理文件
        for idx, file_path in enumerate(json_files, 1):
            logger.info(f"处理进度: {idx}/{len(json_files)}")
            self.process_json_file(str(file_path))
            
            # 定期清理缓存，避免内存溢出
            if idx % 5 == 0:
                logger.info("清理实体缓存...")
                self.entity_cache.clear()
        
        logger.info("所有文件处理完成!")
        logger.info(f"图谱统计信息:")
        self.print_statistics()
    
    def print_statistics(self):
        """打印知识图谱统计信息"""
        try:
            node_count = self.graph.run("MATCH (n:Knowledge) RETURN count(n) as count").data()[0]['count']
            rel_count = self.graph.run("MATCH ()-[r]->() RETURN count(r) as count").data()[0]['count']
            
            logger.info(f"实体节点数: {node_count}")
            logger.info(f"关系数: {rel_count}")
        except Exception as e:
            logger.error(f"获取统计信息时出错: {e}")


def main():
    """主函数"""
    # 配置参数（请根据实际情况修改）
    NEO4J_URI = "bolt://localhost:7687"  # Neo4j地址
    NEO4J_USER = "neo4j"  # 用户名
    NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "")  # 密码
    DATA_DIR = "SRP心肌缺血知识图谱/output"  # 数据目录
    
    # 创建知识图谱构建器
    builder = KnowledgeGraphBuilder(
        neo4j_uri=NEO4J_URI,
        neo4j_user=NEO4J_USER,
        neo4j_password=NEO4J_PASSWORD
    )
    
    # 处理数据目录
    builder.process_directory(DATA_DIR)


if __name__ == "__main__":
    main()