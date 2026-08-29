import os
"""
向量检索模块
用于查询相似Knowledge节点及其一跳三元组
"""

import faiss
import pickle
import numpy as np
from py2neo import Graph
from embedding import get_embedding
from typing import List, Dict, Tuple


class VectorSearcher:
    """向量检索器类"""
    
    def __init__(self, 
                 faiss_path: str = "knowledge_embeddings.faiss",
                 metadata_path: str = "knowledge_embeddings_metadata.pkl"):
        """
        初始化向量检索器
        
        参数：
            faiss_path: FAISS索引文件路径
            metadata_path: 元数据文件路径
        """
        self.faiss_path = faiss_path
        self.metadata_path = metadata_path
        
        # 加载FAISS索引和元数据
        self._load_index()
        
        # 连接图谱数据库
        self.graph = self._connect_graph()
    
    def _load_index(self):
        """加载FAISS索引和元数据"""
        self.index = faiss.read_index(self.faiss_path)
        if getattr(self.index, "metric_type", None) != faiss.METRIC_INNER_PRODUCT:
            raise ValueError("Index is not Inner Product. Rebuild index as IndexFlatIP for cosine.")
        
        with open(self.metadata_path, "rb") as f:
            metadata = pickle.load(f)
        
        self.node_ids = metadata["node_ids"]
        self.texts = metadata["texts"]
        self.dimension = metadata["dimension"]
        
        print(f"加载FAISS索引成功，共{self.index.ntotal}个向量")
    
    def _connect_graph(self):
        """连接到Neo4j图谱数据库"""
        graph = Graph(
            uri="bolt://localhost:7687",
            user="neo4j",
            password=os.environ.get("NEO4J_PASSWORD", "")
        )
        return graph
    
    def search_similar_nodes(self, query_text: str, top_k: int = 5) -> List[Dict]:
        """
        检索与查询文本最相似的Knowledge节点
        
        参数：
            query_text: 查询文本
            top_k: 返回前k个最相似的节点
            
        返回：
            包含相似节点信息的列表，每个元素包括：
            - rank: 排名
            - node_id: 节点ID
            - text: 节点文本
            - similarity: 相似度分数
            - properties: 节点属性
        """
        # 获取查询文本的embedding
        query_embedding = get_embedding("text-embedding-3-small", query_text)
        query_vector = np.array([query_embedding], dtype='float32')
        faiss.normalize_L2(query_vector)
        
        scores, indices = self.index.search(query_vector, top_k)
        
        # 准备结果
        results = []
        for rank, idx in enumerate(indices[0]):
            if idx >= 0:  # 有效的索引
                node_id = self.node_ids[idx]
                text = self.texts[idx]
                score = float(scores[0][rank]) 
                similarity = score
                
                # 从图谱获取节点完整属性
                node_properties = self._get_node_properties(node_id)
                
                results.append({
                    "rank": rank + 1,
                    "node_id": node_id,
                    "text": text,
                    "similarity": similarity,
                    "score": score,
                    "properties": node_properties
                })
        
        return results
    
    def _get_node_properties(self, node_id: int) -> Dict:
        """
        从图谱获取节点的完整属性
        
        参数：
            node_id: 节点ID
            
        返回：
            节点属性字典
        """
        query = """
        MATCH (n:Knowledge)
        WHERE id(n) = $node_id
        RETURN n
        """
        result = self.graph.run(query, node_id=node_id)
        
        for record in result:
            node = record["n"]
            return dict(node)
        
        return {}
    
    def get_one_hop_triples(self, node_id: int) -> List[Dict]:
        """
        获取节点的一跳三元组信息
        
        参数：
            node_id: 节点ID
            
        返回：
            包含三元组信息的列表，每个元素包括：
            - direction: 方向（outgoing/incoming）
            - relationship: 关系类型
            - neighbor_id: 邻居节点ID
            - neighbor_labels: 邻居节点标签
            - neighbor_properties: 邻居节点属性
        """
        triples = []
        
        # 获取出边三元组（作为起点）
        outgoing_query = """
        MATCH (n:Knowledge)-[r]-(m)
        WHERE id(n) = $node_id
        RETURN type(r) as rel_type, id(m) as neighbor_id, labels(m) as neighbor_labels, properties(m) as neighbor_props
        """
        result = self.graph.run(outgoing_query, node_id=node_id)
        
        for record in result:
            triples.append({
                "direction": "outgoing",
                "relationship": record["rel_type"],
                "neighbor_id": record["neighbor_id"],
                "neighbor_labels": record["neighbor_labels"],
                "neighbor_properties": record["neighbor_props"]
            })
        
        # 获取入边三元组（作为终点）
        incoming_query = """
        MATCH (n:Knowledge)<-[r]-(m)
        WHERE id(n) = $node_id
        RETURN type(r) as rel_type, id(m) as neighbor_id, labels(m) as neighbor_labels, properties(m) as neighbor_props
        """
        result = self.graph.run(incoming_query, node_id=node_id)
        
        for record in result:
            triples.append({
                "direction": "incoming",
                "relationship": record["rel_type"],
                "neighbor_id": record["neighbor_id"],
                "neighbor_labels": record["neighbor_labels"],
                "neighbor_properties": record["neighbor_props"]
            })
        
        return triples
    
    def search_with_triples(self, query_text: str, top_k: int = 5) -> Dict:
        """
        检索相似节点并返回完整结果，包括节点属性和一跳三元组
        
        参数：
            query_text: 查询文本
            top_k: 返回前k个最相似的节点
            
        返回：
            包含完整检索结果的字典：
            - query: 查询文本
            - results: 节点结果列表，每个包括节点信息和三元组
        """
        similar_nodes = self.search_similar_nodes(query_text, top_k)
        
        for node_result in similar_nodes:
            node_id = node_result["node_id"]
            triples = self.get_one_hop_triples(node_id)
            node_result["triples"] = triples
            node_result["triple_count"] = len(triples)
        
        return {
            "query": query_text,
            "top_k": top_k,
            "results": similar_nodes
        }
    
    def batch_search_with_triples(self, query_texts: List[str], top_k: int = 5) -> List[Dict]:
        """
        批量检索相似节点并返回完整结果，包括节点属性和一跳三元组
        
        参数：
            query_texts: 查询文本列表
            top_k: 每个查询返回前k个最相似的节点
            
        返回：
            包含完整检索结果的字典列表，每个字典包括：
            - query: 查询文本
            - results: 节点结果列表，每个包括节点信息和三元组
        """
        batch_results = []
        
        for query_text in query_texts:
            result = self.search_with_triples(query_text, top_k)
            batch_results.append(result)
        
        return batch_results
    
    def print_search_results(self, search_result: Dict):
        """
        打印检索结果（格式化输出）
        
        参数：
            search_result: search_with_triples方法的返回结果
        """
        print("=" * 80)
        print(f"查询文本: {search_result['query']}")
        print(f"返回前 {search_result['top_k']} 个相似节点")
        print("=" * 80)
        
        for result in search_result["results"]:
            print(f"\n【排名 {result['rank']}】相似度: {result['similarity']:.4f}")
            print(f"节点ID: {result['node_id']}")
            print(f"文本内容: {result['text']}")
            
            # 打印节点属性（排除空值）
            print("\n节点属性:")
            for key, value in result['properties'].items():
                if value is not None and value != "":
                    print(f"  - {key}: {value}")
            
            # 打印三元组
            triples = result["triples"]
            print(f"\n一跳三元组 (共{len(triples)}个):")
            
            if triples:
                for i, triple in enumerate(triples, 1):
                    direction_str = "→" if triple["direction"] == "outgoing" else "←"
                    neighbor_name = triple["neighbor_properties"].get("text", 
                                    triple["neighbor_properties"].get("name", 
                                    triple["neighbor_properties"].get("name_cn", str(triple["neighbor_id"]))))
                    
                    if triple["direction"] == "outgoing":
                        print(f"  {i}. [本节点] {direction_str} {triple['relationship']} {direction_str} [{neighbor_name}]")
                    else:
                        print(f"  {i}. [{neighbor_name}] {direction_str} {triple['relationship']} {direction_str} [本节点]")
            else:
                print("  (无关联关系)")
            
            print("-" * 80)


def main():
    """主函数：演示向量检索功能"""
    
    # 初始化检索器
    searcher = VectorSearcher(faiss_path="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/graph_embedding/knowledge_embeddings.faiss",
                              metadata_path="/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/graph_embedding/knowledge_embeddings_metadata.pkl")
    
    # ========== 测试1: 单查询检索 ==========
    print("\n" + "=" * 80)
    print("测试1: 单查询检索")
    print("=" * 80 + "\n")
    
    query_text = "angina pectoris"
    
    # 执行检索
    result = searcher.search_with_triples(query_text, top_k=5)
    
    # 打印结果
    searcher.print_search_results(result)
    
    # # ========== 测试2: 批量查询检索 ==========
    # print("\n" + "=" * 80)
    # print("测试2: 批量查询检索")
    # print("=" * 80 + "\n")
    
    # # 定义多个查询文本
    # query_texts = [
    #     "anticentromere antibodies",
    #     " myocardial ischemia",
    #     "ECG abnormalities"
    # ]
    
    # print(f"执行批量检索，共 {len(query_texts)} 个查询\n")
    
    # # 执行批量检索
    # batch_results = searcher.batch_search_with_triples(query_texts, top_k=3)
    
    # # 打印每个查询的结果
    # for i, search_result in enumerate(batch_results, 1):
    #     print(f"\n{'=' * 80}")
    #     print(f"【批量查询 {i}/{len(query_texts)}】")
    #     print('=' * 80)
    #     searcher.print_search_results(search_result)
    
    # # 输出批量检索汇总信息
    # print(f"\n{'=' * 80}")
    # print("批量检索汇总")
    # print(f"=" * 80)
    # print(f"总查询数: {len(batch_results)}")
    # print(f"每个查询返回节点数: {batch_results[0]['top_k'] if batch_results else 0}")
    # for i, result in enumerate(batch_results, 1):
    #     query = result['query']
    #     result_count = len(result['results'])
    #     print(f"  查询 {i} ({query}): 返回 {result_count} 个结果")


if __name__ == "__main__":
    main()