"""
对图谱所有节点进行embedding编码
存储在FAISS数据库中以支持后面的图谱rag test
"""

import os
from py2neo import Graph
import numpy as np
import faiss
import pickle
from embedding import get_embedding


BASE_URL = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/graph_embedding/"

def connect_to_graph():
    """
    连接到Neo4j图谱数据库
    
    返回：
        Graph: py2neo的Graph对象
    """
    graph = Graph(
        uri="bolt://localhost:7687",
        user="neo4j",
        password=os.environ.get("NEO4J_PASSWORD", "")
    )
    return graph


def get_knowledge_nodes(graph):
    """
    获取所有节点（包括Ontology层）
    
    参数：
        graph: py2neo的Graph对象
    
    返回：
        list: 包含节点数据的列表，每个元素是包含节点信息的字典
    """
    # 查询所有节点
    query = """
    MATCH (n)
    RETURN n
    """
    
    result = graph.run(query)
    knowledge_nodes = []
    
    for record in result:
        node = record["n"]
        node_data = {
            "id": node.identity,
            "labels": list(node.labels),
            "properties": dict(node)
        }
        knowledge_nodes.append(node_data)
    
    return knowledge_nodes


def prepare_text_for_embedding(node_data):
    """
    为节点准备用于embedding的文本
    
    参数：
        node_data (dict): 节点数据字典
    
    返回：
        str: 用于embedding的文本，如果无法获取文本则返回 None
    """
    props = node_data["properties"]
    labels = node_data["labels"]
    
    # 初始化 text 为 None
    text = None
    
    if "Knowledge" in labels:
        text = props.get("normalized_name", props.get("name", ""))
    elif "ECGFinding" in labels:
        text = props.get("name", "")
    elif "Ontology" in labels:
        text = props.get("name_en", "")
    elif "FindingLabel" in labels:
        text = props.get("name", "")
    elif "DiseaseLabel" in labels:
        text = props.get("label", "")
    else:
        # 对于未知类型的节点，尝试获取常见的 name 字段
        text = props.get("name", "") or props.get("normalized_name", "") or props.get("name_en", "")
    
    # 确保返回的值是字符串或 None
    if text is not None and isinstance(text, str):
        text = text.strip()
        # 返回空字符串而不是 None，方便后续判断
        return text
    else:
        return None


def embed_knowledge_nodes(graph):
    """
    对所有节点进行embedding并存储为FAISS文件
    支持增量更新：检查已有embedding，只为新节点生成embedding
    
    参数：
        graph: py2neo的Graph对象
    """
    print("开始获取所有节点...")
    knowledge_nodes = get_knowledge_nodes(graph)
    print(f"共找到 {len(knowledge_nodes)} 个节点")
    
    if len(knowledge_nodes) == 0:
        print("没有找到节点，程序退出")
        return
    
    # 尝试加载已有的metadata
    metadata_path = "knowledge_embeddings_metadata.pkl"
    faiss_index_path = "knowledge_embeddings.faiss"
    
    existing_node_ids_list = []
    existing_node_ids_set  = set()
    existing_texts_list    = []
    dimension = None
    
    if os.path.exists(BASE_URL + metadata_path):
        print(f"发现已有的metadata文件: {metadata_path}")
        with open(metadata_path, "rb") as f:
            existing_metadata = pickle.load(f)
            existing_node_ids_list = existing_metadata["node_ids"] 
            existing_node_ids_set  = set(existing_node_ids_list)
            existing_texts_list    = existing_metadata["texts"]   
            dimension = existing_metadata["dimension"]
            
        print(f"已有 {len(existing_node_ids_list)} 个节点的embedding")
    else:
        print("未发现已有的metadata文件，将创建新的")
    
    # 准备新节点的文本
    print("准备文本数据...")
    new_texts = []
    new_node_ids = []
    
    for node_data in knowledge_nodes:
        node_id = node_data["id"]
        text = prepare_text_for_embedding(node_data)
        
        # 只处理有文本内容且不在已有列表中的节点
        if text and node_id not in existing_node_ids_set:
            new_texts.append(text)
            new_node_ids.append(node_id)
    
    print(f"找到 {len(new_texts)} 个新节点需要生成embedding")
    
    if len(new_texts) == 0:
        print("没有新节点需要处理，程序退出")
        return
    
    # 获取新节点的embedding（分批处理以避免单次请求数量过多）
    print("开始生成新节点的embeddings...")
    
    # 定义批次大小
    batch_size = 100
    total_batches = (len(new_texts) + batch_size - 1) // batch_size
    
    new_embeddings = []
    
    for batch_idx in range(total_batches):
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(new_texts))
        batch_texts = new_texts[start_idx:end_idx]
        
        try:
            print(f"正在处理批次 {batch_idx + 1}/{total_batches} (索引 {start_idx}-{end_idx-1})...")
            batch_embeddings = get_embedding("text-embedding-3-small", batch_texts)
            new_embeddings.extend(batch_embeddings)
            print(f"批次 {batch_idx + 1} 完成，生成 {len(batch_embeddings)} 个向量")
        except Exception as e:
            print(f"批次 {batch_idx + 1} 出错: {e}")
            print("尝试逐个生成该批次的节点...")
            # 如果该批次失败，回退到逐个生成
            for i, text in enumerate(batch_texts):
                try:
                    embedding = get_embedding("text-embedding-3-small", text)
                    new_embeddings.append(embedding)
                except Exception as e:
                    print(f"处理第 {start_idx + i} 个节点时出错: {e}")
                    new_embeddings.append(None)
    
    print(f"总共生成 {len(new_embeddings)} 个向量")
    
    # 过滤掉失败的embedding（如果有）
    if any(emb is None for emb in new_embeddings):
        print("发现失败的embedding，进行过滤...")
        filtered_embeddings = [emb for emb in new_embeddings if emb is not None]
        filtered_node_ids = []
        filtered_texts = []
        for i, emb in enumerate(new_embeddings):
            if emb is not None:
                filtered_node_ids.append(new_node_ids[i])
                filtered_texts.append(new_texts[i])
        
        new_embeddings = filtered_embeddings
        new_node_ids = filtered_node_ids
        new_texts = filtered_texts
        print(f"过滤后剩余 {len(new_embeddings)} 个有效向量")
    
    # 转换为numpy数组
    new_embedding_array = np.array(new_embeddings, dtype='float32')
    print(f"向量维度: {new_embedding_array.shape[1]}")
    
    # 加载或创建FAISS索引
    if os.path.exists(BASE_URL + faiss_index_path):
        print(f"加载已有的FAISS索引: {faiss_index_path}")
        index = faiss.read_index(faiss_index_path)
        
        # 验证维度一致性
        if index.d != new_embedding_array.shape[1]:
            print(f"错误: 维度不匹配！已有索引维度={index.d}, 新向量维度={new_embedding_array.shape[1]}")
            return
    else:
        print("创建新的FAISS索引...")
        dimension = new_embedding_array.shape[1]
        index = faiss.IndexFlatIP(dimension)
    
    # 添加新向量到索引
    print("添加新向量到FAISS索引...")
    faiss.normalize_L2(new_embedding_array) # 归一化
    index.add(new_embedding_array)
    
    # 保存FAISS索引
    faiss.write_index(index, BASE_URL + faiss_index_path)
    print(f"FAISS索引已保存到: {BASE_URL + faiss_index_path}")
    
    # 更新并保存节点ID映射
    all_node_ids = existing_node_ids_list + new_node_ids
    all_texts    = existing_texts_list + new_texts
    
    updated_metadata = {
        "node_ids": all_node_ids,
        "texts": all_texts,
        "dimension": dimension
    }
    
    with open(BASE_URL + metadata_path, "wb") as f:
        pickle.dump(updated_metadata, f)
    print(f"元数据已更新并保存到: {BASE_URL + metadata_path}")
    
    # 统计信息
    print("\n" + "="*60)
    print("处理完成！统计信息：")
    print("="*60)
    print(f"原有节点数: {len(existing_node_ids_list)}")
    print(f"新节点数: {len(new_node_ids)}")
    print(f"总节点数: {len(all_node_ids)}")
    print(f"FAISS索引中向量总数: {index.ntotal}")
    print("="*60)


def main():
    """
    主函数
    """
    
    try:
        # 连接图谱
        print("连接到Neo4j数据库...")
        graph = connect_to_graph()
        print("连接成功！")
        
        # 执行embedding
        embed_knowledge_nodes(graph)
        
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()