import os
import json
import requests
import openai

def get_embedding(model, text):
    """
    获取文本的向量嵌入表示
    
    参数：
        text (str): 需要转换为向量的文本内容
    
    返回：
        list: 文本的向量嵌入表示（浮点数列表）
    
    功能说明：
        调用 OpenAI Embeddings API，将输入文本转换为高维向量表示。
        这个向量能够捕捉文本的语义信息，相似的文本会有相似的向量。
    """
    openai.api_key = os.environ.get("DMX_API_KEY", "")
    openai.base_url = "https://www.dmxapi.cn/v1/"
    response = openai.embeddings.create(
        model=model,
        input=text
    )
    
    # 根据输入类型返回相应结果
    if isinstance(text, str):
        # 单个文本：返回单个向量
        return response.data[0].embedding
    elif isinstance(text, list):
        # 批量文本：返回向量列表
        return [item.embedding for item in response.data]

# # 示例1：单个文本embedding
# print("="*60)
# print("单个文本embedding")
# print("="*60)
# single_text = "这是一个示例文本,用于演示如何获取文本嵌入。"
# single_embedding = get_embedding("text-embedding-3-small", single_text)
# print(f"原始文本: {single_text}")
# print(f"嵌入向量维度: {len(single_embedding)}")
# print(f"向量前5个元素: {single_embedding[:5]}")

# # 示例2：批量文本embedding
# print("\n" + "="*60)
# print("批量文本embedding")
# print("="*60)
# batch_texts = [
#     "今天天气很好",
#     "我喜欢学习人工智能",
#     "OpenAI的API很好用",
#     "向量嵌入很有用",
#     "机器学习是未来的趋势"
# ]

# batch_embeddings = get_embedding("text-embedding-3-small", batch_texts)
# print(f"处理文本数量: {len(batch_texts)}")
# print(f"返回向量数量: {len(batch_embeddings)}")

# for i, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
#     print(f"\n文本 {i+1}: {text[:20]}...")
#     print(f"向量维度: {len(embedding)}")
#     print(f"前3个元素: {embedding[:3]}")
