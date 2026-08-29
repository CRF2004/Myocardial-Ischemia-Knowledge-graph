import os
import json
import requests
import openai

url = "https://www.dmxapi.cn/v1/chat/completions" 
headers = {
    'Accept': 'application/json',
    'Authorization': os.environ.get("DMX_API_KEY", ""),
    'User-Agent': 'DMXAPI/1.0.0 (https://www.dmxapi.cn)', 
    'Content-Type': 'application/json'
}

def chat(model_name, prompt):
    payload = json.dumps({
        "model": f"{model_name}",  
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {
                "role": "user",
                "content": f'''{prompt}'''
            }
        ]})

    response = requests.request("POST", url, headers=headers, data=payload)

    resp_data = json.loads(response.text)
    ai_response = resp_data['choices'][0]['message']['content']
    return ai_response

if __name__ == "__main__":
    test_prompt = "请简要介绍一下人工智能的发展历程。"
    response = chat("gpt-4o-mini", test_prompt)
    print("AI Response:", response)
embedding_url = "https://www.dmxapi.cn/v1/"
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
    openai.api_key = headers["Authorization"]
    openai.base_url = embedding_url
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

"""
# 示例1：单个文本embedding
print("="*60)
print("单个文本embedding")
print("="*60)
single_text = "这是一个示例文本,用于演示如何获取文本嵌入。"
single_embedding = get_embedding("text-embedding-3-small", single_text)
print(f"原始文本: {single_text}")
print(f"嵌入向量维度: {len(single_embedding)}")
print(f"向量前5个元素: {single_embedding[:5]}")

# 示例2：批量文本embedding
print("\n" + "="*60)
print("批量文本embedding")
print("="*60)
batch_texts = [
    "今天天气很好",
    "我喜欢学习人工智能",
    "OpenAI的API很好用",
    "向量嵌入很有用",
    "机器学习是未来的趋势"
]

batch_embeddings = get_embedding("text-embedding-3-small", batch_texts)
print(f"处理文本数量: {len(batch_texts)}")
print(f"返回向量数量: {len(batch_embeddings)}")

for i, (text, embedding) in enumerate(zip(batch_texts, batch_embeddings)):
    print(f"\n文本 {i+1}: {text[:20]}...")
    print(f"向量维度: {len(embedding)}")
    print(f"前3个元素: {embedding[:3]}")
"""
    

import json
import re

def parse_llm_response(text):
    """
    从文本中提取JSON字符串并解析
    """
    try:
        # 查找第一个 { 的位置
        start_index = text.find('{')
        # 查找最后一个 } 的位置
        end_index = text.rfind('}')
        
        if start_index == -1 or end_index == -1:
            print("未找到有效的JSON边界")
            return None
        
        # 提取JSON字符串（包含边界）
        json_str = text[start_index:end_index + 1]
        
        # 清理控制字符（0x00-0x1F, 0x7F）
        json_str = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F\x7F]', '', json_str)
        
        # 修复字符串值内部未转义的双引号
        def escape_inner_quotes(match):
            content = match.group(1)
            # 将内容中的未转义双引号替换为 \"
            escaped_content = content.replace('"', '\\"')
            return f'"{escaped_content}"'
        
        # 修复无效的单引号转义（如 \' 替换为 '）
        json_str = re.sub(r'\\\'', '\'', json_str)
        
        # 使用正则表达式匹配JSON字符串值（"..."），并修复内部未转义的双引号
        json_str = re.sub(r'"((?:[^"\\]|\\.)*)"', escape_inner_quotes, json_str)
        
        # 解析JSON
        parsed_data = json.loads(json_str)
        return parsed_data
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(f"问题JSON: {json_str}")
        
        # 尝试更简单的修复方法
        try:
            # 清理换行符和多余空格
            simple_fix = json_str.replace('\n', '\\n').replace('\r', '\\r')
            # 再次清理无效的单引号转义
            simple_fix = re.sub(r'\\\'', '\'', simple_fix)
            # 再次尝试修复未转义的双引号
            simple_fix = re.sub(r'"((?:[^"\\]|\\.)*)"', escape_inner_quotes, simple_fix)
            parsed_data = json.loads(simple_fix)
            print("使用简单修复方法成功解析")
            return parsed_data
        except json.JSONDecodeError as e2:
            print(f"简单修复失败: {e2}")
            print(f"尝试修复后的JSON: {simple_fix}")
            return None
            
    except Exception as e:
        print(f"其他错误: {e}")
        return None