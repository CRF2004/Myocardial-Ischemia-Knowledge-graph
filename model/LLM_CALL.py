import os
import json
import requests
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
