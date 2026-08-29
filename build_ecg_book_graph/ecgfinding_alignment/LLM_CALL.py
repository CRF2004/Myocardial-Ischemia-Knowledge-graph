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
            {"role": "system", "content": "ECG Finding → Knowledge Layer Relation Extraction"},
            {
                "role": "user",
                "content": f'''{prompt}'''
            }
        ]})

    response = requests.request("POST", url, headers=headers, data=payload)

    # 解析 JSON 字符串为 Python 对象（字典）
    resp_data = json.loads(response.text)
    # print(resp_data)  # 打印完整的响应数据，便于调试
    # 提取 AI 生成的回答内容
    ai_response = resp_data['choices'][0]['message']['content']
    # entry["response"] = ai_response
    return ai_response

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
        
        # 解析JSON
        parsed_data = json.loads(json_str)
        return parsed_data
        
    except json.JSONDecodeError as e:
        print(f"JSON解析错误: {e}")
        print(text)
        return None
    except Exception as e:
        print(f"其他错误: {e}")
        return None