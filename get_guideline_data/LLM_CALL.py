import json
import requests  # 如果您使用的是 REST API 进行调用

# ------------------------------------------------------------------------------------
#         3秒步接入 DMXAPI ：  修改 Key 和 Base url (https://www.dmxapi.com)
# ------------------------------------------------------------------------------------
url = "https://www.dmxapi.cn/v1/chat/completions"   # 这里不要用 openai base url，需要改成DMXAPI的中转 https://www.dmxapi.com ，下面是已经改好的。【无需动】

headers = {
    'Accept': 'application/json',
    'Authorization': os.environ.get("DMX_API_KEY", ""), # 这里放你的 DMXapi key
    'User-Agent': 'DMXAPI/1.0.0 (https://www.dmxapi.cn)', 
    'Content-Type': 'application/json'
}

def chat(model_name, prompt):
    payload = json.dumps({
        "model": f"{model_name}",  
        "messages": [
            # {
            #     "role": "developer",
            #     "content": "You are a helpful assistant."
            # },
            {
                "role": "user",
                "content": f'''{prompt}''' #这里传入Prompt
            }
        ]})

    response = requests.request("POST", url, headers=headers, data=payload) #这里是请求数据

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
"""from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModelForSequenceClassification
import torch
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
def load_model(model_path):
    #device = "cuda:0"  # 将模型加载到指定GPU 上
    
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype="auto",
        device_map="auto",
        #device_map={"": "cuda:6"} # 指定模型加载到 GPU 上
    )
    tokenizer = AutoTokenizer.from_pretrained(model_path)#加载分词器
    return tokenizer, model  

# 模型交互。
def chat(device, tokenizer, model, prompt):
    messages = (
        {"role": "system", "content": ""},
        {"role": "user", "content": prompt}
    )
    # 使用分词器的 apply_chat_template 方法将消息格式化为模型可理解的输入格式
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer((text), return_tensors="pt").to(device)    #pt为pytorch张量格式
    #生成模型输出
    generated_ids = model.generate(
        model_inputs.input_ids,
        max_new_tokens=8000,
        #temperature=1.3  # 设置温度参数
    )
    # 由于模型输出包括输入模型，这里切去输入部分
    generated_ids = (output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids))
    # 将模型输出解码为文本
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return response


# from vllm import LLM, SamplingParams
# from transformers import AutoTokenizer
# import os

# # 设置 CUDA 可见设备
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# def load_model(model_path):
#     # 加载分词器（与 Hugging Face 兼容）
#     tokenizer = AutoTokenizer.from_pretrained(model_path)
    
#     # 使用 vLLM 加载模型
#     model = LLM(model=model_path)
    
#     return tokenizer, model

# # 模型交互
# def chat_qwen(tokenizer, model, prompt):
#     # 准备消息
#     messages = [
#         {"role": "system", "content": ""},
#         {"role": "user", "content": prompt}
#     ]
    
#     # 使用分词器的 apply_chat_template 方法将消息格式化为模型可理解的输入格式
#     text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    
#     # 设置生成参数
#     sampling_params = SamplingParams(
#         temperature=0.7,  # 温度参数
#         max_tokens=3000  # 最大生成 token 数
#     )
    
#     # 使用 vLLM 进行推理
#     outputs = model.generate([text], sampling_params)
    
#     # 提取生成的文本
#     response = outputs[0].outputs[0].text
    
#     return response"""