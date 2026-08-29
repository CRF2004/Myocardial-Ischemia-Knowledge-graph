import json
import re
import LLM_CALL

# 判断文本是否包含中文
def contains_chinese(text):
    return bool(re.search(r'[\u4e00-\u9fff]', text))

# 带上下文翻译函数
def translate_to_english(item):
    # 拼接上下文提示
    prompt = f"""
请根据以下概念信息生成专业、准确的英文定义：
概念ID: {item.get('ontology_id', '')}
概念中文名: {item.get('name_cn', '')}
概念英文名: {item.get('name_en', '')}
概念类型: {item.get('type', '')}
中文定义: {item.get('definition', '')}

如果原中文定义不够专业或不完整，请使用专业术语补充完善英文定义。
只返回英文定义文本，不要其他说明。
"""
    response = LLM_CALL.chat("gpt-4o-mini", prompt)
    return response.strip()

# 主函数
def process_json(file_path, output_path=None):
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    for item in data:
        definition = item.get('definition', '')
        # 如果是中文，生成专业英文定义
        if contains_chinese(definition):
            print(f"处理中文定义: {definition}")
            generated_en = translate_to_english(item)
            item['definition'] = generated_en
        else:
            # 如果已经是英文，可以直接复制到新字段
            item['definition'] = definition

    # 保存修改后的 JSON
    if output_path is None:
        output_path = file_path  # 覆盖原文件
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    return data

# 示例调用
if __name__ == "__main__":
    processed_data = process_json("SRP心肌缺血知识图谱/ontology/disease.json", "SRP心肌缺血知识图谱/ontology/disease_translated.json")
