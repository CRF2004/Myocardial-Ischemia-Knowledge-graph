#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
使用LLM检查测试数据，筛选出考察心肌缺血领域相关知识的问题
"""

import json
import sys
from LLM_CALL import chat

def load_json_file(file_path):
    """加载JSON文件"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def save_json_file(data, file_path):
    """保存JSON文件"""
    with open(file_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def check_myocardial_ischemia_relevance(question_data):
    """
    使用LLM检查问题是否考察心肌缺血领域相关知识
    
    参数:
        question_data (dict): 包含question等信息的字典
    
    返回:
        bool: 如果考察心肌缺血相关知识则返回True，否则返回False
    """
    # 提取问题文本
    question_text = question_data.get('question', '')
    
    # 构建LLM提示词
    prompt = f"""请仔细分析以下医学问题，判断它是否考察心肌缺血（myocardial ischemia）领域的相关知识。

    问题内容：
    {question_text}

    请根据以下标准进行判断：
    1. 问题是否直接涉及心肌缺血、心绞痛、心肌梗死等心肌缺血相关疾病
    2. 问题是否涉及冠心病的诊断、治疗、病理生理等
    3. 问题是否涉及冠状动脉疾病、冠状动脉综合征等
    4. 问题是否涉及心肌缺血的危险因素、预防、并发症等
    5. 问题是否涉及心肌缺血的病理机制、临床表现、诊断方法等

    如果问题主要考察心肌缺血相关知识，请返回 "yes"；如果问题不考察或仅间接涉及心肌缺血（如其他心血管疾病但非缺血性），请返回 "no"。

    请只返回 "yes" 或 "no"，不要包含其他解释。"""

    try:
        response = chat("qwen-max-latest", prompt)
        # 清理响应，去除可能的空格和换行
        response = response.strip().lower()
        
        # 判断响应
        if "yes" in response:
            return True
        elif "no" in response:
            return False
        else:
            # 如果响应不明确，默认为不相关
            print(f"警告: LLM响应不明确: {response}")
            return False
            
    except Exception as e:
        print(f"调用LLM时出错: {e}")
        return False

def filter_questions(input_file, output_file):
    """
    筛选问题，仅保留考察心肌缺血相关知识的题目
    
    参数:
        input_file (str): 输入JSON文件路径
        output_file (str): 输出JSON文件路径
    """
    print(f"正在加载输入文件: {input_file}")
    questions = load_json_file(input_file)
    
    print(f"共加载 {len(questions)} 个问题")
    
    filtered_questions = []
    
    for i, question in enumerate(questions, 1):
        print(f"\n正在检查问题 {i}/{len(questions)} (q_id: {question.get('q_id', 'N/A')})...")
        
        is_relevant = check_myocardial_ischemia_relevance(question)
        
        if is_relevant:
            filtered_questions.append(question)
            print(f"  ✓ 保留 - 问题与心肌缺血相关")
        else:
            print(f"  ✗ 跳过 - 问题与心肌缺血不相关")
    
    print(f"\n筛选完成!")
    print(f"原始问题数量: {len(questions)}")
    print(f"保留问题数量: {len(filtered_questions)}")
    print(f"筛选比例: {len(filtered_questions)/len(questions)*100:.2f}%")
    
    # 保存筛选结果
    print(f"\n正在保存筛选结果到: {output_file}")
    save_json_file(filtered_questions, output_file)
    print("保存完成!")

def main():
    """主函数"""
    # 定义输入和输出文件路径
    input_file_1 = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_1_data.json"
    output_file_1 = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_1_filtered.json"
    
    input_file_2 = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_2_data.json"
    output_file_2 = "/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/qa_test/test_data/relevance_2_filtered.json"
    
    # 处理第一个文件
    print("=" * 60)
    print("开始处理 relevance_1_data.json")
    print("=" * 60)
    filter_questions(input_file_1, output_file_1)
    
    # 处理第二个文件
    print("\n" + "=" * 60)
    print("开始处理 relevance_2_data.json")
    print("=" * 60)
    filter_questions(input_file_2, output_file_2)
    
    print("\n" + "=" * 60)
    print("所有文件处理完成!")
    print("=" * 60)

if __name__ == "__main__":
    main()