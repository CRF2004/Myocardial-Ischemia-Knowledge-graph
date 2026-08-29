import re
import json
from typing import List, Dict, Tuple
from LLM_CALL import chat, parse_llm_response

class MedicalTripletExtractor:
    def __init__(self):
        """初始化医学三元组抽取器"""
        # 定义允许的关系类型
        self.allowed_relations = {
            # 核心医学关系
            "TREATS",           # 治疗关系
            "DIAGNOSES",        # 诊断关系  
            "INDICATES",        # 提示/表明
            "CAUSES",           # 导致
            
            # 通用关联关系
            "ASSOCIATED_WITH",  # 关联关系
            "MEASURES",         # 测量关系
            "AFFECTS",          # 影响关系
            
            # 时间/过程关系
            "PRECEDES",         # 先于
            "PREVENTS",         # 预防
            
            # 结构关系
            "PART_OF"           # 部分关系
        }
    
    def split_sentences(self, text: str) -> List[Tuple[int, str]]:
        """
        分句并编号
        返回: [(sentence_id, sentence_text), ...]
        """
        # 清理文本
        text = text.strip()
        
        # 常见缩写列表（包括句点）
        abbreviations = {
            'Dr.', 'Mr.', 'Mrs.', 'Ms.', 'Prof.', 'vs.', 'etc.', 
            'i.e.', 'e.g.', 'et al.', 'Inc.', 'Ltd.', 'Co.',
            'U.S.', 'U.K.', 'a.m.', 'p.m.'
        }
        
        # 使用正则表达式保护缩写
        # 将缩写中的句点替换为临时标记，防止被分割
        for abbr in abbreviations:
            text = text.replace(abbr, abbr.replace('.', '\ue000'))  # 使用 Unicode 私有字符
        
        # 分句：按句号、问号、感叹号分割
        sentences = []
        current_sentence = ""
        i = 0
        while i < len(text):
            char = text[i]
            current_sentence += char
            
            # 检查是否遇到句子结束符号
            if char in '.!?':
                # 检查后面是否有引号
                next_i = i + 1
                while next_i < len(text) and text[next_i] in '"\'':
                    current_sentence += text[next_i]
                    next_i += 1
                
                # 检查是否是句子结束（后面跟空格、换行或结尾）
                if next_i >= len(text) or text[next_i].isspace():
                    sentences.append(current_sentence.strip())
                    current_sentence = ""
                    i = next_i
                    continue
            
            i += 1
        
        # 添加最后一个句子（如果有）
        if current_sentence.strip():
            sentences.append(current_sentence.strip())
        
        # 恢复缩写中的句点
        sentences = [sent.replace('\ue000', '.') for sent in sentences]
        
        # 进一步清理：移除空句子，合并过短的片段
        cleaned_sentences = []
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 3:  # 至少4个字符才算有效句子
                cleaned_sentences.append(sent)
        
        # 编号从1开始
        numbered_sentences = [(i+1, sent) for i, sent in enumerate(cleaned_sentences)]
        
        return numbered_sentences
    
    def create_extraction_prompt(self, numbered_sentences: List[Tuple[int, str]]) -> str:
        """
        创建三元组抽取的prompt
        """
        # 构建句子列表
        sentence_list = "\n".join([f"[{num}] {text}" for num, text in numbered_sentences])
        
        # 关系类型说明
        with open("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/triples/relationships.md", "r", encoding="utf-8") as f:
            relations_desc = f.read()
        
        with open("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/triples/extraction_prompt_en.md", "r", encoding="utf-8") as f:
            prompt = f.read()
        
        return prompt.format(sentence_list=sentence_list, relations_desc=relations_desc)
    
    def extract_triplets(self, text: str, model_name: str = "gemini-2.5-flash-lite") -> Dict:
        """
            三元组抽取流程
        """
        try:
            # 步骤1: 分句并编号
            print("     正在分句...")
            numbered_sentences = self.split_sentences(text)
            print(f"    共分割出 {len(numbered_sentences)} 个句子")
            
            # 显示分句结果
            for num, sent in numbered_sentences:
                print(f"[{num}] {sent}")
            print()
            
            # 步骤2: 动态分批处理（800字符阈值）
            print("     正在进行动态批处理...")
            char_threshold = 1200
            batches = []
            current_batch = []
            current_char_count = 0
            
            for num, sentence in numbered_sentences:
                sentence_with_num = f"[{num}] {sentence}"
                sentence_char_count = len(sentence_with_num)
                
                # 如果单句就超过阈值，单独处理
                if sentence_char_count > char_threshold:
                    if current_batch:
                        batches.append(current_batch)
                        current_batch = []
                        current_char_count = 0
                    batches.append([(num, sentence)])  # 超长句单独成批
                
                # 如果加入当前句子会超过阈值，先处理当前批次
                elif current_char_count + sentence_char_count > char_threshold:
                    if current_batch:
                        batches.append(current_batch)
                    current_batch = [(num, sentence)]
                    current_char_count = sentence_char_count
                
                # 正常加入当前批次
                else:
                    current_batch.append((num, sentence))
                    current_char_count += sentence_char_count
            
            # 处理最后一个批次
            if current_batch:
                batches.append(current_batch)
            
            print(f"    动态分批完成，共分成 {len(batches)} 个批次：")
            for i, batch in enumerate(batches):
                char_count = sum(len(f"[{num}] {sent}") for num, sent in batch)
                print(f"    批次 {i+1}: {len(batch)} 句, {char_count} 字符")
            print()
            
            # 步骤3: 逐批次处理
            all_triplets = []
            all_responses = []
            
            for batch_idx, batch in enumerate(batches):
                print(f"    正在处理批次 {batch_idx + 1}/{len(batches)}...")
                
                # 步骤4: 构建prompt
                print(" 正在构建prompt...")
                prompt = self.create_extraction_prompt(batch)
                
                # 步骤3: 调用LLM
                print("     正在调用LLM进行三元组抽取...")
                response = chat(model_name, prompt)
                all_responses.append(response)
                
                with open("/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/triples/LLM_response.txt", "w", encoding="utf-8") as f:
                    f.write(response)
                print(f"        LLM响应: {response[:200]}...保存到 triples/LLM_response.txt")
            
                # 步骤4: 解析响应
                print("     正在解析响应...")
                try:
                    match = re.search(r'<!-- BEGIN OUTPUT -->(.*?)<!-- END OUTPUT -->', response, re.DOTALL)
                    if match:
                        response = match.group(1).strip()
                except Exception as e:
                    print(f"解析响应时出错: {e}")
                
                batch_result = parse_llm_response(response)
                
                # 合并当前批次的三元组
                if "triplets" in batch_result:
                    all_triplets.extend(batch_result["triplets"])
                    print(f"批次 {batch_idx + 1} 抽取到 {len(batch_result['triplets'])} 个三元组")
                print()
            
            final_result = {"triplets": all_triplets}
            
            # 步骤5: 验证结果
            validated_result = self.validate_triplets(final_result, numbered_sentences)
            
            return {
                "sentences": numbered_sentences,
                "extraction_result": validated_result,
                "total_triplets": len(validated_result.get("triplets", []))
            }
            
        except Exception as e:
            return {
                "error": str(e),
                "sentences": numbered_sentences if 'numbered_sentences' in locals() else [],
                "extraction_result": {},
                "total_triplets": 0
            }
    
    def validate_triplets(self, result: Dict, sentences: List[Tuple[int, str]]) -> Dict:
        """
        验证三元组结果
        """
        if not isinstance(result, dict) or "triplets" not in result:
            print("警告: 响应格式不正确")
            return {"triplets": [], "warnings": ["响应格式不正确"]}
        
        validated_triplets = []
        warnings = []
        sentence_ids = set(num for num, _ in sentences)
        
        for triplet in result["triplets"]:
            # 检查必需字段
            if not all(key in triplet for key in ["head", "relation", "tail", "source_sentences"]):
                warnings.append(f"三元组缺少必需字段: {triplet}")
                continue
            
            # 检查关系类型
            if triplet["relation"] not in self.allowed_relations:
                warnings.append(f"不允许的关系类型: {triplet['relation']}")
                continue
            
            # 检查句子编号
            invalid_ids = [sid for sid in triplet["source_sentences"] if sid not in sentence_ids]
            if invalid_ids:
                warnings.append(f"无效的句子编号: {invalid_ids}")
                # 过滤无效编号
                triplet["source_sentences"] = [sid for sid in triplet["source_sentences"] if sid in sentence_ids]
            
            if triplet["source_sentences"]:  # 如果还有有效的句子编号
                validated_triplets.append(triplet)
        
        return {
            "triplets": validated_triplets,
            "warnings": warnings
        }
    
    def format_results(self, results: Dict) -> str:
        """
        格式化输出结果
        """
        output = []
        output.append("=== 医学文本三元组抽取结果 ===\n")
        
        # 分句结果
        output.append("分句结果:")
        for num, sent in results["sentences"]:
            output.append(f"[{num}] {sent}")
        output.append("")
        
        # 抽取结果
        extraction = results["extraction_result"]
        output.append(f"抽取的三元组数量: {results['total_triplets']}")
        
        if extraction.get("warnings"):
            output.append("\n警告:")
            for warning in extraction["warnings"]:
                output.append(f"- {warning}")
        
        if extraction.get("triplets"):
            output.append("\n抽取的三元组:")
            for i, triplet in enumerate(extraction["triplets"], 1):
                output.append(f"{i}. 头实体: {triplet['head']}")
                output.append(f"   关系: {triplet['relation']}")
                output.append(f"   尾实体: {triplet['tail']}")
                output.append(f"   来源句子: {triplet['source_sentences']}")
                output.append("")
        
        if results.get("error"):
            output.append(f"错误: {results['error']}")
        
        return "\n".join(output)
def extract_content(text):
    # 分割文本行
    lines = text.split('\n')
    
    # 找到包含实际内容的行（通常在这些标记信息之后）
    content_lines = []
    start_reading = False
    
    for line in lines:
        if line.startswith('='):  # 分隔线之后开始读取
            start_reading = True
            continue
        if start_reading and line.strip():  # 只读取有内容的行
            content_lines.append(line)
    
    return '\n'.join(content_lines)

# 使用示例
def test():
    # 初始化抽取器
    extractor = MedicalTripletExtractor()
    
    # 示例医学文本
    sample_text = """
Our understanding of the pathophysiology of CCS is transitioning from a simple to a more complex and dynamic model. Older concepts considered a fixed, focal, flow-limiting atherosclerotic stenosis of a large or medium coronary artery as asine qua nonfor inducible myocardial ischaemia and ischaemic chest pain (angina pectoris). Current concepts have broadened to embrace structural and functional abnormalities in both the macro- and microvascular compartments of the coronary tree that may lead to transient myocardial ischaemia. At the macrovascular level, not only fixed, flow-limiting stenoses but also diffuse atherosclerotic lesions without identifiable luminal narrowing may cause ischaemia under stress;2,3structural abnormalities such as myocardial bridging4and congenital arterial anomalies5or dynamic epicardial vasospasm may be responsible for transient ischaemia. At the microvascular level, coronary microvascular dysfunction (CMD) is increasingly acknowledged as a prevalent factor characterizing the entire spectrum of CCS;6functional and structural microcirculatory abnormalities may cause angina and ischaemia even in patients with non-obstructive disease of the large or medium coronary arteries [angina with non-obstructive coronary arteries (ANOCA); ischaemia with non-obstructive coronary arteries (INOCA)].6Finally, systemic or extracoronary conditions, such as anaemia, tachycardia, blood pressure (BP) changes, myocardial hypertrophy, and fibrosis, may contribute to the complex pathophysiology of non-acute myocardial ischaemia.7
The risk factors that predispose to the development of epicardial coronary atherosclerosis also promote endothelial dysfunction and abnormal vasomotion in the entire coronary tree, including the arterioles that regulate coronary flow and resistance,8–10and adversely affect myocardial capillaries,6,11–14leading to their rarefaction. Potential consequences include a lack of flow-mediated vasodilation in the epicardial conductive arteries9and macro- and microcirculatory vasoconstriction.15Of note, different mechanisms of ischaemia may act concomitantly.
    """
    
    print("示例文本:")
    print(sample_text)
    print("\n" + "="*50 + "\n")
    
    # 执行三元组抽取
    results = extractor.extract_triplets(sample_text, model_name="gpt-4o-mini")
    with open("triples/output.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=4)
    
    # 格式化输出
    formatted_output = extractor.format_results(results)
    print(formatted_output)
    
    # 返回JSON结果供进一步处理
    return results

def main():
    # 初始化抽取器
    extractor = MedicalTripletExtractor()
    
    target_files = [
        # "get_guideline_data\structured_output\chapter_texts\chapter_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_2_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_2_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_2_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_2_4.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_1_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_1_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_1_2_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_1_2_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_2_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_2_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_2_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_2_4.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_2_5.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_1_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_1_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_1_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_2_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_2_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_2_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_2_4.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_2_5.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_3_1.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_3_2.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_3_3.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_3_4.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_4.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_3_5.txt",
        # "get_guideline_data\structured_output\chapter_texts\chapter_3_4.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_1.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_1.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_2_1.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_2_2.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_2.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_3.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_4.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_5_1.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_5_2.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_5.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2_6.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_2.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_1.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_2.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_3.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_4.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_5.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_6.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_7.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_8.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_9.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_10.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3_11.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_3.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_5_4.txt",
        # "SRP心肌缺血知识图谱/get_guideline_data/structured_output/chapter_texts/chapter_05.txt",
        
        # 未处理的文件
        "get_guideline_data/structured_output/chapter_texts/chapter_13.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_2_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_2_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_2_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_2_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_2_5.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_2_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_2_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_2_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_1_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_1_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_1_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3_5.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5_5.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5_6.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5_7.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_5.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_6.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4_7.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_4_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_1_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_1_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_1_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_1_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_1_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_1_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_2_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_2_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_3_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_3_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_3_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_3.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_4_1.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_4_2.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_4.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_5.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_6_6.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_64.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_66.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_67.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_68.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_69.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_7.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_70.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_72.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_73.txt",
        "get_guideline_data/structured_output/chapter_texts/chapter_8.txt",
    ]
    for file in target_files:
        title = re.search(r"[\\/]([^\\/]+)\.\w+$", file).group(1)
        print(f"正在处理：{title}")
        
        with open(file, "r", encoding="utf-8") as f:
            sample_text = f.read()
        sample_text = extract_content(sample_text)
        # 执行三元组抽取
        results = extractor.extract_triplets(sample_text, model_name="gpt-4o-mini")
        results["title"] = title
        
        with open(f"/mnt/chengrongfeng_private/SRP心肌缺血知识图谱/triples/triple_data_new/output_{title}_cot.json", "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=4)

if __name__ == "__main__":
    results = main()