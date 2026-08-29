import spacy
import scispacy
from scispacy.linking import EntityLinker

nlp = spacy.load(r"C:\Users\12879\Desktop\Little code\en_core_sci_sm-0.5.1\en_core_sci_sm\en_core_sci_sm-0.5.1")
nlp.add_pipe("scispacy_linker", config={"linker_name": "umls"})

text =  """
    Older concepts considered a fixed, focal, flow-limiting atherosclerotic stenosis of a large or medium coronary artery as asine qua nonfor inducible myocardial ischaemia and ischaemic chest pain (angina pectoris).
   """
doc = nlp(text)

for ent in doc.ents:
    print(f"实体: {ent.text}")
    for umls_ent in ent._.kb_ents:
        print(f"UMLS ID: {umls_ent[0]}, 置信度: {umls_ent[1]}")


# import spacy
# from scispacy.linking import EntityLinker

# # 加载模型并添加 UMLS 链接器
# nlp = spacy.load(r"C:\Users\12879\Desktop\Little code\en_core_sci_sm-0.5.1\en_core_sci_sm\en_core_sci_sm-0.5.1")
# nlp.add_pipe("scispacy_linker", config={"linker_name": "umls", "threshold": 0.6})

# # 输入文本
# text = """
#     Our understanding of the pathophysiology of CCS is transitioning from a simple to a more complex and dynamic model. Older concepts considered a fixed, focal, flow-limiting atherosclerotic stenosis of a large or medium coronary artery as asine qua nonfor inducible myocardial ischaemia and ischaemic chest pain (angina pectoris). Current concepts have broadened to embrace structural and functional abnormalities in both the macro- and microvascular compartments of the coronary tree that may lead to transient myocardial ischaemia. At the macrovascular level, not only fixed, flow-limiting stenoses but also diffuse atherosclerotic lesions without identifiable luminal narrowing may cause ischaemia under stress;2,3structural abnormalities such as myocardial bridging4and congenital arterial anomalies5or dynamic epicardial vasospasm may be responsible for transient ischaemia. At the microvascular level, coronary microvascular dysfunction (CMD) is increasingly acknowledged as a prevalent factor characterizing the entire spectrum of CCS;6functional and structural microcirculatory abnormalities may cause angina and ischaemia even in patients with non-obstructive disease of the large or medium coronary arteries [angina with non-obstructive coronary arteries (ANOCA); ischaemia with non-obstructive coronary arteries (INOCA)].6Finally, systemic or extracoronary conditions, such as anaemia, tachycardia, blood pressure (BP) changes, myocardial hypertrophy, and fibrosis, may contribute to the complex pathophysiology of non-acute myocardial ischaemia.7
#     The risk factors that predispose to the development of epicardial coronary atherosclerosis also promote endothelial dysfunction and abnormal vasomotion in the entire coronary tree, including the arterioles that regulate coronary flow and resistance,8–10and adversely affect myocardial capillaries,6,11–14leading to their rarefaction. Potential consequences include a lack of flow-mediated vasodilation in the epicardial conductive arteries9and macro- and microcirculatory vasoconstriction.15Of note, different mechanisms of ischaemia may act concomitantly.
#     """
# # 预分割体检指标
# indicators = [indicator.strip() for indicator in text.split(",")]

# # 单独处理每个指标
# for indicator in indicators:
#     print(f"\n处理指标: {indicator}")
#     doc = nlp(indicator)  # 对每个指标单独运行 NLP
#     # 假设整个指标作为一个实体
#     if doc.ents:
#         ent = doc.ents[0]  # 取第一个实体（通常整个指标会被识别为一个实体）
#         print(f"实体: {ent.text}")
#         for umls_ent in ent._.kb_ents:
#             print(f"UMLS ID: {umls_ent[0]}, 置信度: {umls_ent[1]}")
#     else:
#         # 如果 NER 没有识别到实体，直接尝试用整个文本链接 UMLS
#         linker = nlp.get_pipe("scispacy_linker")
#         kb_ents = linker._link_entities(indicator)
#         if kb_ents:
#             print(f"直接链接结果: {indicator}")
#             for umls_ent in kb_ents:
#                 print(f"UMLS ID: {umls_ent[0]}, 置信度: {umls_ent[1]}")
#         else:
#             print(f"未找到 UMLS 匹配: {indicator}")
