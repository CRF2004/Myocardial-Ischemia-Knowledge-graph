#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
本地UMLS Metathesaurus与SciSpacy集成示例
使用本地UMLS数据库，无需API调用
"""

import os
import sqlite3
import pandas as pd
import spacy
from spacy.tokens import Span, Doc
from typing import List, Dict, Tuple, Optional
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class LocalUMLSLinker:
    """本地UMLS链接器，用于实体链接和概念映射"""
    
    def __init__(self, umls_db_path: str):
        """
        初始化本地UMLS链接器
        
        Args:
            umls_db_path: UMLS SQLite数据库路径
        """
        self.umls_db_path = umls_db_path
        self.conn = None
        self._init_database()
    
    def _init_database(self):
        """初始化数据库连接"""
        try:
            self.conn = sqlite3.connect(self.umls_db_path, check_same_thread=False)
            logger.info(f"成功连接UMLS数据库: {self.umls_db_path}")
        except Exception as e:
            logger.error(f"连接UMLS数据库失败: {e}")
            raise
    
    def search_concepts(self, term: str, limit: int = 10) -> List[Dict]:
        """
        搜索UMLS概念
        
        Args:
            term: 搜索词
            limit: 返回结果数量限制
            
        Returns:
            概念列表，每个概念包含CUI、术语、语义类型等信息
        """
        if not self.conn:
            return []
        
        try:
            # 这里的SQL查询需要根据你的UMLS数据库结构调整
            query = """
            SELECT DISTINCT 
                m.CUI, 
                m.STR as term,
                s.STY as semantic_type,
                s.TUI as type_id
            FROM MRCONSO m
            LEFT JOIN MRSTY s ON m.CUI = s.CUI
            WHERE m.STR LIKE ? 
            AND m.LAT = 'ENG'
            AND m.SUPPRESS = 'N'
            ORDER BY LENGTH(m.STR)
            LIMIT ?
            """
            
            cursor = self.conn.cursor()
            cursor.execute(query, (f"%{term}%", limit))
            results = cursor.fetchall()
            
            concepts = []
            for row in results:
                concepts.append({
                    'cui': row[0],
                    'term': row[1],
                    'semantic_type': row[2] or 'Unknown',
                    'type_id': row[3] or 'Unknown',
                    'score': self._calculate_similarity_score(term, row[1])
                })
            
            # 按相似度分数排序
            concepts.sort(key=lambda x: x['score'], reverse=True)
            return concepts
            
        except Exception as e:
            logger.error(f"搜索概念时出错: {e}")
            return []
    
    def _calculate_similarity_score(self, query: str, term: str) -> float:
        """计算相似度分数（简单的字符串匹配）"""
        query_lower = query.lower()
        term_lower = term.lower()
        
        if query_lower == term_lower:
            return 1.0
        elif query_lower in term_lower or term_lower in query_lower:
            return 0.8
        else:
            # 可以使用更复杂的相似度算法，如编辑距离等
            common_chars = len(set(query_lower) & set(term_lower))
            total_chars = len(set(query_lower) | set(term_lower))
            return common_chars / total_chars if total_chars > 0 else 0.0
    
    def get_concept_info(self, cui: str) -> Optional[Dict]:
        """获取特定CUI的详细信息"""
        if not self.conn:
            return None
        
        try:
            query = """
            SELECT DISTINCT 
                m.CUI, 
                m.STR,
                s.STY,
                d.DEF
            FROM MRCONSO m
            LEFT JOIN MRSTY s ON m.CUI = s.CUI
            LEFT JOIN MRDEF d ON m.CUI = d.CUI
            WHERE m.CUI = ?
            AND m.LAT = 'ENG'
            AND m.SUPPRESS = 'N'
            LIMIT 1
            """
            
            cursor = self.conn.cursor()
            cursor.execute(query, (cui,))
            result = cursor.fetchone()
            
            if result:
                return {
                    'cui': result[0],
                    'preferred_term': result[1],
                    'semantic_type': result[2] or 'Unknown',
                    'definition': result[3] or 'No definition available'
                }
            return None
            
        except Exception as e:
            logger.error(f"获取概念信息时出错: {e}")
            return None

# 全局变量存储UMLS链接器实例
_umls_linker = None

@spacy.Language.factory("umls_linker", 
                       assigns=["doc._.umls_entities", "span._.umls_concepts"])
def create_umls_linker(nlp, name: str, confidence_threshold: float = 0.7):
    """创建UMLS实体链接组件的工厂函数"""
    return UMLSEntityLinkerComponent(nlp, name, _umls_linker, confidence_threshold)

class UMLSEntityLinkerComponent:
    """spaCy管道组件，用于UMLS实体链接"""
    
    def __init__(self, nlp, name: str, umls_linker: LocalUMLSLinker, 
                 confidence_threshold: float = 0.7):
        """
        初始化UMLS实体链接组件
        
        Args:
            nlp: spaCy模型
            name: 组件名称
            umls_linker: UMLS链接器
            confidence_threshold: 置信度阈值
        """
        self.nlp = nlp
        self.name = name
        self.umls_linker = umls_linker
        self.confidence_threshold = confidence_threshold
        
        # 为Doc和Span添加自定义属性
        if not Doc.has_extension("umls_entities"):
            Doc.set_extension("umls_entities", default=[])
        if not Span.has_extension("umls_concepts"):
            Span.set_extension("umls_concepts", default=[])
    
    def __call__(self, doc):
        """处理文档并添加UMLS链接"""
        umls_entities = []
        
        # 对命名实体进行UMLS链接
        for ent in doc.ents:
            concepts = self.umls_linker.search_concepts(ent.text, limit=5)
            
            # 过滤低置信度的概念
            filtered_concepts = [
                c for c in concepts 
                if c['score'] >= self.confidence_threshold
            ]
            
            if filtered_concepts:
                # 设置实体的UMLS概念
                ent.set_extension("umls_concepts", default=filtered_concepts, force=True)
                
                umls_entities.append({
                    'text': ent.text,
                    'label': ent.label_,
                    'start': ent.start,
                    'end': ent.end,
                    'umls_concepts': filtered_concepts
                })
        
        # 设置文档的UMLS实体
        doc._.umls_entities = umls_entities
        return doc

def create_umls_database(umls_dir: str, db_path: str):
    """
    从UMLS文件创建SQLite数据库
    这是一个简化版本，实际使用时可能需要更复杂的处理
    
    Args:
        umls_dir: UMLS文件目录（包含MRCONSO.RRF等文件）
        db_path: 输出的SQLite数据库路径
    """
    logger.info("开始创建UMLS SQLite数据库...")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 创建MRCONSO表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MRCONSO (
            CUI TEXT,
            LAT TEXT,
            TS TEXT,
            LUI TEXT,
            STT TEXT,
            SUI TEXT,
            ISPREF TEXT,
            AUI TEXT,
            SAUI TEXT,
            SCUI TEXT,
            SDUI TEXT,
            SAB TEXT,
            TTY TEXT,
            CODE TEXT,
            STR TEXT,
            SRL TEXT,
            SUPPRESS TEXT,
            CVF TEXT
        )
    """)
    
    # 创建MRSTY表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MRSTY (
            CUI TEXT,
            TUI TEXT,
            STN TEXT,
            STY TEXT,
            ATUI TEXT,
            CVF TEXT
        )
    """)
    
    # 创建MRDEF表
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS MRDEF (
            CUI TEXT,
            AUI TEXT,
            ATUI TEXT,
            SATUI TEXT,
            SAB TEXT,
            DEF TEXT,
            SUPPRESS TEXT,
            CVF TEXT
        )
    """)
    
    # 导入MRCONSO数据
    mrconso_path = os.path.join(umls_dir, "MRCONSO.RRF")
    if os.path.exists(mrconso_path):
        logger.info("导入MRCONSO数据...")
        with open(mrconso_path, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 100000 == 0:
                    logger.info(f"已处理 {line_num} 行...")
                
                fields = line.strip().split('|')
                if len(fields) >= 18:
                    cursor.execute("""
                        INSERT INTO MRCONSO VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                    """, fields[:18])
        
        conn.commit()
        logger.info("MRCONSO数据导入完成")
    
    # 创建索引以提高查询性能
    logger.info("创建数据库索引...")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mrconso_cui ON MRCONSO(CUI)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mrconso_str ON MRCONSO(STR)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mrsty_cui ON MRSTY(CUI)")
    
    conn.commit()
    conn.close()
    logger.info(f"UMLS数据库创建完成: {db_path}")

def main():
    """主函数示例"""
    # 配置路径
    UMLS_DIR = "/mnt/29T_HardDisk/dataset/UMLS_2025AA"  # 替换为你的UMLS目录
    UMLS_DB_PATH = "umls_2025aa.db"  # SQLite数据库路径
    SPACY_MODEL = r"C:\Users\12879\Desktop\Little code\en_core_sci_sm-0.5.1\en_core_sci_sm\en_core_sci_sm-0.5.1"
    
    # 1. 如果数据库不存在，从UMLS文件创建数据库
    if not os.path.exists(UMLS_DB_PATH):
        logger.info("UMLS数据库不存在，开始创建...")
        create_umls_database(UMLS_DIR, UMLS_DB_PATH)
    
    # 2. 加载spaCy模型
    logger.info(f"加载spaCy模型: {SPACY_MODEL}")
    nlp = spacy.load(SPACY_MODEL)
    
    # 3. 创建本地UMLS链接器
    global _umls_linker
    _umls_linker = LocalUMLSLinker(UMLS_DB_PATH)
    
    # 4. 添加UMLS链接组件到spaCy管道
    nlp.add_pipe("umls_linker", config={"confidence_threshold": 0.6}, last=True)
    
    # 5. 测试文本处理
    test_text = """
    Careful and detailed history taking is the initial step in diagnostic management for all clinical scenarios within the spectrum of CCS. Although chest pain or discomfort (Figure 3) is the most cardinal symptom of CCS, it must be emphasized that many patients do not present with characteristic anginal symptoms and that the symptomatology may vary with age, sex, race, socioeconomic class, and geographical location. In contemporary studies, only 10% to 25% of patients with suspected CCS present with angina with classic aggravating and relieving factors, while 57% to 78% have symptoms less characteristic of angina and 10% to 15% have dyspnoea on exertion.
    """
    
    logger.info("处理测试文本...")
    doc = nlp(test_text)
    
    # 6. 输出结果
    print("=== 实体识别和UMLS链接结果 ===")
    for ent in doc.ents:
        print(f"\n实体: {ent.text} ({ent.label_})")
        
        if hasattr(ent._, 'umls_concepts') and ent._.umls_concepts:
            print("UMLS概念:")
            for concept in ent._.umls_concepts[:5]:  # 只显示前3个概念
                print(f"  - CUI: {concept['cui']}")
                print(f"    术语: {concept['term']}")
                print(f"    语义类型: {concept['semantic_type']}")
                print(f"    置信度: {concept['score']:.3f}")
        else:
            print("  未找到匹配的UMLS概念")
    
    # 7. 显示文档级别的UMLS实体
    print(f"\n=== 文档UMLS实体总数: {len(doc._.umls_entities)} ===")

if __name__ == "__main__":
    main()

    
