import json
import subprocess
import os
import tempfile
from pathlib import Path
from typing import List, Dict, Any
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class MedicalGuideParser:
    def __init__(self, pdftotext_path: str, pdf_path: str, toc_json_path: str):
        """
        初始化医学指南解析器
        
        Args:
            pdftotext_path: pdftotext.exe的完整路径
            pdf_path: PDF文件路径
            toc_json_path: 目录JSON文件路径
        """
        self.pdftotext_path = pdftotext_path
        self.pdf_path = pdf_path
        self.toc_json_path = toc_json_path
        self.parsed_content = {}
        self.index_structure = {}
        
        # 验证文件存在性
        self._validate_files()
    
    def _validate_files(self):
        """验证必要文件是否存在"""
        if not os.path.exists(self.pdftotext_path):
            raise FileNotFoundError(f"pdftotext工具未找到: {self.pdftotext_path}")
        
        if not os.path.exists(self.pdf_path):
            raise FileNotFoundError(f"PDF文件未找到: {self.pdf_path}")
        
        if not os.path.exists(self.toc_json_path):
            raise FileNotFoundError(f"目录JSON文件未找到: {self.toc_json_path}")
    
    def load_toc(self) -> List[Dict[str, Any]]:
        """加载目录JSON文件"""
        try:
            with open(self.toc_json_path, 'r', encoding='utf-8') as f:
                toc_data = json.load(f)
            logger.info(f"成功加载目录，共{len(toc_data)}个章节")
            return toc_data
        except Exception as e:
            logger.error(f"加载目录JSON文件失败: {e}")
            raise
    
    def extract_text_from_pages(self, start_page: int, end_page: int) -> str:
        """
        使用pdftotext提取指定页面范围的文本
        
        Args:
            start_page: 起始页码
            end_page: 结束页码
            
        Returns:
            提取的文本内容
        """
        try:
            # 创建临时文件来存储提取的文本
            with tempfile.NamedTemporaryFile(mode='w+', suffix='.txt', delete=False) as temp_file:
                temp_path = temp_file.name
            
            # 构建pdftotext命令
            # -f: 起始页面, -l: 结束页面, -enc: 编码格式
            cmd = [
                self.pdftotext_path,
                '-f', str(start_page),
                '-l', str(end_page),
                '-enc', 'UTF-8',
                self.pdf_path,
                temp_path
            ]
            
            # 执行命令
            result = subprocess.run(cmd, capture_output=True, text=True, encoding='utf-8')
            
            if result.returncode != 0:
                logger.error(f"pdftotext执行失败: {result.stderr}")
                return ""
            
            # 读取提取的文本
            with open(temp_path, 'r', encoding='utf-8') as f:
                text_content = f.read()
            
            # 清理临时文件
            os.unlink(temp_path)
            
            return text_content.strip()
            
        except Exception as e:
            logger.error(f"提取页面{start_page}-{end_page}文本失败: {e}")
            return ""
    
    def clean_text(self, text: str) -> str:
        """
        清理提取的文本内容
        
        Args:
            text: 原始文本
            
        Returns:
            清理后的文本
        """
        if not text:
            return ""
        
        # 移除多余的空行
        lines = text.split('\n')
        cleaned_lines = []
        
        for line in lines:
            line = line.strip()
            if line:  # 只保留非空行
                cleaned_lines.append(line)
        
        # 重新组合文本，每行之间用单个换行符连接
        cleaned_text = '\n'.join(cleaned_lines)
        
        return cleaned_text
    
    def build_index_structure(self, toc_item: Dict[str, Any], text_content: str):
        """
        构建索引结构
        
        Args:
            toc_item: 目录项
            text_content: 文本内容
        """
        section_id = toc_item['id']
        
        # 构建索引条目
        index_entry = {
            'id': section_id,
            'title': toc_item['title'],
            'start_page': toc_item['start_page'],
            'end_page': toc_item['end_page'],
            'level': toc_item['level'],
            'parent_id': toc_item.get('parent_id', ''),
            'text_length': len(text_content),
            'word_count': len(text_content.split()) if text_content else 0,
            'has_content': bool(text_content.strip())
        }
        
        # 存储到索引结构中
        self.index_structure[section_id] = index_entry
        
        # 存储实际文本内容
        self.parsed_content[section_id] = text_content
    
    def parse_all_sections(self):
        """解析所有章节"""
        toc_data = self.load_toc()
        
        logger.info("开始解析PDF内容...")
        
        for i, toc_item in enumerate(toc_data, 1):
            section_id = toc_item['id']
            title = toc_item['title']
            start_page = toc_item['start_page']
            end_page = toc_item['end_page']
            
            logger.info(f"处理第{i}/{len(toc_data)}个章节: {title} (页面{start_page}-{end_page})")
            
            # 提取文本内容
            raw_text = self.extract_text_from_pages(start_page, end_page)
            
            # 清理文本
            cleaned_text = self.clean_text(raw_text)
            
            # 构建索引
            self.build_index_structure(toc_item, cleaned_text)
            
            logger.info(f"完成章节{section_id}，提取文本{len(cleaned_text)}字符")
    
    def save_results(self, output_dir: str = "output"):
        """
        保存解析结果
        
        Args:
            output_dir: 输出目录
        """
        # 创建输出目录
        Path(output_dir).mkdir(exist_ok=True)
        
        # 保存索引结构
        index_path = os.path.join(output_dir, "index_structure.json")
        with open(index_path, 'w', encoding='utf-8') as f:
            json.dump(self.index_structure, f, ensure_ascii=False, indent=2)
        
        logger.info(f"索引结构已保存到: {index_path}")
        
        # 保存每个章节的文本内容
        content_dir = os.path.join(output_dir, "sections")
        Path(content_dir).mkdir(exist_ok=True)
        
        for section_id, text_content in self.parsed_content.items():
            section_file = os.path.join(content_dir, f"{section_id}.txt")
            with open(section_file, 'w', encoding='utf-8') as f:
                f.write(text_content)
        
        logger.info(f"所有章节文本已保存到: {content_dir}")
        
        # 生成统计报告
        self._generate_statistics_report(output_dir)
    
    def _generate_statistics_report(self, output_dir: str):
        """生成统计报告"""
        stats = {
            'total_sections': len(self.index_structure),
            'sections_with_content': sum(1 for entry in self.index_structure.values() if entry['has_content']),
            'total_characters': sum(entry['text_length'] for entry in self.index_structure.values()),
            'total_words': sum(entry['word_count'] for entry in self.index_structure.values()),
            'level_distribution': {}
        }
        
        # 统计各级别章节数量
        for entry in self.index_structure.values():
            level = entry['level']
            stats['level_distribution'][level] = stats['level_distribution'].get(level, 0) + 1
        
        # 保存统计报告
        stats_path = os.path.join(output_dir, "statistics.json")
        with open(stats_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, ensure_ascii=False, indent=2)
        
        logger.info(f"统计报告已保存到: {stats_path}")
        logger.info(f"解析完成！共处理{stats['total_sections']}个章节，"
                   f"其中{stats['sections_with_content']}个包含内容")
    
    def get_section_text(self, section_id: str) -> str:
        """
        获取指定章节的文本内容
        
        Args:
            section_id: 章节ID
            
        Returns:
            章节文本内容
        """
        return self.parsed_content.get(section_id, "")
    
    def search_text(self, keyword: str) -> List[Dict[str, Any]]:
        """
        在所有章节中搜索关键词
        
        Args:
            keyword: 搜索关键词
            
        Returns:
            包含关键词的章节列表
        """
        results = []
        
        for section_id, text_content in self.parsed_content.items():
            if keyword.lower() in text_content.lower():
                index_entry = self.index_structure[section_id]
                results.append({
                    'section_id': section_id,
                    'title': index_entry['title'],
                    'start_page': index_entry['start_page'],
                    'end_page': index_entry['end_page'],
                    'text_preview': text_content[:200] + "..." if len(text_content) > 200 else text_content
                })
        
        return results


def main():
    """主函数示例"""
    # 配置路径（请根据实际情况修改）
    pdftotext_path = r"C:\Users\12879\Downloads\poppler-24.08.0\Library\bin\pdftotext.exe"
    pdf_path = "CCS ESC 指南.pdf" 
    toc_json_path = "ESC_content.json"
    
    try:
        # 创建解析器实例
        parser = MedicalGuideParser(pdftotext_path, pdf_path, toc_json_path)
        
        # 解析所有章节
        parser.parse_all_sections()
        
        # 保存结果
        parser.save_results("output")
        
        # 示例：搜索特定关键词
        search_results = parser.search_text("coronary")
        if search_results:
            print(f"\n找到{len(search_results)}个包含'coronary'的章节:")
            for result in search_results[:3]:  # 只显示前3个结果
                print(f"- {result['title']} (页面{result['start_page']}-{result['end_page']})")
        
        print("\n解析完成！请查看output目录中的结果文件。")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")


if __name__ == "__main__":
    main()