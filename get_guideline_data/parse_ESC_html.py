import os
import json
import re
from bs4 import BeautifulSoup, NavigableString
from typing import Dict, List, Any

class ESCDebugParser:
    """ESC指南HTML调试解析器"""
    
    def __init__(self, html_file: str, save_dir: str = "esc_parsed_output"):
        """
        初始化解析器
        
        Args:
            html_file: HTML文件路径
            save_dir: 保存目录路径
        """
        self.html_file = html_file
        self.save_dir = save_dir
        self.content_structure = []
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
    
    def analyze_html_structure(self):
        """分析HTML结构，找出内容所在的位置"""
        print("\n" + "="*60)
        print("开始分析HTML结构...")
        print("="*60)
        
        with open(self.html_file, 'r', encoding='utf-8', errors='ignore') as f:
            html_content = f.read()
        
        # 尝试修复可能的HTML问题
        html_content = self._fix_html_issues(html_content)
        
        # 使用不同的解析器尝试
        parsers = ['html.parser', 'lxml', 'html5lib']
        soup = None
        
        for parser in parsers:
            try:
                print(f"\n尝试使用 {parser} 解析器...")
                soup = BeautifulSoup(html_content, parser)
                print(f"✓ {parser} 解析成功")
                break
            except Exception as e:
                print(f"✗ {parser} 解析失败: {str(e)}")
                continue
        
        if not soup:
            print("所有解析器都失败了，尝试使用容错模式...")
            soup = BeautifulSoup(html_content, 'html.parser', from_encoding='utf-8')
        
        # 分析标题结构
        print("\n" + "-"*40)
        print("查找所有标题元素...")
        headings = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6'])
        print(f"找到 {len(headings)} 个标题")
        
        if headings:
            print("\n前10个标题示例：")
            for i, h in enumerate(headings[:10], 1):
                text = h.get_text(strip=True)[:80]
                print(f"  {i}. <{h.name}>: {text}...")
        
        # 查找可能的内容容器
        print("\n" + "-"*40)
        print("查找内容容器...")
        
        potential_containers = [
            ('class', 'article-content'),
            ('class', 'widget-items'),
            ('class', 'chapter-para'),
            ('class', 'section'),
            ('class', 'content'),
            ('class', 'main-content'),
            ('class', 'full-text'),
            ('class', 'article-body'),
            ('id', 'content'),
            ('id', 'main'),
            ('role', 'main'),
            ('tag', 'article'),
            ('tag', 'main')
        ]
        
        found_containers = []
        for attr_type, attr_value in potential_containers:
            if attr_type == 'class':
                elements = soup.find_all(class_=attr_value)
            elif attr_type == 'id':
                element = soup.find(id=attr_value)
                elements = [element] if element else []
            elif attr_type == 'role':
                elements = soup.find_all(attrs={'role': attr_value})
            elif attr_type == 'tag':
                elements = soup.find_all(attr_value)
            
            if elements:
                for elem in elements:
                    text_length = len(elem.get_text(strip=True))
                    if text_length > 100:  # 只考虑有实质内容的容器
                        found_containers.append({
                            'type': attr_type,
                            'value': attr_value,
                            'text_length': text_length,
                            'element': elem
                        })
                        print(f"✓ 找到 {attr_type}='{attr_value}' (文本长度: {text_length})")
        
        # 选择最佳容器
        if found_containers:
            # 按文本长度排序，选择最长的
            found_containers.sort(key=lambda x: x['text_length'], reverse=True)
            best_container = found_containers[0]
            print(f"\n选择最佳容器: {best_container['type']}='{best_container['value']}'")
            return soup, best_container['element']
        else:
            print("\n未找到明确的内容容器，将使用body标签")
            return soup, soup.find('body')
    
    def _fix_html_issues(self, html_content: str) -> str:
        """修复常见的HTML问题"""
        # 移除可能导致问题的特殊字符
        html_content = html_content.replace('\x00', '')
        
        # 修复未闭合的标签（简单处理）
        # 这里可以根据需要添加更多修复逻辑
        
        return html_content
    
    def parse_content_from_html(self, container_element) -> List[Dict[str, Any]]:
        """从HTML容器元素中解析内容"""
        content = []
        current_h1 = None
        current_h2 = None
        current_h3 = None
        current_h4 = None
        current_h5 = None
        
        # 遍历所有子元素
        for element in container_element.descendants:
            if isinstance(element, NavigableString):
                continue
            
            if element.name == 'h1':
                current_h1 = {
                    'type': 'heading',
                    'level': 1,
                    'text': element.get_text(strip=True),
                    'children': []
                }
                content.append(current_h1)
                # 重置下级标题
                current_h2 = None
                current_h3 = None
                current_h4 = None
                current_h5 = None
                
            elif element.name == 'h2':
                current_h2 = {
                    'type': 'heading',
                    'level': 2,
                    'text': element.get_text(strip=True),
                    'children': []
                }
                if current_h1:
                    current_h1['children'].append(current_h2)
                else:
                    content.append(current_h2)
                # 重置下级标题
                current_h3 = None
                current_h4 = None
                current_h5 = None
                
            elif element.name == 'h3':
                current_h3 = {
                    'type': 'heading',
                    'level': 3,
                    'text': element.get_text(strip=True),
                    'children': []
                }
                if current_h2:
                    current_h2['children'].append(current_h3)
                elif current_h1:
                    current_h1['children'].append(current_h3)
                else:
                    content.append(current_h3)
                # 重置下级标题
                current_h4 = None
                current_h5 = None
            
            elif element.name == 'h4':
                current_h4 = {  # 修复：这里应该是 current_h4，不是 current_h3
                    'type': 'heading',
                    'level': 4,
                    'text': element.get_text(strip=True),
                    'children': []
                }
                if current_h3:
                    current_h3['children'].append(current_h4)
                elif current_h2:
                    current_h2['children'].append(current_h4)
                elif current_h1:
                    current_h1['children'].append(current_h4)
                else:
                    content.append(current_h4)
                # 重置下级标题
                current_h5 = None
                    
            elif element.name == 'h5':
                current_h5 = {  # 修复：这里应该是 current_h5，不是 current_h3
                    'type': 'heading',
                    'level': 5,
                    'text': element.get_text(strip=True),
                    'children': []
                }
                if current_h4:
                    current_h4['children'].append(current_h5)
                elif current_h3:
                    current_h3['children'].append(current_h5)
                elif current_h2:
                    current_h2['children'].append(current_h5)
                elif current_h1:
                    current_h1['children'].append(current_h5)
                else:
                    content.append(current_h5)
                    
            elif element.name == 'h6':  # 添加 h6 支持
                current_h6 = {
                    'type': 'heading',
                    'level': 6,
                    'text': element.get_text(strip=True),
                    'children': []
                }
                if current_h5:
                    current_h5['children'].append(current_h6)
                elif current_h4:
                    current_h4['children'].append(current_h6)
                elif current_h3:
                    current_h3['children'].append(current_h6)
                elif current_h2:
                    current_h2['children'].append(current_h6)
                elif current_h1:
                    current_h1['children'].append(current_h6)
                else:
                    content.append(current_h6)
                    
            elif element.name == 'p':
                # 检查是否是真正的段落（不是嵌套在其他p标签内的）
                if not element.find_parent('p'):
                    para_text = element.get_text(strip=True)
                    if para_text:  # 只添加非空段落
                        paragraph = {
                            'type': 'paragraph',
                            'text': para_text
                        }
                        
                        # 添加到适当的层级
                        if current_h5:
                            current_h5['children'].append(paragraph)
                        elif current_h4:
                            current_h4['children'].append(paragraph)
                        elif current_h3:
                            current_h3['children'].append(paragraph)
                        elif current_h2:
                            current_h2['children'].append(paragraph)
                        elif current_h1:
                            current_h1['children'].append(paragraph)
                        else:
                            content.append(paragraph)
                            
            elif element.name in ['ul', 'ol']:
                # 检查是否已经处理过（避免嵌套列表重复）
                if not element.find_parent(['ul', 'ol']):
                    list_items = []
                    for li in element.find_all('li', recursive=False):
                        list_items.append(li.get_text(strip=True))
                    
                    if list_items:
                        list_content = {
                            'type': 'list',
                            'list_type': element.name,
                            'items': list_items
                        }
                        
                        # 添加到适当的层级
                        if current_h5:
                            current_h5['children'].append(list_content)
                        elif current_h4:
                            current_h4['children'].append(list_content)
                        elif current_h3:
                            current_h3['children'].append(list_content)
                        elif current_h2:
                            current_h2['children'].append(list_content)
                        elif current_h1:
                            current_h1['children'].append(list_content)
                        else:
                            content.append(list_content)
        
        return content
    
    def extract_all_text(self, container_element) -> str:
        """提取所有纯文本（作为备用方案）"""
        # 移除script和style标签
        for script in container_element(['script', 'style']):
            script.decompose()
        
        # 获取所有文本
        text = container_element.get_text()
        
        # 清理文本
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return '\n'.join(lines)
    
    def save_results(self, content: List[Dict[str, Any]], raw_text: str = None):
        """保存解析结果"""
        # 保存JSON结构
        json_path = os.path.join(self.save_dir, "parsed_structure.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(content, f, ensure_ascii=False, indent=2)
        print(f"\n✓ JSON结构已保存到: {json_path}")
        
        # 保存Markdown
        md_content = self._convert_to_markdown(content)
        md_path = os.path.join(self.save_dir, "parsed_content.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(md_content)
        print(f"✓ Markdown已保存到: {md_path}")
        
        # 如果有原始文本，也保存
        if raw_text:
            txt_path = os.path.join(self.save_dir, "raw_text.txt")
            with open(txt_path, 'w', encoding='utf-8') as f:
                f.write(raw_text)
            print(f"✓ 原始文本已保存到: {txt_path}")
    
    def _convert_to_markdown(self, content: List[Dict]) -> str:
        """转换为Markdown格式"""
        lines = []
        
        def process_items(items, base_level=0):
            for item in items:
                if item['type'] == 'heading':
                    level = item['level'] + base_level
                    lines.append(f"\n{'#' * level} {item['text']}\n")
                    if item.get('children'):
                        process_items(item['children'], base_level)
                        
                elif item['type'] == 'paragraph':
                    lines.append(f"{item['text']}\n")
                    
                elif item['type'] == 'list':
                    for list_item in item['items']:
                        prefix = '-' if item['list_type'] == 'ul' else '1.'
                        lines.append(f"{prefix} {list_item}")
                    lines.append("")
        
        process_items(content)
        return '\n'.join(lines)
    
    def print_structure_summary(self, content: List[Dict[str, Any]]):
        """打印结构摘要，便于调试"""
        def count_by_level(items, level_counts=None):
            if level_counts is None:
                level_counts = {}
            
            for item in items:
                if item['type'] == 'heading':
                    level = item['level']
                    level_counts[level] = level_counts.get(level, 0) + 1
                
                if item.get('children'):
                    count_by_level(item['children'], level_counts)
            
            return level_counts
        
        level_counts = count_by_level(content)
        print(f"\n标题层级统计:")
        for level in sorted(level_counts.keys()):
            print(f"  H{level}: {level_counts[level]} 个")
    
    def run(self):
        """执行解析流程"""
        try:
            print("\n" + "="*60)
            print("ESC指南HTML调试解析器")
            print("="*60)
            
            # 检查文件是否存在
            if not os.path.exists(self.html_file):
                print(f"❌ 文件不存在: {self.html_file}")
                return
            
            file_size = os.path.getsize(self.html_file) / 1024 / 1024  # MB
            print(f"HTML文件: {self.html_file}")
            print(f"文件大小: {file_size:.2f} MB")
            
            # 分析HTML结构
            soup, container = self.analyze_html_structure()
            
            if not container:
                print("❌ 无法找到内容容器")
                return
            
            # 解析内容
            print("\n" + "="*60)
            print("开始解析内容...")
            print("="*60)
            
            # 方法1：结构化解析
            structured_content = self.parse_content_from_html(container)
            
            # 方法2：提取原始文本（备用）
            raw_text = self.extract_all_text(container)
            
            # 统计信息
            total_headings = sum(1 for item in self._flatten_content(structured_content) 
                               if item.get('type') == 'heading')
            total_paragraphs = sum(1 for item in self._flatten_content(structured_content) 
                                 if item.get('type') == 'paragraph')
            
            print(f"\n解析统计:")
            print(f"  - 标题数: {total_headings}")
            print(f"  - 段落数: {total_paragraphs}")
            print(f"  - 原始文本长度: {len(raw_text)} 字符")
            
            # 打印结构摘要
            self.print_structure_summary(structured_content)
            
            # 保存结果
            self.save_results(structured_content, raw_text)
            
            print(f"\n✅ 解析完成！结果已保存到: {self.save_dir}")
            
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def _flatten_content(self, content: List[Dict]) -> List[Dict]:
        """展平嵌套内容结构"""
        flat = []
        for item in content:
            flat.append(item)
            if item.get('children'):
                flat.extend(self._flatten_content(item['children']))
        return flat


def main():
    """主函数"""
    import sys
    
    # 可以通过命令行参数指定HTML文件
    if len(sys.argv) > 1:
        html_file = sys.argv[1]
    else:
        # 默认查找之前保存的文件
        possible_files = [
            "esc_parsed_output\page_source.html",
        ]
        
        html_file = None
        for file in possible_files:
            if os.path.exists(file):
                html_file = file
                break
        
        if not html_file:
            print("请指定HTML文件路径:")
            print("用法: python esc_debug_parser.py <html_file_path>")
            return
    
    # 创建解析器并运行
    parser = ESCDebugParser(html_file, save_dir="esc_parsed_output")
    parser.run()


if __name__ == "__main__":
    main()