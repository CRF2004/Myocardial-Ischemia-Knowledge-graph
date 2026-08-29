import os
import json
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

@dataclass
class Chapter:
    """章节数据类"""
    id: str
    title: str
    start_page: int
    end_page: int
    level: int
    parent_id: Optional[str]
    text: str
    children_ids: List[str] = None
    
    def __post_init__(self):
        if self.children_ids is None:
            self.children_ids = []
    
    def to_dict(self):
        """转换为字典，不包含children_ids"""
        result = {
            "id": self.id,
            "title": self.title,
            "start_page": self.start_page,
            "end_page": self.end_page,
            "level": self.level,
            "parent_id": self.parent_id,
            "text": self.text
        }
        return result


class MarkdownToJsonParser:
    """Markdown转结构化JSON解析器"""
    
    def __init__(self, markdown_file: str, output_dir: str = "structured_output"):
        """
        初始化解析器
        
        Args:
            markdown_file: Markdown文件路径
            output_dir: 输出目录
        """
        self.markdown_file = markdown_file
        self.output_dir = output_dir
        self.chapters = []
        self.chapter_counter = 0
        self.current_page = 1  # 模拟页码
        
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)
    
    def parse_markdown(self) -> List[Chapter]:
        """解析Markdown文件"""
        print("\n" + "="*60)
        print("开始解析Markdown文件...")
        print("="*60)
        
        with open(self.markdown_file, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # 分割成行
        lines = content.split('\n')
        
        # 当前层级的父章节
        parent_stack = {
            0: None,  # 顶级没有父节点
            1: None,
            2: None,
            3: None,
            4: None,
            5: None,
            6: None
        }
        
        current_chapter = None
        current_text_lines = []
        
        i = 0
        while i < len(lines):
            line = lines[i]
            
            # 检查是否是标题行
            heading_match = re.match(r'^(#{1,6})\s+(.+)$', line)
            
            if heading_match:
                # 如果有当前章节，保存其文本
                if current_chapter:
                    current_chapter.text = '\n'.join(current_text_lines).strip()
                    current_text_lines = []
                
                # 解析标题
                level = len(heading_match.group(1))
                title = heading_match.group(2).strip()
                
                # 生成章节ID
                self.chapter_counter += 1
                chapter_id = self._generate_chapter_id(title, self.chapter_counter)
                
                # 确定父节点
                parent_id = None
                if level > 1:
                    # 查找最近的上级章节
                    for parent_level in range(level - 1, 0, -1):
                        if parent_stack[parent_level]:
                            parent_id = parent_stack[parent_level].id
                            parent_stack[parent_level].children_ids.append(chapter_id)
                            break
                
                # 估算页码（这里使用简单的规则，实际可能需要调整）
                start_page = self._estimate_page_number(i, len(lines))
                
                # 创建新章节
                current_chapter = Chapter(
                    id=chapter_id,
                    title=title,
                    start_page=start_page,
                    end_page=start_page,  # 稍后更新
                    level=level,
                    parent_id=parent_id,
                    text=""
                )
                
                # 更新父节点栈
                parent_stack[level] = current_chapter
                # 清除更深层级的父节点
                for deeper_level in range(level + 1, 7):
                    parent_stack[deeper_level] = None
                
                self.chapters.append(current_chapter)
                
            else:
                # 非标题行，添加到当前章节的文本中
                if line.strip():  # 忽略空行
                    # 处理列表项
                    if line.strip().startswith(('-', '*', '+')):
                        # 无序列表
                        current_text_lines.append(line.strip())
                    elif re.match(r'^\d+\.', line.strip()):
                        # 有序列表
                        current_text_lines.append(line.strip())
                    else:
                        # 普通段落
                        current_text_lines.append(line.strip())
            
            i += 1
        
        # 保存最后一个章节的文本
        if current_chapter and current_text_lines:
            current_chapter.text = '\n'.join(current_text_lines).strip()
        
        # 更新结束页码
        self._update_end_pages()
        
        return self.chapters
    
    def _generate_chapter_id(self, title: str, counter: int) -> str:
        """生成章节ID"""
        # 尝试提取章节编号
        number_match = re.match(r'^(\d+(?:\.\d+)*)', title)
        
        if number_match:
            # 使用章节编号
            chapter_num = number_match.group(1).replace('.', '_')
            return f"chapter_{chapter_num}"
        else:
            # 使用计数器
            return f"chapter_{counter:02d}"
    
    def _estimate_page_number(self, line_index: int, total_lines: int) -> int:
        """估算页码（简单的线性映射）"""
        # 假设文档大约有100页
        # 这里可以根据实际情况调整
        estimated_total_pages = 100
        
        if total_lines == 0:
            return 1
        
        page = int((line_index / total_lines) * estimated_total_pages) + 1
        return max(1, min(page, estimated_total_pages))
    
    def _update_end_pages(self):
        """更新每个章节的结束页码"""
        for i in range(len(self.chapters)):
            if i < len(self.chapters) - 1:
                # 结束页码是下一章节的开始页码 - 1
                self.chapters[i].end_page = max(
                    self.chapters[i].start_page,
                    self.chapters[i + 1].start_page - 1
                )
            else:
                # 最后一个章节，估算结束页码
                self.chapters[i].end_page = self.chapters[i].start_page + 2
    
    def create_hierarchical_structure(self) -> Dict:
        """创建层级结构（用于可视化）"""
        root = {
            "title": "ESC Guidelines",
            "children": []
        }
        
        # 构建ID到节点的映射
        nodes = {}
        
        for chapter in self.chapters:
            node = {
                "id": chapter.id,
                "title": chapter.title,
                "level": chapter.level,
                "pages": f"{chapter.start_page}-{chapter.end_page}",
                "text_length": len(chapter.text),
                "children": []
            }
            nodes[chapter.id] = node
            
            if chapter.parent_id and chapter.parent_id in nodes:
                nodes[chapter.parent_id]["children"].append(node)
            elif chapter.level == 1:
                root["children"].append(node)
        
        return root
    
    def save_results(self):
        """保存解析结果"""
        # 1. 保存扁平化的JSON列表（你要求的格式）
        flat_json_path = os.path.join(self.output_dir, "chapters_flat.json")
        flat_data = [chapter.to_dict() for chapter in self.chapters]
        
        with open(flat_json_path, 'w', encoding='utf-8') as f:
            json.dump(flat_data, f, ensure_ascii=False, indent=4)
        print(f"\n✓ 扁平化JSON已保存到: {flat_json_path}")
        
        # 2. 保存层级结构（便于可视化）
        hierarchy_json_path = os.path.join(self.output_dir, "chapters_hierarchy.json")
        hierarchy_data = self.create_hierarchical_structure()
        
        with open(hierarchy_json_path, 'w', encoding='utf-8') as f:
            json.dump(hierarchy_data, f, ensure_ascii=False, indent=4)
        print(f"✓ 层级结构JSON已保存到: {hierarchy_json_path}")
        
        # 3. 保存章节摘要
        self.save_summary()
        
        # 4. 保存纯文本版本（每个章节单独的文件）
        self.save_text_files()
    
    def save_summary(self):
        """保存章节摘要"""
        summary_path = os.path.join(self.output_dir, "chapters_summary.txt")
        
        with open(summary_path, 'w', encoding='utf-8') as f:
            f.write("ESC指南章节摘要\n")
            f.write("="*60 + "\n\n")
            
            # 统计信息
            total_chapters = len(self.chapters)
            level_counts = {}
            for chapter in self.chapters:
                level_counts[chapter.level] = level_counts.get(chapter.level, 0) + 1
            
            f.write(f"总章节数: {total_chapters}\n")
            for level in sorted(level_counts.keys()):
                f.write(f"  - 第{level}级标题: {level_counts[level]}个\n")
            f.write("\n" + "-"*60 + "\n\n")
            
            # 章节列表
            f.write("章节结构:\n\n")
            for chapter in self.chapters:
                indent = "  " * (chapter.level - 1)
                text_preview = chapter.text[:100] + "..." if len(chapter.text) > 100 else chapter.text
                text_preview = text_preview.replace('\n', ' ')
                
                f.write(f"{indent}[{chapter.id}] {chapter.title}\n")
                f.write(f"{indent}  页码: {chapter.start_page}-{chapter.end_page}\n")
                f.write(f"{indent}  文本长度: {len(chapter.text)} 字符\n")
                if text_preview:
                    f.write(f"{indent}  预览: {text_preview}\n")
                f.write("\n")
        
        print(f"✓ 章节摘要已保存到: {summary_path}")
    
    def save_text_files(self):
        """保存每个章节的纯文本文件"""
        text_dir = os.path.join(self.output_dir, "chapter_texts")
        os.makedirs(text_dir, exist_ok=True)
        
        for chapter in self.chapters:
            if chapter.text:  # 只保存有内容的章节
                filename = f"{chapter.id}.txt"
                filepath = os.path.join(text_dir, filename)
                
                with open(filepath, 'w', encoding='utf-8') as f:
                    f.write(f"章节: {chapter.title}\n")
                    f.write(f"级别: {chapter.level}\n")
                    f.write(f"页码: {chapter.start_page}-{chapter.end_page}\n")
                    if chapter.parent_id:
                        parent = next((c for c in self.chapters if c.id == chapter.parent_id), None)
                        if parent:
                            f.write(f"父章节: {parent.title}\n")
                    f.write("\n" + "="*60 + "\n\n")
                    f.write(chapter.text)
        
        print(f"✓ 章节文本文件已保存到: {text_dir}/")
    
    def print_statistics(self):
        """打印统计信息"""
        print("\n" + "="*60)
        print("解析统计")
        print("="*60)
        
        total_chapters = len(self.chapters)
        total_text_length = sum(len(c.text) for c in self.chapters)
        chapters_with_text = sum(1 for c in self.chapters if c.text)
        
        print(f"总章节数: {total_chapters}")
        print(f"包含文本的章节: {chapters_with_text}")
        print(f"总文本长度: {total_text_length:,} 字符")
        
        # 按级别统计
        level_stats = {}
        for chapter in self.chapters:
            if chapter.level not in level_stats:
                level_stats[chapter.level] = {
                    'count': 0,
                    'total_text': 0,
                    'chapters': []
                }
            level_stats[chapter.level]['count'] += 1
            level_stats[chapter.level]['total_text'] += len(chapter.text)
            level_stats[chapter.level]['chapters'].append(chapter.title[:50])
        
        print("\n按级别统计:")
        for level in sorted(level_stats.keys()):
            stats = level_stats[level]
            print(f"\n第{level}级标题:")
            print(f"  数量: {stats['count']}")
            print(f"  总文本: {stats['total_text']:,} 字符")
            print(f"  前3个标题:")
            for title in stats['chapters'][:3]:
                print(f"    - {title}")
        
        # 找出最长的章节
        if self.chapters:
            longest_chapter = max(self.chapters, key=lambda c: len(c.text))
            print(f"\n最长章节:")
            print(f"  标题: {longest_chapter.title}")
            print(f"  文本长度: {len(longest_chapter.text):,} 字符")
    
    def run(self):
        """执行解析流程"""
        try:
            print("\n" + "="*60)
            print("Markdown转结构化JSON解析器")
            print("="*60)
            print(f"输入文件: {self.markdown_file}")
            print(f"输出目录: {self.output_dir}")
            
            # 检查文件是否存在
            if not os.path.exists(self.markdown_file):
                print(f"❌ 文件不存在: {self.markdown_file}")
                return
            
            # 解析Markdown
            self.parse_markdown()
            
            # 打印统计信息
            self.print_statistics()
            
            # 保存结果
            print("\n" + "="*60)
            print("保存结果...")
            print("="*60)
            self.save_results()
            
            print("\n✅ 解析完成！")
            print(f"所有文件已保存到: {self.output_dir}/")
            print("\n生成的文件:")
            print("  1. chapters_flat.json - 扁平化的章节列表（你要求的格式）")
            print("  2. chapters_hierarchy.json - 层级结构（便于可视化）")
            print("  3. chapters_summary.txt - 章节摘要")
            print("  4. chapter_texts/ - 每个章节的独立文本文件")
            
        except Exception as e:
            print(f"\n❌ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()


def main():
    """主函数"""
    import sys
    
    # 可以通过命令行参数指定Markdown文件
    if len(sys.argv) > 1:
        markdown_file = sys.argv[1]
    else:
        # 默认查找之前生成的Markdown文件
        possible_files = [
            "esc_parsed_output\parsed_content.md"
        ]
        
        markdown_file = None
        for file in possible_files:
            if os.path.exists(file):
                markdown_file = file
                print(f"找到Markdown文件: {file}")
                break
        
        if not markdown_file:
            print("请指定Markdown文件路径:")
            print("用法: python markdown_to_json_parser.py <markdown_file_path>")
            return
    
    # 创建解析器并运行
    parser = MarkdownToJsonParser(markdown_file, output_dir="structured_output")
    parser.run()


if __name__ == "__main__":
    main()