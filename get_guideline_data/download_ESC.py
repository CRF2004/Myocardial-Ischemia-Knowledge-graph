import os
import json
import time
from datetime import datetime
from typing import Dict, List, Any
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
from bs4 import BeautifulSoup
import requests
from urllib.parse import urljoin, urlparse

class ESCGuidelineScraper:
    """ESC指南结构化文本爬取器"""
    
    def __init__(self, save_dir: str = "esc_guideline_output"):
        """
        初始化爬取器
        
        Args:
            save_dir: 保存目录路径
        """
        self.save_dir = save_dir
        self.base_url = "https://academic.oup.com"
        self.content_structure = []
        self.images = []
        
        # 创建保存目录
        os.makedirs(save_dir, exist_ok=True)
        os.makedirs(os.path.join(save_dir, "images"), exist_ok=True)
    
    def setup_driver(self) -> webdriver.Chrome:
        """
        设置Chrome驱动，配置反检测参数
        
        Returns:
            配置好的Chrome驱动实例
        """
        options = webdriver.ChromeOptions()
        
        # 反检测设置
        options.add_argument('--disable-blink-features=AutomationControlled')
        options.add_experimental_option("excludeSwitches", ["enable-automation"])
        options.add_experimental_option('useAutomationExtension', False)
        
        # 其他有用的设置
        options.add_argument('--no-sandbox')
        options.add_argument('--disable-dev-shm-usage')
        options.add_argument('--disable-gpu')
        options.add_argument('--window-size=1920,1080')
        
        # 如果需要无头模式（不显示浏览器窗口），取消下面的注释
        # options.add_argument('--headless')
        
        driver = webdriver.Chrome(options=options)
        
        # 执行CDP命令来修改navigator.webdriver标志
        driver.execute_cdp_cmd('Page.addScriptToEvaluateOnNewDocument', {
            'source': 'Object.defineProperty(navigator, "webdriver", {get: () => undefined})'
        })
        
        return driver
    
    def handle_cloudflare(self, driver: webdriver.Chrome, url: str) -> bool:
        """
        处理Cloudflare人机验证
        
        Args:
            driver: Chrome驱动实例
            url: 目标URL
            
        Returns:
            是否成功通过验证
        """
        print(f"正在访问: {url}")
        driver.get(url)
        
        # 等待页面初步加载
        time.sleep(5)
        
        # 检查是否需要人工验证
        print("\n" + "="*60)
        print("⚠️  请检查浏览器窗口：")
        print("1. 如果看到Cloudflare验证页面，请手动完成验证")
        print("2. 如果看到'Verify you are human'，请点击复选框")
        print("3. 等待页面完全加载，直到看到文章内容")
        print("4. 完成后，在此处按Enter键继续...")
        print("="*60)
        input("按Enter键继续自动爬取 >>> ")
        
        # 给页面一些额外的加载时间
        time.sleep(2)
        
        try:
            # 尝试多种方式查找文章内容
            selectors = [
                (By.CLASS_NAME, "article-content"),
                (By.CLASS_NAME, "widget-items"),
                (By.XPATH, "//article"),
                (By.XPATH, "//div[contains(@class, 'content-block')]"),
                (By.XPATH, "//main"),
                (By.XPATH, "//div[@id='content']")
            ]
            
            content_found = False
            for selector_type, selector_value in selectors:
                elements = driver.find_elements(selector_type, selector_value)
                if elements:
                    print(f"✓ 找到内容区域: {selector_value}")
                    content_found = True
                    break
            
            if not content_found:
                # 让用户确认是否继续
                print("\n⚠️  未能自动检测到文章内容区域")
                print("请确认页面是否已完全加载，是否看到文章内容？")
                user_confirm = input("输入 'y' 继续爬取，输入 'n' 退出 (y/n): ").lower()
                
                if user_confirm == 'y':
                    return True
                else:
                    return False
            
            return True
            
        except Exception as e:
            print(f"检测页面内容时出错: {str(e)}")
            # 让用户决定是否继续
            user_confirm = input("是否仍要尝试继续爬取？(y/n): ").lower()
            return user_confirm == 'y'
    
    def parse_content(self, driver: webdriver.Chrome) -> Dict[str, Any]:
        """
        解析页面内容，提取结构化文本
        
        Args:
            driver: Chrome驱动实例
            
        Returns:
            结构化的内容字典
        """
        # 滚动页面以加载所有内容
        print("正在滚动页面以加载完整内容...")
        last_height = driver.execute_script("return document.body.scrollHeight")
        
        while True:
            # 向下滚动
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(2)
            
            # 检查是否已经滚动到底部
            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height
        
        # 滚回顶部
        driver.execute_script("window.scrollTo(0, 0);")
        time.sleep(1)
        
        # 获取页面源代码
        page_source = driver.page_source
        soup = BeautifulSoup(page_source, 'html.parser')
        
        # 尝试多种选择器查找文章主体
        article_content = None
        content_selectors = [
            {'class': 'article-content'},
            {'class': 'widget-items'},
            {'class': 'content-block'},
            {'id': 'content'},
            {'class': 'article-body'},
            {'class': 'full-text'}
        ]
        
        for selector in content_selectors:
            article_content = soup.find('div', selector)
            if article_content:
                print(f"使用选择器找到内容: {selector}")
                break
        
        # 如果还是找不到，尝试找article标签或main标签
        if not article_content:
            article_content = soup.find('article') or soup.find('main')
        
        if not article_content:
            print("警告：未找到明确的文章内容区域，将尝试解析整个body")
            article_content = soup.find('body')
        
        # 提取文章元数据
        metadata = self._extract_metadata(soup)
        
        # 递归解析内容结构
        content_tree = self._parse_hierarchical_content(article_content)
        
        # 提取图片
        self._extract_images(article_content)
        
        return {
            "metadata": metadata,
            "content": content_tree,
            "images": self.images,
            "stats": {
                "total_sections": self._count_sections(content_tree),
                "total_paragraphs": self._count_paragraphs(content_tree),
                "total_images": len(self.images)
            }
        }
    
    def _count_sections(self, content: List[Dict]) -> int:
        """统计章节数量"""
        count = 0
        for item in content:
            if item.get('type') == 'heading':
                count += 1
                if item.get('children'):
                    count += self._count_sections(item['children'])
        return count
    
    def _count_paragraphs(self, content: List[Dict]) -> int:
        """统计段落数量"""
        count = 0
        for item in content:
            if item.get('type') == 'paragraph':
                count += 1
            if item.get('children'):
                count += self._count_paragraphs(item['children'])
        return count
    
    def _extract_metadata(self, soup: BeautifulSoup) -> Dict[str, str]:
        """提取文章元数据"""
        metadata = {}
        
        # 标题
        title = soup.find('h1', class_='article-title')
        if title:
            metadata['title'] = title.get_text(strip=True)
        
        # 作者
        authors = soup.find_all('a', class_='author-name')
        if authors:
            metadata['authors'] = [author.get_text(strip=True) for author in authors]
        
        # DOI
        doi = soup.find('a', class_='doi')
        if doi:
            metadata['doi'] = doi.get_text(strip=True)
        
        # 发布日期
        pub_date = soup.find('div', class_='publication-date')
        if pub_date:
            metadata['publication_date'] = pub_date.get_text(strip=True)
        
        return metadata
    
    def _parse_hierarchical_content(self, element) -> List[Dict[str, Any]]:
        """
        递归解析层级化内容
        
        Args:
            element: BeautifulSoup元素
            
        Returns:
            层级化的内容列表
        """
        content = []
        current_section = None
        
        for child in element.children:
            if not hasattr(child, 'name'):
                continue
            
            # 处理标题标签
            if child.name in ['h1', 'h2', 'h3', 'h4', 'h5', 'h6']:
                level = int(child.name[1])
                section = {
                    'type': 'heading',
                    'level': level,
                    'text': child.get_text(strip=True),
                    'id': child.get('id', ''),
                    'children': []
                }
                
                # 根据层级关系组织内容
                if level == 1:
                    content.append(section)
                    current_section = section
                elif current_section:
                    self._add_to_appropriate_level(current_section, section, level)
                else:
                    content.append(section)
                    current_section = section
            
            # 处理段落
            elif child.name == 'p':
                paragraph = {
                    'type': 'paragraph',
                    'text': child.get_text(strip=True)
                }
                
                if current_section:
                    current_section['children'].append(paragraph)
                else:
                    content.append(paragraph)
            
            # 处理列表
            elif child.name in ['ul', 'ol']:
                list_items = []
                for li in child.find_all('li', recursive=False):
                    list_items.append(li.get_text(strip=True))
                
                list_content = {
                    'type': 'list',
                    'list_type': child.name,
                    'items': list_items
                }
                
                if current_section:
                    current_section['children'].append(list_content)
                else:
                    content.append(list_content)
            
            # 处理表格
            elif child.name == 'table':
                table_data = self._parse_table(child)
                if table_data:
                    if current_section:
                        current_section['children'].append(table_data)
                    else:
                        content.append(table_data)
            
            # 处理图片
            elif child.name == 'figure' or (child.name == 'div' and 'figure' in child.get('class', [])):
                figure_data = self._parse_figure(child)
                if figure_data:
                    if current_section:
                        current_section['children'].append(figure_data)
                    else:
                        content.append(figure_data)
        
        return content
    
    def _add_to_appropriate_level(self, parent: Dict, child: Dict, level: int):
        """根据层级关系添加子节点"""
        if level == 2:
            parent['children'].append(child)
        elif level > 2 and parent['children']:
            # 找到最后一个heading类型的子节点
            for item in reversed(parent['children']):
                if item.get('type') == 'heading' and item.get('level', 0) < level:
                    item.setdefault('children', []).append(child)
                    break
            else:
                parent['children'].append(child)
        else:
            parent['children'].append(child)
    
    def _parse_table(self, table_element) -> Dict[str, Any]:
        """解析表格"""
        headers = []
        rows = []
        
        # 提取表头
        thead = table_element.find('thead')
        if thead:
            for th in thead.find_all('th'):
                headers.append(th.get_text(strip=True))
        
        # 提取表格内容
        tbody = table_element.find('tbody')
        if tbody:
            for tr in tbody.find_all('tr'):
                row = []
                for td in tr.find_all(['td', 'th']):
                    row.append(td.get_text(strip=True))
                if row:
                    rows.append(row)
        
        # 提取表格标题
        caption = table_element.find('caption')
        table_caption = caption.get_text(strip=True) if caption else ""
        
        return {
            'type': 'table',
            'caption': table_caption,
            'headers': headers,
            'rows': rows
        }
    
    def _parse_figure(self, figure_element) -> Dict[str, Any]:
        """解析图片"""
        img = figure_element.find('img')
        if not img:
            return None
        
        img_src = img.get('src', '')
        img_alt = img.get('alt', '')
        
        # 获取图片标题
        figcaption = figure_element.find('figcaption')
        caption = figcaption.get_text(strip=True) if figcaption else ""
        
        # 构建完整URL
        if img_src and not img_src.startswith('http'):
            img_src = urljoin(self.base_url, img_src)
        
        figure_data = {
            'type': 'figure',
            'src': img_src,
            'alt': img_alt,
            'caption': caption
        }
        
        # 添加到图片列表
        self.images.append({
            'url': img_src,
            'caption': caption,
            'filename': f"image_{len(self.images) + 1}.jpg"
        })
        
        return figure_data
    
    def _extract_images(self, content_element):
        """提取所有图片"""
        images = content_element.find_all('img')
        for img in images:
            src = img.get('src', '')
            if src and not any(image['url'] == src for image in self.images):
                if not src.startswith('http'):
                    src = urljoin(self.base_url, src)
                
                self.images.append({
                    'url': src,
                    'caption': img.get('alt', ''),
                    'filename': f"image_{len(self.images) + 1}.jpg"
                })
    
    def download_images(self):
        """下载所有图片"""
        print(f"\n开始下载 {len(self.images)} 张图片...")
        
        for i, img_info in enumerate(self.images, 1):
            try:
                response = requests.get(img_info['url'], timeout=10)
                if response.status_code == 200:
                    filepath = os.path.join(self.save_dir, "images", img_info['filename'])
                    with open(filepath, 'wb') as f:
                        f.write(response.content)
                    print(f"  [{i}/{len(self.images)}] 下载成功: {img_info['filename']}")
                else:
                    print(f"  [{i}/{len(self.images)}] 下载失败: {img_info['url']}")
            except Exception as e:
                print(f"  [{i}/{len(self.images)}] 下载出错: {str(e)}")
    
    def save_results(self, data: Dict[str, Any]):
        """保存结果到文件"""
        # 保存JSON格式
        json_path = os.path.join(self.save_dir, "content_structure.json")
        with open(json_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        print(f"JSON结构已保存到: {json_path}")
        
        # 保存Markdown格式
        md_path = os.path.join(self.save_dir, "content.md")
        with open(md_path, 'w', encoding='utf-8') as f:
            f.write(self._convert_to_markdown(data))
        print(f"Markdown文档已保存到: {md_path}")
        
        # 保存纯文本格式（保留层级结构）
        txt_path = os.path.join(self.save_dir, "content_hierarchical.txt")
        with open(txt_path, 'w', encoding='utf-8') as f:
            f.write(self._convert_to_hierarchical_text(data))
        print(f"层级文本已保存到: {txt_path}")
    
    def _convert_to_markdown(self, data: Dict[str, Any]) -> str:
        """转换为Markdown格式"""
        md_lines = []
        
        # 添加元数据
        if 'metadata' in data:
            meta = data['metadata']
            if meta.get('title'):
                md_lines.append(f"# {meta['title']}\n")
            if meta.get('authors'):
                md_lines.append(f"**作者**: {', '.join(meta['authors'])}\n")
            if meta.get('doi'):
                md_lines.append(f"**DOI**: {meta['doi']}\n")
            if meta.get('publication_date'):
                md_lines.append(f"**发布日期**: {meta['publication_date']}\n")
            md_lines.append("\n---\n\n")
        
        # 添加内容
        if 'content' in data:
            md_lines.extend(self._content_to_markdown(data['content']))
        
        return '\n'.join(md_lines)
    
    def _content_to_markdown(self, content: List[Dict], base_level: int = 0) -> List[str]:
        """递归转换内容为Markdown"""
        lines = []
        
        for item in content:
            if item['type'] == 'heading':
                level = item['level'] + base_level
                lines.append(f"{'#' * level} {item['text']}\n")
                if item.get('children'):
                    lines.extend(self._content_to_markdown(item['children'], base_level))
            
            elif item['type'] == 'paragraph':
                lines.append(f"{item['text']}\n")
            
            elif item['type'] == 'list':
                for list_item in item['items']:
                    prefix = '-' if item['list_type'] == 'ul' else '1.'
                    lines.append(f"{prefix} {list_item}")
                lines.append("")
            
            elif item['type'] == 'table':
                if item.get('caption'):
                    lines.append(f"**表格**: {item['caption']}\n")
                if item.get('headers'):
                    lines.append('| ' + ' | '.join(item['headers']) + ' |')
                    lines.append('|' + '---|' * len(item['headers']))
                for row in item.get('rows', []):
                    lines.append('| ' + ' | '.join(row) + ' |')
                lines.append("")
            
            elif item['type'] == 'figure':
                lines.append(f"![{item.get('alt', 'Image')}]({item['src']})")
                if item.get('caption'):
                    lines.append(f"*{item['caption']}*\n")
        
        return lines
    
    def _convert_to_hierarchical_text(self, data: Dict[str, Any]) -> str:
        """转换为层级化纯文本"""
        lines = []
        
        # 添加元数据
        if 'metadata' in data:
            meta = data['metadata']
            if meta.get('title'):
                lines.append(f"标题: {meta['title']}")
            if meta.get('authors'):
                lines.append(f"作者: {', '.join(meta['authors'])}")
            if meta.get('doi'):
                lines.append(f"DOI: {meta['doi']}")
            if meta.get('publication_date'):
                lines.append(f"发布日期: {meta['publication_date']}")
            lines.append("\n" + "="*50 + "\n")
        
        # 添加内容
        if 'content' in data:
            lines.extend(self._content_to_hierarchical_text(data['content']))
        
        return '\n'.join(lines)
    
    def _content_to_hierarchical_text(self, content: List[Dict], indent: int = 0) -> List[str]:
        """递归转换内容为层级文本"""
        lines = []
        indent_str = "  " * indent
        
        for item in content:
            if item['type'] == 'heading':
                level_marker = "►" * item['level']
                lines.append(f"{indent_str}{level_marker} {item['text']}")
                if item.get('children'):
                    lines.extend(self._content_to_hierarchical_text(item['children'], indent + 1))
            
            elif item['type'] == 'paragraph':
                # 将长段落按80字符宽度换行
                text = item['text']
                while text:
                    lines.append(f"{indent_str}{text[:80]}")
                    text = text[80:]
            
            elif item['type'] == 'list':
                for i, list_item in enumerate(item['items'], 1):
                    prefix = "•" if item['list_type'] == 'ul' else f"{i}."
                    lines.append(f"{indent_str}{prefix} {list_item}")
            
            elif item['type'] == 'table':
                if item.get('caption'):
                    lines.append(f"{indent_str}[表格] {item['caption']}")
                # 简化表格显示
                lines.append(f"{indent_str}  (包含 {len(item.get('rows', []))} 行数据)")
            
            elif item['type'] == 'figure':
                lines.append(f"{indent_str}[图片] {item.get('caption', '无标题')}")
        
        return lines
    
    def run(self, url: str):
        """
        执行完整的爬取流程
        
        Args:
            url: 目标文章URL
        """
        driver = None
        try:
            # 设置驱动
            print("正在启动浏览器...")
            driver = self.setup_driver()
            
            # 处理Cloudflare验证并加载页面
            if not self.handle_cloudflare(driver, url):
                print("无法加载页面内容")
                return
            
            # 解析内容
            print("\n正在解析页面内容...")
            data = self.parse_content(driver)
            
            if not data or not data.get('content'):
                print("未能解析到有效内容")
                print("尝试保存页面源代码供调试...")
                with open(os.path.join(self.save_dir, "page_source.html"), 'w', encoding='utf-8') as f:
                    f.write(driver.page_source)
                print(f"页面源代码已保存到: {os.path.join(self.save_dir, 'page_source.html')}")
                return
            
            # 显示解析统计
            if 'stats' in data:
                print(f"\n📊 解析统计:")
                print(f"  - 章节数: {data['stats']['total_sections']}")
                print(f"  - 段落数: {data['stats']['total_paragraphs']}")
                print(f"  - 图片数: {data['stats']['total_images']}")
            
            # 保存结果
            print("\n正在保存结果...")
            self.save_results(data)
            
            # 下载图片
            if self.images:
                user_choice = input("\n是否下载图片？(y/n): ").lower()
                if user_choice == 'y':
                    self.download_images()
            
            print(f"\n✅ 所有内容已保存到: {self.save_dir}")
            print(f"  - JSON结构: content_structure.json")
            print(f"  - Markdown文档: content.md")
            print(f"  - 层级文本: content_hierarchical.txt")
            if self.images and user_choice == 'y':
                print(f"  - 图片: images/")
            
        except Exception as e:
            print(f"❌ 发生错误: {str(e)}")
            import traceback
            traceback.print_exc()
            
            # 保存错误时的页面源代码
            if driver:
                try:
                    error_html_path = os.path.join(self.save_dir, "error_page.html")
                    with open(error_html_path, 'w', encoding='utf-8') as f:
                        f.write(driver.page_source)
                    print(f"错误页面已保存到: {error_html_path}")
                except:
                    pass
        
        finally:
            if driver:
                input("\n按Enter键关闭浏览器...")
                driver.quit()


def main():
    """主函数"""
    # 目标URL 2024 ESC Guidelines for the management of chronic coronary syndromes: Developed by the task force for the management of chronic coronary syndromes of the European Society of Cardiology (ESC) Endorsed by the European Association for Cardio-Thoracic Surgery (EACTS)
    url = "https://academic.oup.com/eurheartj/article/45/36/3415/7743115?login=false"
    
    # 创建爬取器实例
    scraper = ESCGuidelineScraper(save_dir="esc_guideline_2024")
    
    # 执行爬取
    print("="*60)
    print("ESC指南结构化文本爬取程序")
    print("="*60)
    print(f"目标URL: {url}")
    print(f"保存目录: {scraper.save_dir}")
    print("="*60)
    
    scraper.run(url)


if __name__ == "__main__":
    main()