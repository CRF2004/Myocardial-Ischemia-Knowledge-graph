import time
from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from webdriver_manager.chrome import ChromeDriverManager
from bs4 import BeautifulSoup

# 目标 URL
URL = "https://www.ahajournals.org/doi/10.1161/CIR.0000000000001309"

# 1）配置 ChromeOptions
options = Options()
options.add_argument("--headless")  # 启用无头模式
options.add_argument("--ignore-certificate-errors")
options.add_argument("--ignore-ssl-errors")
options.add_argument("--allow-running-insecure-content")
options.add_argument("--disable-web-security")
options.add_argument("--disable-gpu")
options.add_argument("--window-size=1920,1080")
options.add_argument("--log-level=3")
options.set_capability("acceptInsecureCerts", True)

# 设置请求头，模拟普通浏览器请求
options.add_argument("user-agent=Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36")

# 2）启动 ChromeDriver
service = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=service, options=options)

try:
    # 3）加载网页
    driver.get(URL)
    
    # 等待页面加载完成
    WebDriverWait(driver, 15).until(
        EC.presence_of_element_located((By.TAG_NAME, "body"))
    )

    # 4）抓取页面内容
    body_content = driver.find_element(By.TAG_NAME, "body").get_attribute("outerHTML")

    # 使用 BeautifulSoup 提取纯文本
    soup = BeautifulSoup(body_content, "html.parser")
    
    # 5）找到所有带 #sec 的链接
    sec_links = soup.find_all("a", href=True)
    sec_links = [a['href'] for a in sec_links if '#sec' in a['href']]

    # 6）遍历所有的 #sec 链接并抓取该部分的正文
    filtered_content = []
    for link in sec_links:
        # 获取包含正文的部分
        sec_id = link.split("#")[-1]  # 获取#sec后的部分，作为定位标识
        sec_element = soup.find(id=sec_id)  # 根据id查找对应部分
        if sec_element:
            filtered_content.append(sec_element.get_text())

    # 7）保存正文内容到 .txt 文件
    with open("acc_guidelines.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(filtered_content))

    print("[✔] 页面带 #sec 的内容已保存到 acc_guidelines.txt")

finally:
    driver.quit()
