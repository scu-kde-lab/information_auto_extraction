# 安装模块
!pip install revChatGPT
!pip install openai
!pip install langchain
!pip install unstructured
!pip install pdf2image
!pip install pytesseract
!pip install PyMuPDF
!pip install neo4j

import openai
import os
import json
import time

#open api
openai.api_key = "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"
openai.api_base = "https://ai.tudb.work/v1"


from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI

# 文件切割
def split_text(file_name, chunk_size=1500, chunk_overlap=200):
    loader = UnstructuredFileLoader(file_name) 
    # 将文本转成 Document 对象
    document = loader.load()
    print(f'documents:{len(document)}')
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    # 切分文本
    split_documents = text_splitter.split_documents(document)
    print(f'documents:{len(split_documents)}')
    return split_documents
  
  # 文件处理函数
 import re
import fitz

def preprocess(text):
    text = text.replace('\n', '')
    text = re.sub('\s+', ' ', text)
    return text


def pdf_to_text(path, start_page=1, end_page=None):
    doc = fitz.open(path)
    total_pages = doc.page_count

    if end_page is None:
        end_page = total_pages
    text_list = []

    for i in range(start_page - 1, end_page):
        text = doc.load_page(i).get_text("text")
        text = preprocess(text)
        text_list.append(text)

    doc.close()
    return "".join(text_list)

def write_to_file(text, filename):
    with open(filename, 'w') as f:
        f.write(text)

def read_file(file_path):
    with open(file_path, 'r') as file:
        file_content = file.read()
    return file_content
  
# open请求函数
def get_completion(prompt, model="gpt-3.5-turbo"):
    # print("=========To OpenAI=========")
    start_time = time.time()
    try:
        messages = [{"role": "user", "content": prompt}]
        response = openai.ChatCompletion.create(
        model=model,
        messages=messages,
        temperature=0, # this is the degree of randomness of the model's output
        )
        response = response.choices[0].message["content"]
        # print(response)
        # raise json.JSONDecodeError

    except openai.error.APIError as e:
        print("=====APIError=====")
        print('Error:', e)
        response = "APIError occurred!"
    except openai.error.RateLimitError:
        print("=====RateLimitError!=====")
        response = "RateLimit occurred!"
    # Print execution time
    end_time = time.time()
    print("Open API call took %.2f seconds" % (end_time - start_time))
    # print("=============================")
    return response

def flow_get_completion(prompt, model="gpt-3.5-turbo"):
    start_time = time.time()
    string = ""
    try:
        for chunk in openai.ChatCompletion.create(
                model=model,
                messages = [{
                    "role": "user",
                    "content": prompt
                }],
                temperature=0, # this is the degree of randomness of the model's output
                stream = True,
            ):
            # print(chunk)
            content = chunk["choices"][0].get("delta", {}).get("content")
            
            if content is not None:
                string += content
                print(content, end="")
    except openai.error.APIError as e:
        print("=====APIError=====")
        print('Error:', e)
        string = "APIError occurred!"
    except openai.error.RateLimitError:
        print("=====RateLimitError!=====")
        string = "RateLimit occurred!"
    # Print execution time
    end_time = time.time()
    print("Open API call took %.2f seconds" % (end_time - start_time))
    return string
  
  def prompt_paper_info(text):
    #print("=========Generating abstract Prompt=========")
    prompt = f"""
    The text delimited by "<>" is a first chunk of a research paper of Tobacco.
    Your response must be in JSON format.
    TASK:
    Your task is to extract information in the text delimited by "<>" to JSON string by following these steps:
    Step 1: Collect the author, title, abstract, and keywords of the text.
    Step 2: Generate a JSON object that includes the following fields: 
        "authors": a list representing the author(s) names of the article;
        "title: a string representing the title of the article;
        "abstract": a string representing the abstract of the article;
        "keywords": an array of strings representing the keywords of the article;
        "classification": a string representing the classification of the article;
        "document_id": a string representing the document ID of the article;
        "doi": a string representing the DOI (Digital Object Identifier) of the article;
        "osid": a string representing the OSID (Object Identifier System) of the article.
    You must ensure that the information you response is indeed from the text and does not contain any extra information.

    Attention:
    - If the information is missing, use "unknown" to represent it.
    - Find relevant information based on the text.
    - Be as accurate as possible.
    - In your response, Chinese is preferable.
    - If there is both Chinese and English information in the text, prioritize the Chinese information.
    - 尽可能用中文
    Text:
    '''<{text}>'''
    """
    return prompt
    # If you don't know, simply write unknown.

def get_title(json_str):
    data = json.loads(json_str)
    return data["title"]
 
# prompt部分
def prompt_author_info(text):
    #print("=========Generating abstract Prompt=========")
    prompt = f"""
    The text delimited by "<>" is a first chunk of a research paper of Tobacco.
    Your response must be in JSON format.
    TASK:
    Your task is to extract information in the text delimited by "<>" to JSON string by following these steps:
    Step 1: Collect the author information in of the text.
    Step 2: Generate a JSON Array "Authors" about authors that each includes the following fields: 

        "Name": a string represents the author's name
        "Academic Title": a string represents the author's academic title, such as "Professor", "Associate Professor"
        "Affiliation": a string represents the author's affiliation, such as a university, research institute, company, etc
        "Research Field": a string represents the author's primary research field or area of expertise
        "Mailing Address": a string represents the author's city for mailing address
        "Email Address": a string represents the author's email address
        "Personal Website": a string represents the author's personal website or blog URL

    You must ensure that the information you response is indeed from the text and does not contain any extra information.

    Attention:
    - If the information is missing, use "unknown" to represent it.
    - Find relevant information based on the text.
    - Be as accurate as possible.
    - In your response, Chinese is preferable.
    - If there is both Chinese and English information in the text, prioritize the Chinese information.
    - The result is for Chinese people, so when information is represent in several langages, Chinese is prefered.
    Text:
    '''<{text}>'''
    """
    return prompt
    # If you don't know, simply write unknown.
  
  def prompt_reference(text):
    #print("=========Generating abstract Prompt=========")
    prompt = f"""
    The text delimited by "<>" is a chunk of a research paper of Tobacco contains the references.
    Your response must be in JSON format.
    TASK:
    Your task is to extract information in the text delimited by "<>" to JSON string by following these steps:
    Step 1: Collect the references or bibliography information from the text.
    Step 2: Generate a JSON Array about "references" that each includes the following fields:
    - "author": a list represents the author(s) of the reference
    - "title": a string represents the title of the reference
    - "journal": a string represents the name of the journal where the reference is published
    - "year": a string represents the year when the reference is published
    - "volume": a string represents the volume number of the journal where the reference is published
    - "issue": a string represents the issue number of the journal where the reference is published
    - "pages": a string represents the page range of the reference in the journal
    You must ensure that the information you response is indeed from the text and does not contain any extra information.
    you should response in following format:
    {{
    "references": [
        {{
            "author": ["","",""],
            "title": "",
            "journal": "",
            "year": "",
            "volume": "",
            "issue": "",
            "pages": ""
        }}, ...]
    }}
    Attention:
    - If the information is missing, use "unknown" to represent it.
    - Find relevant information based on the text.
    - Be as accurate as possible.
    - Ignore incomplete reference information suspected
    - The result is for Chinese people, so when information is represent in several langages, Chinese is prefered.
    - 尽可能用中文
    Text:
    '''<{text}>'''
    """
    return prompt


# neo4j  
uri = "neo4j+s://2ffef87e.databases.neo4j.io"
user = "neo4j"
password = "yULLv4mjiwYg-q1JI5J6Dd9Vy7Dv6h3F3g4TV58R8MM"
  
import json
def write_authors_to_neo4j(json_str):
    # 连接 Neo4j 数据库
    driver = GraphDatabase.driver(uri, auth=(user, password))
    session = driver.session()
    # 遍历作者列表，创建节点
    data = json.loads(json_str)
    for author in data["Authors"]:
        # 创建或更新节点
        query = "MERGE (:Author {Name: $name})"
        session.run(query, {"name": author["Name"]})

        # 更新节点属性
        query = "MATCH (a:Author {Name: $name}) SET a.AcademicTitle = $title, a.Affiliation = $affiliation, a.ResearchField = $field, a.MailingAddress = $address, a.EmailAddress = $email, a.PersonalWebsite = $website"
        params = {
            "name": author["Name"],
            "title": author["Academic Title"],
            "affiliation": author["Affiliation"],
            "field": author["Research Field"],
            "address": author["Mailing Address"],
            "email": author["Email Address"],
            "website": author["Personal Website"]
        }
        session.run(query, params)

    # 关闭连接
    session.close()
    driver.close()

def write_references_to_neo4j(json_str, paper_name):
    # Connect to Neo4j database
    driver = GraphDatabase.driver(uri, auth=(user, password))
    data = json.loads(json_str)
    with driver.session() as session:
        for reference in data["references"]:
            # Create a node for each reference
            node = {
                "author": reference["author"],
                "title": reference["title"],
                "journal": reference["journal"],
                "year": reference["year"],
                "volume": reference["volume"],
                # "issue": reference.get("issue", "unknown"),
                "issue": reference["issue"],
                "pages": reference["pages"]
            }
            session.run("CREATE (:Reference $node)", node=node)
        # Create a relationship between the reference node and the paper node
            session.run("""
                MATCH (p:Paper {title: $title})
                CREATE (p)-[:CITES]->(:Reference $node)
            """, title=paper_name, node=node)
    driver.close()

    
# 新的呈现
def write_paper_to_neo4j(paper_str):
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        paper = json.loads(paper_str)
        # 创建论文节点
        session.run("MERGE (p:Paper {title: $title, abstract: $abstract, classification: $classification, doi: $doi, osid: $osid, document_id: $document_id})", **paper)
        
        # 建立作者与论文的关系
        for author in paper['authors']:
            session.run("MERGE (a:Author {Name: $name})", name=author)
            session.run("MATCH (a:Author {Name: $author_name}), (p:Paper {title: $paper_title}) MERGE (a)-[:AUTHOR_OF]->(p)", author_name=author, paper_title=paper['title'])
        
        # 建立关键词与论文的关系
        for keyword in paper['keywords']:
            session.run("MERGE (k:Keyword {name: $name})", name=keyword)
            session.run("MATCH (p:Paper {title: $title}), (k:Keyword {name: $name}) MERGE (p)-[:HAS_KEYWORD]->(k)", title=paper['title'], name=keyword)
    
    driver.close()

def create_reference(tx, paper_node, author_node, reference):
    # 创建参考文献节点
    reference_node = tx.run("MERGE (r:Reference {title: $title, journal: $journal, year: $year, volume: $volume, issue: $issue, pages: $pages}) RETURN r", 
                            title=reference['title'], journal=reference['journal'], year=reference['year'], 
                            volume=reference['volume'], issue=reference['issue'], pages=reference['pages']).single()[0]
    
    # 创建引用关系
    tx.run("MATCH (p:Paper),(r:Reference) WHERE p.title = $paper_name AND r.title = $title CREATE (p)-[:CITES]->(r)", 
           paper_name=paper_node["title"], title=reference_node["title"])
    
    # 创建作者节点
    for author in reference['author']:
        author_node = tx.run("MERGE (a:Author {name: $name}) RETURN a", name=author).single()[0]
        
        # 创建作者与参考文献之间的关系
        tx.run("MATCH (a:Author),(r:Reference) WHERE a.Name = $author_name AND r.title = $title CREATE (a)-[:AUTHOR_OF]->(r)", 
               author_name=author_node["name"], title=reference_node["title"])

def insert_references_into_neo4j(references_json, paper_name):
    references = json.loads(references_json)["references"]
    driver = GraphDatabase.driver(uri, auth=(user, password))
    with driver.session() as session:
        # 获取论文节点
        paper_node = session.run("MATCH (p:Paper {title: $name}) RETURN p", name=paper_name).single()[0]
        
        for reference in references:
            # 创建作者节点
            author_nodes = []
            for author in reference['author']:
                author_node = session.run("MERGE (a:Author {Name: $name}) RETURN a", name=author).single()[0]
                author_nodes.append(author_node)
            
            # 创建参考文献节点并与作者和论文相关联
            session.write_transaction(create_reference, paper_node, author_nodes, reference)
    
    driver.close()

# 生成prompt，执行prompt，返回json，写入pdf_name_article.json文件，并返回文件名称
def text_to_json(text, create_prompt, pdf_name, type_str="paper"):
    # 生成并打印prompt
    prompt = create_prompt(text)
    print(prompt)
    # res = get_completion(prompt)
    res = flow_get_completion(prompt)
    # print(res)
    file_name = pdf_name + "_"+ type_str + ".json"
    write_to_file(res, file_name)
    return file_name
 
def get_reference_text(str):
    index = str.rfind('参考文献')
    if index == -1:
        return ''
    else:
        return "参考文献" + str[index+len('参考文献'):].strip()

def get_reference_json(txt, pdf_name):
    reference_text = get_reference_text(txt)
    if reference_text == "":
        return ""
    reference_file = pdf_name + "_references.txt"
    write_to_file(reference_text, reference_file)
    reference_docs = split_text(reference_file, 1000, 100)
    json_list = []
    for doc in reference_docs:
        prompt = prompt_reference(doc.page_content)
        res_json = flow_get_completion(prompt)
        if "Error" not in res_json and "Rate_Limit" not in res_json:
            res_json = res_json.replace(", \"等\"","")
            res_json = res_json.replace(", \"et al.\"","")
            res_json = res_json.replace("[J]","")
            json_list.append(res_json)

    # 将列表中的 JSON 字符串合并为一个 JSON 字符串
    merged_json_string = json_list[0] 

    for json_string in json_list[1:]:
        merged_json_string = merge_ref_json(merged_json_string, json_string)

    json_name = pdf_name + "_references.json"
    write_to_file(merged_json_string, json_name)
    return json_name

# 合并json
def merge_ref_json(a, b):
    # 将 JSON 字符串转换成 Python 对象
    a_obj = json.loads(a)
    b_obj = json.loads(b)

    # 创建一个字典来存储合并后的结果
    merged_obj = {}

    # 遍历 A 中的每个参考文献对象
    for ref_a in a_obj['references']:
        # 在 B 中查找与 A 中当前参考文献对象对应的参考文献对象
        ref_b = next((r for r in b_obj['references'] if r['title'] == ref_a['title']), None)

        # 如果找到了对应的参考文献对象，则将其合并到 A 中的参考文献对象中
        if ref_b:
            ref_a.update(ref_b)

        # 将合并后的参考文献对象添加到 merged_obj 中
        merged_obj.setdefault('references', []).append(ref_a)

    # 将 B 中剩余的参考文献对象添加到 merged_obj 中
    for ref_b in b_obj['references']:
        if not any(r['title'] == ref_b['title'] for r in merged_obj['references']):
            merged_obj.setdefault('references', []).append(ref_b)

    # 将合并后的 Python 对象转换成 JSON 字符串并返回
    return json.dumps(merged_obj, ensure_ascii=False, indent=4)

# colab 打包文件
import zipfile

# 要打包的目录路径
dir_path = "/content"

# 要打包的文件列表
file_paths = [os.path.join(dir_path, filename) for filename in os.listdir(dir_path) if filename.startswith("test")]

# 创建一个 ZIP 文件
zip_file = zipfile.ZipFile("myfiles.zip", "w")

# 向 ZIP 文件中添加文件
for file_path in file_paths:
    zip_file.write(file_path, os.path.basename(file_path))

# 关闭 ZIP 文件
zip_file.close()

# 下载 ZIP 文件到本地计算机
from google.colab import files
files.download("myfiles.zip")

# 主循环
pdf_list = ["1.pdf"]
for pdf_name in pdf_list:
    # print("正在执行：" + pdf_name)
    # txt = pdf_to_text(pdf_name)
    # print("Generate intermediate file: ", pdf_name,".txt")
    # txt = txt[:txt.find("上接第 １６９ 页")]
    # special_chars = "�"

    # # 使用 replace() 方法删除特殊字符
    # for char in special_chars:
    #     txt = txt.replace(char, "")
    # print(txt[-100:])
    # write_to_file(txt, pdf_name + ".txt")
    documents = split_text(pdf_name + ".txt")

    # 作者
    author_json_name = text_to_json(documents[0].page_content, prompt_author_info, pdf_name, "author")
    author_data = read_file(author_json_name)
    print(author_data)
    # 将论文信息写入到Neo4j数据库中
    write_authors_to_neo4j(author_data)

    # paper
    paper_json_name = text_to_json(documents[0].page_content, prompt_paper_info, pdf_name, "paper")
    paper_data = read_file(paper_json_name)
    title = get_title(paper_data)
    print(paper_data)
    write_paper_to_neo4j(paper_data)

    # reference_json
    # reference_json_name = text_to_json(documents[-1].page_content, prompt_reference, pdf_name, "reference")
    reference_json_name = get_reference_json(txt, pdf_name)
    if reference_json_name == "":
        continue
    reference_data = read_file(reference_json_name)
    # print(reference_data)
    # write_references_to_neo4j(reference_data, title)
    insert_references_into_neo4j(reference_data, title)
 

# 在线解压压缩包
import os

def unrar_file(rar_file_path, extract_path):
    # 安装 unrar 工具
    !apt-get install unrar

    # 创建解压目标文件夹
    if not os.path.exists(extract_path):
        os.makedirs(extract_path)

    # 解压 rar 文件到目标文件夹
    !unrar x "{rar_file_path}" "{extract_path}/"

# 追加prompt
你是一个文本信息过滤器，你现在需要按我的要求对尖括号<>中的文本进行过滤和规范，最后回答文本中参考文献信息。注意，这一文本是从烟草相关论文pdf文件中直接提取的参考文献部分，因此，它的头尾可能并不完整，并且内容是不规范的。
你需要按照这样的步骤完成这项任务, 最终只需按顺序回答完整的文献信息即可，不要多余的信息：
1. 文本本身可能存在格式问题，先将它规范化，去除特殊字符和不必要的空白符号
2. 找到文本中完整的引用一条完整的引用。它可以是这样的： [序号] 作者.文章题目.期刊名,年份,卷号(期号):起止页码.因此注意，没有“[序号]”这一部分的引用，你需要认为它是不完整的，并且忽视它。不用担心缺失部分的影响
3. 文献信息选取尽可能是中文内容，如果同一条信息包含中英部分，只选取中文部分。如果仅包含英语部分，那么就选取英语部分
4. 将选取的完整文献信息，按照中文文献引用习惯返回
文本信息如下：<>

你是一个文本信息过滤器，你现在需要按我的要求对尖括号<>中的文本进行过滤和规范，最后返回我需要的信息。
这一文本是从烟草相关论文pdf文件中直接提取的开头部分，因此，它的可能并不完整，并且内容可能是不规范的。主要关注论文的信息和来源，作者和作者的信息，以及摘要。
你需要按照这样的步骤完成这项任务, 最终只需返回规范后的文本，不要多余的信息：
1. 文本本身可能存在格式问题，先将它规范化，并去除特殊字符和不必要的空白符号
2. 找到文本中论文的信息和来源，作者和作者的信息，以及摘要内容
文本可能包含不完整的部分，你可以选择性忽视它，不用担心缺失部分的影响
3. 所有信息选取尽可能是中文内容，如果同一条信息包含中英部分，只选取中文部分。如果仅包含英语部分，那么就选取英语部分
4. 最后将规范后的论文文本，按照中文论文书写习惯返回
文本信息如下：<>
  
