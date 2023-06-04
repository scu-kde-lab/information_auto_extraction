!pip install openai
!pip install langchain
!pip install unstructured
!pip install pdf2image
!pip install pytesseract
!pip install PyMuPDF

import openai
import os
import json
import time
import os

from langchain.document_loaders import UnstructuredFileLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain import OpenAI
import re
import fitz

import requests
from bs4 import BeautifulSoup
import re

# Need to set
openai.api_key = "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"
openai.api_base = "https://ai.tudb.work/v1"

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
                temperature=0.3, # this is the degree of randomness of the model's output
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
  
  
  def split_text(file_name):
    loader = UnstructuredFileLoader(file_name) 
    # 将文本转成 Document 对象
    document = loader.load()
    print(f'documents:{len(document)}')
    # 初始化文本分割器
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200
    )
    # 切分文本
    split_documents = text_splitter.split_documents(document)
    print(f'documents:{len(split_documents)}')
    return split_documents


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
        
def prompt_text(text):
    #print("=========Generating pdf_to_json Prompt=========")
    prompt = f"""
    I am a researcher in the field of datasets trying to collect imformation (names and url links) of datasets.
    A dataset there can be used to validate the results presented in a paper or to reproduce the study and conduct further analysis.
    And the given Text delimited by "<>" is a chunk of a research paper.
    Your task is to extract information about datasets name and url links in the Text.
    Your response must be a parsable json str.

    Following these steps to finish the work:
    Step 1: Determine if the given text mentions datasets. If not, just write "{{"dataset_mentioned": false}}" and stop.
    Step 2: Find the dataset names and their urls.
    Step 3: Generate a JSON Array about "dataset" that each includes the following fields:
    - "name": a string represents dataset name
    - "url": a string represents the accessible url links
    The names and url links must match.

    you should response in following format:
    {{
        "dataset_mentioned": ""
        "datasets": [
            {{
                "dataset_name": "",
                "url":""
            }}, ...]
    }}

    You must ensure that the information you response is indeed about the datasets and does not contain any extra information.
    If you don't know the dataset url, simply write unknown.
    Attention:
    - Limiting each "url" in schema no more than 100 letters.
    - Ignore any nameless datasets.
    - If the "url" is missing, use "unknown" to represent it.
    - Find relevant information based on the text.
    - Be as accurate as possible.
    Text:
    '''<{text}>'''
    """
    return prompt
 
import requests

def check_link(url):
    try:
        response = requests.head(url, timeout=3.0)
        if response.status_code == 200:
            return True
        else:
            return False
    except requests.exceptions.RequestException as e:
        print("可疑链接：", url)
        return True
  
  def filter_dataset_json(json_str):
    # 将 JSON 数据解析为 Python 对象
    print("解析字符串")
    data = json.loads(json_str)
    print("解析完成")
    # 遍历每个数据集，检查链接是否可访问
    datasets = data['datasets']
    new_datasets = []
    for dataset in datasets:
        # url and dataset_name condition
        if (dataset['url'] == "unknown" or check_link(dataset['url'])) and len(dataset["dataset_name"]) < 50:
            new_datasets.append(dataset)
    print("运行到此处")

    # 更新数据集列表
    data['datasets'] = new_datasets

    # 将更新后的数据重新转换为 JSON 格式并返回
    new_json_str = json.dumps(data, indent=4)
    print(new_json_str)
    print("====")
    return new_json_str
  
  import json
def json_list_parse(json_list):
    datasets = []
    name_set = set()

    for json_str in json_list:
        data = json.loads(json_str)
        datasets += data['datasets']

    print("dataset:")
    print(datasets)
    result = {'datasets': {}}
    for dataset in datasets:
        name = dataset['dataset_name']
        name_set.add(name)
        url = dataset['url']
        if name in result['datasets']:
            if url not in result['datasets'][name] and url != "unknown":
                result['datasets'][name].append(url)
        else:
            result['datasets'][name] = [url]
    
    for i in result['datasets'].values():
        if len(i)>1:
            try:
                i.remove("unknown")
            except ValueError:
                pass

    result_json_str = json.dumps(result,indent=4)

    res = {}
    data = json.loads(result_json_str)

    for dataset_name, urls in data['datasets'].items():
        if len(urls) > 1 or "unknown" not in urls:
            res[dataset_name] = urls

    print(list(name_set))
    return json.dumps(res,indent=4), list(name_set)
  
  def extract_pdf_datasets(pdf_name):
    print("Preprocess: " + pdf_name)
    txt = preprocess(pdf_to_text(pdf_name))
    print("Generate intermediate file: ", pdf_name,".txt")
    write_to_file(txt, pdf_name + ".txt")
    documents = split_text(pdf_name + ".txt")
    print("Getting json list!")
    json_list = [flow_get_completion(prompt_text(i.page_content)) for i in documents]

    # 两次过滤
    print("Start to filter json list!")
    print("Step 1")
    # json_list = [s for s in json_list if  not in s]
    # 展开这里
    filtered_list=[]
    for j in json_list:
        # 有数据集 并且没有出错
        if '"dataset_mentioned": false' not in j and "Error" not in j:
            try:
                tem_json = filter_dataset_json(j)
                filtered_list.append(tem_json)
                print(tem_json)
            except:
                print("过滤以下json出现问题：")
                print(tem_json)
    print("过滤后json list的长度为：", end="")
    print(len(filtered_list))
    print("Step 2")
    filtered_json_list = [i for i in filtered_list if "\"datasets\": []" not in i]
    print("过滤后json list的长度为：", end="")
    print(len(filtered_list))
    print("Finish filtering!")
    print("最终的json_list串")
    print(filtered_json_list)
    print("=======================================")
    j,l = json_list_parse(filtered_json_list)
    print(l)
    return j,l
  
# 爬虫部分
# 发送 HTTP 请求获取页面内容
url = "https://arxiv.org/search/?query=dataset&searchtype=all&source=header"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# 解析页面内容，获取每篇论文的链接
paper_links = []
for a in soup.select("p.list-title > a"):
    paper_links.append(a.get("href"))
print(paper_links)
pdf_list = []
cnt = 0
for link in paper_links:
    pdf_url = link.replace("abs", "pdf")
    print(pdf_url)
    pdf_name = pdf_url.split("/")[-1]
    pdf_list.append(pdf_name)
    response = requests.get(pdf_url)
    with open(pdf_name+".pdf", "wb") as f:
        f.write(response.content)
    cnt+=1
    if cnt>15:
        break
  
for i in [j + ".pdf" for j in pdf_list]:
    pdf_name = i
    js, ls = extract_pdf_datasets(pdf_name)
    write_to_file("Json 内容：\n"+js +"\n列表内容：\n" + str(ls), pdf_name + "_res.txt")
 
# 两次过滤
print("Start to filter json list!")
print("Step 1")
# json_list = [s for s in json_list if  not in s]
# 展开这里
filtered_list=[]
for j in json_list:
    # 有数据集 并且没有出错
    if '"dataset_mentioned": false' not in j and "Error" not in j:
        try:
            tem_json = filter_dataset_json(j)
            filtered_list.append(tem_json)
            print(tem_json)
        except:
            print("过滤以下json出现问题：")
            print(tem_json)
print("过滤后json list的长度为：", end="")
print(len(filtered_list))
print("Step 2")
filtered_json_list = [i for i in filtered_list if "\"datasets\": []" not in i]
print("Finish filtering!")
j,l = json_list_parse(filtered_json_list)
print(j,l)
