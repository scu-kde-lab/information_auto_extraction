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
openai.api_key = ""
openai.api_base = ""

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
 

def prompt_text_to_json(text):
    #print("=========Generating pdf_to_json Prompt=========")
    prompt = f"""
    The text delimited by "<>" is a chunk of a research paper.
    A dataset there can be used to validate the results presented in a paper, as well as to reproduce the study and conduct further analysis.
    Your response must be in JSON format.
    TASK:
    Your task is to extract information about datasets mentioned in the text delimited by "<>" to JSON string by following these steps:
    Step 1: Determine if the given text mentions datasets. If not, just write "{{"dataset_mentioned": false}}" and stop.
    Step 2: Find information about the datasets and its source.
    Step 3: Standardize all the information about the datasets in json:{{
   "dataset_name": name,
   "source": url or related work or reseach paper
}} and each dataset information must be in the form of brackets and one unit describes no more than 1 dataset!
    A good example:
    {{
            "dataset_name": "ImageNet",
            "source": "https://image-net.org/"
        }}
    You must ensure that the information you response is indeed about the datasets and does not contain any extra information.
    If you don't know the dataset resouce, simply write unknown.
    Attention:
    - Limiting each "source" in schema no more than 50 letters.
    - Providing a link to the dataset source is preferable to using abstract or vague descriptions.
    - If the "source" is missing, use "unknown" to represent it.
    - Find relevant information based on the text.
    - Be as accurate as possible.

    - A "dataset_name" can't be unkown and ignore it in your response string. The following example is bad:
    {{
            "dataset_name": "unknown",
            "source": "German Conference"
        }}

    Text:
    '''<{text}>'''
    """
    return prompt
  
  
  def prompt_merge_json_new(json_1,json_2):
    # print("=========Generating merge_json Prompt=========")
    prompt = f"""
    Your task is to merge two JSON string (Json_1 and Json_2 are delimited by "<>") into one JSON string. So your response must be in JSON format.  
    Json_2 contains a comprehensive dataset information while Json_1 contains new datasets information that needs to be merged into Json_2.
    Complete this task by following these steps:
    Step 1: Add the datasets from Json_1 that were not originally included in Json_2 to Json_2.
    Step 2: If there are objects in both JSON strings with the same dataset_name but different "source", select one or merge them into a more specific and convincing object to replace the original object in Json_2.
All datasets in json should in such format:{{
   "dataset_name": name,
   "source": url or related work or reseach paper
}} and each dataset information must be in the form of brackets and one unit describes no more than 1 dataset!
    Step 3: response in JSON
    You must ensure that the information you response is indeed generated from the two json strings and does not contain any extra information.
    Attention:
    - A link to the dataset source is preferable to using abstract or vague descriptions.
    - If a dataset name is "None" or empty or so, it shouldn't be added into the response JSON.
    Json_1:
    '''<{json_1}>'''
    Json_2:
    '''<{json_2}>'''
    """
    # print("Now the prompt is as following……")
    # print(prompt)
    # print("==========================================================")
    return prompt

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
        
        
# 爬虫部分
# 发送 HTTP 请求获取页面内容
url = "https://arxiv.org/search/?query=image+processing&searchtype=all&source=header"
response = requests.get(url)
soup = BeautifulSoup(response.text, "html.parser")

# 解析页面内容，获取每篇论文的链接
paper_links = []
for a in soup.select("p.list-title > a"):
    paper_links.append(a.get("href"))
print(paper_links)
pdf_list = []
for link in paper_links:
    pdf_url = link.replace("abs", "pdf")
    print(pdf_url)
    pdf_name = pdf_url.split("/")[-1]
    pdf_list.append(pdf_name)
    response = requests.get(pdf_url)


test_list = []
cnt = 0
for i in pdf_list:
    pdf_name = "pdf//"+i+".pdf"
    with fitz.open(pdf_name) as pdf_file:
    # 获取 PDF 文件的页数
        num_pages = pdf_file.page_count
    # 获取 PDF 文件的页数
    print(pdf_name," pages = ", num_pages)
    if num_pages > 15 or pdf_name in test_list:
        print("跳过")
        print("=============")
        continue
    txt = preprocess(pdf_to_text(pdf_name))

    test_list.append(pdf_name)
    print("Generate intermediate file: ", pdf_name,".txt")
    write_to_file(txt, pdf_name + ".txt")
    documents = split_text(pdf_name + ".txt")
    json_list = [get_completion(prompt_text_to_json(i.page_content)) for i in documents]
    # 过滤json
    a = [i for i in json_list if "\"dataset_mentioned\": false" not in i and "==RateLimitError!==" not in i and "RateLimit occurred!" not in i]
    # 归并json
    for i in range(len(a)):
        if i == 0:
            continue
        tem = get_completion(prompt_merge_json_new(a[i], a[0]))
        if tem != "APIError occurred!" and tem!="RateLimit occurred!":
            a[0] = tem

    print(a[0])
    write_to_file(a[0],pdf_name+".json")
    with open("pdf//"+pdf_name+".pdf", "wb") as f:
        f.write(response.content)
