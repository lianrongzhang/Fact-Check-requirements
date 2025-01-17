# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb = Chroma(persist_directory='2024-11-20-db', embedding_function=embeddings)
# for i in vectordb.get()['documents']:
#     print(i)
#     print('-'*50)

# from sentence_transformers import SentenceTransformer
# from sklearn.metrics.pairwise import cosine_similarity

# # 加載預訓練的 Sentence-BERT 模型
# model = SentenceTransformer('all-MiniLM-L6-v2')

# # 定義兩段句子
# sentence_1 = "Kas kaitsevägi lasi õhku äsja 12 miljoni eest renoveeritud viadukti?"
# sentence_2 = """
# ### Analysis of Claim:\n\n- **Key Information from Context**: The claim is that Barack Obama said Donald Trump \'tried to kill\' Mike Pence.\n\n### Step-by-Step Evaluation:\n\n1. According to USA Today\'s fact-checking article on October 23rd, the quote was "misrepresents" an Obama campaign speech in Pittsburgh, suggesting that the quote is taken out of context or manipulated in some way.\n\n2. Check Your Fact\'s article on October 25th also fact-checked the claim and determined it to be "False", without providing further details on why the claim is false, but suggesting that there may be no credible evidence to support the quote.\n\n### Conclusion:\n\n- **Claim Status**: False\n- **Supporting URL**: https://www.usatoday.com/story/news/factcheck/2024/10/23/obama-misquoted-trump-jan-6-fact-check/75678173007/\n\nIn this case, both pieces of evidence suggest that the claim is false.
# # """

# # 將句子轉換為向量
# embedding_1 = model.encode(sentence_1)
# embedding_2 = model.encode(sentence_2)

# # 計算餘弦相似度
# similarity = cosine_similarity([embedding_1], [embedding_2])

# # 輸出相似度
# print(f"Cosine Similarity: {similarity[0][0]}")



# 定義正則表達式模式
# patterns = {
#     "Claim Status": [r"Claim Status:\s*([^\n]+)", r"\*\*Claim Status\*\*:\s*([^\n]+)"],
#     "Language": [r"Language:\s*([^\(\n]+)", r"\*\*Language\*\*:\s*([^\(\n]+)"],
#     "Date": [r"Date:\s*([^\(\n]+)", r"\*\*Date\*\*:\s*([^\(\n]+)"],
#     "Country": [r"Country:\s*([A-Z]+)", r"\*\*Country\*\*:\s*([A-Z]+)"],
#     "URL": [r"URL:\s*(https?://[^\s]+)", r"https?://[^\s]+"]
# }

# # 定義解析函式
# def extract_data(response, patterns):
#     result = {}
#     for key, regex_list in patterns.items():
#         if key == "URL":  # 特別處理多個URL的情況
#             matches = []
#             for regex in regex_list:
#                 matches.extend(re.findall(regex, response))
#             result[key] = matches if matches else None
#         else:
#             for regex in regex_list:
#                 match = re.search(regex, response)
#                 if match:
#                     result[key] = match.group(1).strip()
#                     break
#             if key not in result:
#                 result[key] = None  # 若無匹配，填充 None
#     return result

# # 從 vectordb1 中獲取所有 ids
# try:
#     ids = vectordb1.get().get('metadatas', [])
# except Exception as e:
#     print(f"Error fetching IDs: {e}")
#     ids = []

# # 解析每個文檔
# if not ids:
#     print("No IDs found in the question database.")
# else:
#     for i in ids:
#         try:
#             doc_data = vectordb2.get(ids=[i['id']])  # 提取文檔
#             documents = doc_data.get('documents', [])
            
#             if not documents:
#                 print(f"No documents found for ID: {i['id']}")
#                 continue
            
#             for response in documents:  # 迭代文檔內容
#                 parsed_data = extract_data(response, patterns)
#                 print(parsed_data)  # 輸出解析結果
#                 print('-' * 50)
#         except Exception as e:
#             print(f"Error processing ID {i['id']}: {e}")


# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from datetime import datetime
# import re

# # 初始化 Chroma 和嵌入模型
# today = datetime.today().strftime('%Y-%m-%d')
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# vectordb2 = Chroma(persist_directory=f"{today}-answer-db", embedding_function=embeddings)
# vectordb1 = Chroma(persist_directory=f"{today}-question-db", embedding_function=embeddings)


# query_result = vectordb1.get()['metadatas']

# print(query_result)  # 輸出查詢結果

# import json

# with open('APIKey.json', 'r') as f:
#     config = json.load(f)

# factcheckapi = config['Fact_Check_API']
# print(factcheckapi)  # 輸出 FactCheckAPI 配置


# import timeit


# def compute_factorials(n):
#     from math import factorial
#     results = []
#     for i in range(1, n + 1):
#         results.append(factorial(i))
#     return results


# # 計算 1 到 10 的階乘
# n = 1000
# timeit_results = timeit.timeit("compute_factorials(n)", globals=globals(), number=1000)
# print(f"Time taken to compute factorials from 1 to {n}: {timeit_results:.5f} seconds")

# from datasets import load_dataset
# import pandas as pd

# # 載入數據集
# dataset = load_dataset("pszemraj/multi_fc")

# # 處理 train 子集並儲存為 JSON
# train_data = pd.DataFrame(dataset['train'])  # 將 train 子集轉為 Pandas DataFrame

# # 檢查數據的前幾行（可選）
# print(train_data.head())

# # 將 DataFrame 儲存為 JSON 文件
# output_file = "train_data.json"
# train_data.to_json(output_file, orient='records', lines=True, force_ascii=False)

# import pandas as pd

# # 讀取 Parquet 文件
# df = pd.read_parquet("hf://datasets/tdiggelm/climate_fever/data/test-00000-of-00001.parquet")

# print(df.head())  # 輸出數據的前幾行

# # 將 DataFrame 儲存為 JSON 文件，生成完整的 JSON 數組格式
# output_file = "climate_fever.json"

# df.to_json(output_file, orient='records', lines=False, force_ascii=False)

# print(f"JSON 文件已儲存為: {output_file}")

# from datasets import load_dataset
# import pandas as pd
# ds = load_dataset("Cofacts/line-msg-fact-check-tw", "articles")

# print(ds)  # 輸出數據集的描述信息

# train_data = pd.DataFrame(ds['train'])

# # 去除 `normalArticleReplyCount` 為 0 的行
# train_data.drop(train_data[train_data['normalArticleReplyCount'] == 0].index, inplace=True)
# #去除 status 不為 'normal' 的行
# train_data = train_data[train_data['status'].isin(['NORMAL'])]
# print(train_data.shape)
# print(train_data.head())  # 輸出數據的前幾行

# # 將 DataFrame 儲存為 JSON 文件
# output_file = "Cofacts_articles.json"

# train_data.to_json(output_file, orient='records', lines=False, force_ascii=False)


# from datasets import load_dataset
# import pandas as pd
# ds = load_dataset("Cofacts/line-msg-fact-check-tw", "article_replies")

# print(ds)  # 輸出數據集的描述信息

# train_data = pd.DataFrame(ds['train'])

# # 去除 replyType 不為 'NOT_RUMOR' 或 'RUMOR' 的行

# train_data = train_data[train_data['replyType'].isin(['NOT_RUMOR', 'RUMOR'])]

# print(train_data.shape)

# print(train_data.head())  # 輸出數據的前幾行

# # 將 DataFrame 儲存為 JSON 文件
# output_file = "Cofacts_articles_replies.json"

# train_data.to_json(output_file, orient='records', lines=False, force_ascii=False)



# import json

# # 開啟並讀取 JSON 檔案
# file_path = "/home/user/talen-python/data/AVeriTeC.json"  # 替換為你的 JSON 檔案路徑
# try:
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)  # 將 JSON 資料讀取並轉換為 Python 字典或列表
# except FileNotFoundError:
#     print(f"找不到檔案：{file_path}")
# except json.JSONDecodeError as e:
#     print(f"解析 JSON 檔案時發生錯誤：{e}")
# test = []
# for i in data:
#     if i['label'] == 'Supported' or i['label'] == 'Refuted':
#         test.append([i['claim'],i['label']])
#     if i['label'] == 'Not Enough Evidence':
#         test.append([i['claim'],'Not Enough Information'])

# for i in test:
#     print(i)

# import json

# # 開啟並讀取 JSON 檔案
# file_path = "/home/user/talen-python/data/Climate_Fever.json"  # 替換為你的 JSON 檔案路徑
# try:
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)  # 將 JSON 資料讀取並轉換為 Python 字典或列表
# except FileNotFoundError:
#     print(f"找不到檔案：{file_path}")
# except json.JSONDecodeError as e:
#     print(f"解析 JSON 檔案時發生錯誤：{e}")

# test = []
# for i in data:
#     if i['claim_label'] == 0 :
#         test.append([i['claim'],'Supported'])
#     if i['claim_label'] == 1:
#         test.append([i['claim'],'Refuted'])
#     if i['claim_label'] == 2:
#         test.append([i['claim'],'Not Enough Information'])

# # 開啟並讀取 JSON 檔案
# file_path = "/home/user/talen-python/data/MultiFC.tsv"  # 替換為你的 JSON 檔案路徑

# try:
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = file.readlines()  # 將 JSON 資料讀取並轉換為 Python 字典或列表
# except FileNotFoundError:
#     print(f"找不到檔案：{file_path}")

# test = []
# for i in data:
#     if i.split('\t')[2] == 'true':
#         test.append([i.split('\t')[1].replace("\"",""),'Supported'])
#     if i.split('\t')[2] == 'false':
#         test.append([i.split('\t')[1].replace("\"",""),'Refuted'])

# for i in test:
#     print(i)

# import json

# # 開啟並讀取 JSON 檔案
# file_path = "/home/user/talen-python/data/JP_fakenews.json"  # 替換為你的 JSON 檔案路徑

# try:
#     with open(file_path, "r", encoding="utf-8") as file:
#         data = json.load(file)  # 將 JSON 資料讀取並轉換為 Python 字典或列表
# except FileNotFoundError:
#     print(f"找不到檔案：{file_path}")
# except json.JSONDecodeError as e:
#     print(f"解析 JSON 檔案時發生錯誤：{e}")

# test = []
# for i in data:
#     if i['Q1'] == 'True':
#         test.append([i['Article'],'Supported'])
#     if i['Q1'] == 'False':
#         test.append([i['Article'],'Refuted'])

# import json

# # 開啟並讀取 JSON 檔案
# articles = "/home/user/talen-python/Cofacts_articles.json"  # 替換為你的 JSON 檔案路徑
# article_replies = "/home/user/talen-python/Cofacts_articles_replies.json"  # 替換為你的 JSON 檔案路徑

# try:
#     with open(articles, "r", encoding="utf-8") as file:
#         articles_data = json.load(file)  # 將 JSON 資料讀取並轉換為 Python 字典或列表
# except FileNotFoundError:
#     print(f"找不到檔案：{articles}")
# except json.JSONDecodeError as e:
#     print(f"解析 JSON 檔案時發生錯誤：{e}")

# try:
#     with open(article_replies, "r", encoding="utf-8") as file:
#         article_replies_data = json.load(file)  # 將 JSON 資料讀取並轉換為 Python 字典或列表
# except FileNotFoundError:
#     print(f"找不到檔案：{article_replies}")
# except json.JSONDecodeError as e:
#     print(f"解析 JSON 檔案時發生錯誤：{e}")

# test = []
# # 構建 replyType 為 "RUMOR" 的回應並根據 articleId 查找對應的文章資料
# for reply in article_replies_data:
#     if reply.get('replyType') == 'RUMOR':
#         article_id = reply.get('articleId')
        
#         # 查找對應的文章
#         article = next((item for item in articles_data if item['id'] == article_id), None)
        
#         if article:
#             # 把對應的文章和回應資料加入 test 清單
#             test.append({
#                 'id': article.get('id'),
#                 'claim': article.get('text'),
#                 'label': 'Refuted',
#                 'created_at': article.get('createdAt')
#             })

#     else:
#         article_id = reply.get('articleId')
        
#         # 查找對應的文章
#         article = next((item for item in articles_data if item['id'] == article_id), None)
        
#         if article:
#             # 把對應的文章和回應資料加入 test 清單
#             test.append({
#                 'id': article.get('id'),
#                 'claim': article.get('text'),
#                 'label': 'Supported',
#                 'created_at': article.get('createdAt')
#             })


# file_path = "/home/user/talen-python/data/Cofacts.json"  # 替換為你的 JSON 檔案路徑\

# with open(file_path, "w", encoding="utf-8") as file:
#     json.dump(test, file, ensure_ascii=False, indent=4)

# from langchain_community.document_loaders import WebBaseLoader

# # 初始化 WebBaseLoader
# loader = WebBaseLoader('https://tfc-taiwan.org.tw/articles/11337')

# # 載入數據
# data = loader.load()

# # 輸出數據的前幾行

# print(data)

# with open('/home/user/talen-python/data/MultiFC.json', 'r') as f:
#     data = f.readlines()

# import json

# test = []
# import os
# import re
# from fake_useragent import UserAgent

# # 初始化 UserAgent 物件
# ua = UserAgent()
# os.environ['USER_AGENT'] = ua.random
# import json
# import requests
# from time import time
# import timeit
# from langchain_community.document_loaders import WebBaseLoader


# with open('APIKey.json', 'r') as f:
#     config = json.load(f)

# fact_check_api_key = config.get('fact_check_api')
# search_engine_id = config.get('search_engine_id')
# search_api_key = config.get('search_api_key')

# def MultiFC():
#     with open('/home/user/talen-python/data/Climate_Fever.json', 'r') as f:
#         data = json.load(f)
#     test = []  # 用於保留順序
#     for i in data:
#         test.append(i['claim'])

#     return test

# def get_fact_check_url(query, API_KEY, SEARCH_ENGINE_ID):
#     url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&num=5&cx={SEARCH_ENGINE_ID}'

#     fact_check_urls = []
#     response = requests.get(url)
#     results = response.json()
#     if 'items' in results:
#         for item in results['items']:
#             fact_check_urls.append(item['link'])
#         return fact_check_urls
#     else:
#         return None
    
# def get_fact_check_content(urls):
#     if urls == None:
#         return None
#     print('-'*50)
#     print('searching relavant info...')
#     fact_check_content = []
#     loader = WebBaseLoader(urls, verify_ssl=False)
#     loader.aload()
#     print('searching completed.')
#     print('-'*50)
#     return fact_check_content

    
# query = MultiFC()

# for i in query:
#     urls = get_fact_check_url(i[0], search_api_key, search_engine_id)
#     print(urls)
#     content = get_fact_check_content(urls)
#     print(content)
#     print('-'*50)
#     break

# import signal
# import time
# # 定義超時處理函數
# def timeout_handler(signum, frame):
#     raise TimeoutError()

# # 設定超時機制的函數
# def run_with_timeout(func, args=(), kwargs={}, timeout=10):
#     # 設定信號和超時處理
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)  # 設定超時時間

#     try:
#         # 執行函數
#         result = func(*args, **kwargs)
#     finally:
#         # 清除定時器
#         signal.alarm(0)
#     return result

# # 測試函數
# def long_running_task(duration):
#     while True:
#         print(f"執行中...已過 {duration} 秒")
#         time.sleep(1)
#         duration -= 1
#         if duration <= 0:
#             break
#     return "任務已完成"

# # 使用範例
# try:
#     result = run_with_timeout(long_running_task, args=(5,))
#     print(f"結果: {result}")
# except TimeoutError as e:
#     print(e)


# import requests
# from langchain.schema import Document
# import json

# def google_fact_check(query, API_KEY):
#     # Base URL for Fact Check API
#     url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'

#     # Parameters for the API request, including languageCode
#     params = {
#         'query': query,
#         'languageCode': 'all',  # Specify language code here (e.g., 'en' for English, 'zh' for Chinese)
#         'key': API_KEY,
#         'page_size': 5
#     }
#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         if 'claims' in data:
#             processed_claims = []
#             seen_claims = set()  # To keep track of unique claims

#             for index, claim in enumerate(data['claims']):
#                 claim_text = claim.get('text')
#                 # Check if the claim has already been seen
#                 if claim_text not in seen_claims:
#                     seen_claims.add(claim_text)  # Add claim to the set

#                     # Create a dictionary for each claim with relevant details
#                     claim_info = {
#                         'ID': index + 1,
#                         'Claim': claim_text,
#                         'Claimant': claim.get('claimant'),
#                         'Claim Date': claim.get('claimDate'),
#                         'Claim Review': []
#                     }

#                     # Extract the claim review details

#                     processed_claims.append(claim_info)
#             return processed_claims
#         else:
#             return None
#     else:
#             print(f"Error: {response.status_code}, {response.json()}")

# def MultiFC():
#     # 開啟並讀取 JSON 檔案
#     file_path = "/home/user/talen-python/data/AVeriTeC.json"  # 替換為你的 JSON 檔案路徑
#     try:
#         with open(file_path, "r", encoding="utf-8") as file:
#             data = json.load(file)  # 將 JSON 資料讀取並轉換為 Python 字典或列表
#     except FileNotFoundError:
#         print(f"找不到檔案：{file_path}")
#     except json.JSONDecodeError as e:
#         print(f"解析 JSON 檔案時發生錯誤：{e}")
#     test = []
#     for i in data:
#         test.append(i["claim"])
#     return test

# query = MultiFC()
# result = []
# with open('APIKey.json', 'r') as f:
#     config = json.load(f)

# fact_check_api_key = config.get('fact_check_api')
# search_engine_id = config.get('search_engine_id')
# search_api_key = config.get('search_api_key')

# count = 0
# for i in query:
#     if count == 5:
#         break
#     print(i)
#     response = google_fact_check(i, fact_check_api_key)
#     if response != None:
#         result.append(response)
#     count += 1

# for i in result:
#     print(i)
#     print('-'*50)

# with open('fact_check.json', 'w') as f:
#     json.dump(result, f, indent=4)


# import re
# import requests
# import json

# # 定義函式以檢索 Fact Check API 的結果
# def google_fact_check(query,ans, API_KEY):
#     url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'

#     # API 請求的參數
#     params = {
#         'query': query,
#         'languageCode': 'all',  # 設置語言
#         'key': API_KEY,
#         'pageSize': 5  # 限制返回的結果數量
#     }

#     response = requests.get(url, params=params)
#     if response.status_code == 200:
#         data = response.json()
#         if 'claims' in data:
#             processed_claims = []
#             for claim in data['claims']:
#                 # 提取主要聲明資訊
#                 processed_claim = {
#                     'claim': clean_text(query),
#                     'text': clean_text(claim.get('text', 'N/A')),
#                     'label': clean_text(ans),
#                     'claimant': clean_text(claim.get('claimant', 'N/A')),
#                     'claimDate': clean_text(claim.get('claimDate', 'N/A')),
#                     'claimReview': []
#                 }
#                 # 提取 ClaimReview 詳細資訊
#                 for review in claim.get('claimReview', []):
#                     review_data = {
#                         'publisher_name': clean_text(review['publisher'].get('name', 'N/A')),
#                         'publisher_site': clean_text(review['publisher'].get('site', 'N/A')),
#                         'url': clean_text(review.get('url', 'N/A')),
#                         'title': clean_text(review.get('title', 'N/A')),
#                         'reviewDate': clean_text(review.get('reviewDate', 'N/A')),
#                         'textualRating': clean_text(review.get('textualRating', 'N/A')),
#                         'languageCode': clean_text(review.get('languageCode', 'N/A'))
#                     }
#                     processed_claim['claimReview'].append(review_data)
#                 processed_claims.append(processed_claim)
#             return processed_claims
#         else:
#             return {'error': 'No claims found in the response.'}
#     else:
#         return {'error': f'HTTP error {response.status_code}: {response.text}'}


# # 定義清理文本的函式
# def clean_text(text):
#     # 去除 \uXXXX 類型的 Unicode 字元
#     return re.sub(r'\\u[0-9a-fA-F]{4}', '', text)


# # 設定 Fact Check API 金鑰
# with open('APIKey.json', 'r') as f:
#     config = json.load(f)

# fact_check_api_key1 = config.get('fact_check_api1')
# fact_check_api_key2 = config.get('fact_check_api2')
# fact_check_api_key3 = config.get('fact_check_api3')

# API_list = [fact_check_api_key1,fact_check_api_key2,fact_check_api_key3]

# def test_data():
#     with open('/home/user/talen-python/test2.json', 'r') as f:
#         data = json.load(f)
#     test = []
#     for i in data:
#         test.append([i['claim'],i['label']])
#     return test


# query = test_data()



# result = []
# # 檢索 Fact Check API 的結果

# for i in query:
#     print(i[0])
#     response = google_fact_check(i[0],i[1],API_list[2])
#     if response and 'error' not in response:
#         result.append(response)

# # 將結果保存到文件
# with open('fact_check1.json', 'w', encoding='utf-8') as f:
#     json.dump(result, f, indent=4, ensure_ascii=False)


# import json
# with open('fact_check.json', 'r') as f:
#     data = json.load(f)

# print(len(data))  # 輸出 Fact Check API 的結果


# import requests
# import json
# import os
# import re
# from fake_useragent import UserAgent

# # 初始化 UserAgent 物件
# ua = UserAgent()
# os.environ['USER_AGENT'] = ua.random
# from langchain_community.document_loaders import WebBaseLoader
# from langchain.prompts import PromptTemplate
# import requests
# from langchain_ollama import OllamaLLM
# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from langchain.schema import Document
# import timeit
# from fp.fp import FreeProxy
# import urllib3
# import signal

# # 禁用 InsecureRequestWarning 警告
# urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
# def timeout_handler(signum, frame):
#     raise TimeoutError()

# def run_with_timeout(func, args=(), kwargs={}, timeout=15):
#     signal.signal(signal.SIGALRM, timeout_handler)
#     signal.alarm(timeout)
#     try:
#         result = func(*args, **kwargs)
#     except TimeoutError:
#         result = None
#     finally:
#         signal.alarm(0)
#     return result


# def get_fact_check_url(query, api_key, search_engine_id, num_results=5):
#     url = (
#         f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}&num={num_results}'
#     )
#     try:
#         response = requests.get(url, timeout=5)
#         response.raise_for_status()  # 確保請求成功
#         results = response.json()

#         if 'items' in results:
#             fact_check_urls = [item['link'] for item in results['items']]
#             return fact_check_urls
#         else:
#             print("No items found in the search results.")
#             return []
#     except requests.RequestException as e:
#         print(f"Error during API request: {e}")
#         return []


# def get_fact_check_content(urls, max_retries=3):
#     if not urls:
#         print("No URLs provided.")
#         return []

#     print('-' * 50)
#     print('Searching relevant info...')
#     fact_check_content = []

#     for i, url in enumerate(urls, start=1):
#         retries = 0
#         while retries < max_retries:
#             try:
#                 # 嘗試獲取代理
#                 proxy = FreeProxy(rand=True, timeout=3).get()
#                 print(f"[{i}/{len(urls)}] Attempt {retries + 1}: Using proxy: {proxy}")
#                 # 加載網頁內容
#                 docs = run_with_timeout(web_loader, args=(url,), kwargs={"proxies": {"http": proxy, "https": proxy}})
#                 fact_check_content.append([docs, url])
#                 break  # 如果成功，跳出重試循環
#             except Exception as e:
#                 print(f"[{i}/{len(urls)}] Attempt {retries + 1} failed for {url}: {e}")
#                 retries += 1

#         if retries == max_retries:
#             print(f"[{i}/{len(urls)}] Failed to load {url} after {max_retries} attempts.")

#     print('Searching completed.')
#     print('-' * 50)
#     return fact_check_content

# def web_loader(url, proxies=None, verify_ssl=False):
#     try:
#         loader = WebBaseLoader(url, proxies=proxies, verify_ssl=verify_ssl)
#         docs = loader.load()
#         return docs
#     except Exception as e:
#         print(f"Error loading content from {url}: {e}")
#         return None
    
# def analyze_fact_check(fact_check_content, model):
#     if not fact_check_content:
#         return
#     info = []
#     for i in fact_check_content:
#         # 初始化 LLM
#         llm = OllamaLLM(model=model)
#         # 定義 PromptTemplate
#         template = """
#         Write a concise summary of the following:
#         {context}
#         """
#         prompt = PromptTemplate.from_template(template)

#         # 格式化 prompt
#         formatted_prompt = prompt.format(context=i[0])

#         # 執行 LLM
#         time_start = timeit.default_timer()
#         result = llm.invoke(formatted_prompt)
#         time_end = timeit.default_timer()
#         # 將結果與 URL 結合
#         info.append(f"{result}, url: {i[1]}")

#     # 將結果轉為 Document 格式
#     documents = [Document(page_content=item) for item in info]

#     return documents, time_end - time_start


# def fact_check(query, documents, model,analyzer_time=None):
#     if not documents:
#         return
#     llm = OllamaLLM(model=model)
#     # 定義 prompt
#     template = """
#         You are a professional fact-checker tasked with evaluating the following claim.
#         Let's break down the evidence and reasoning step by step.
#         First, analyze the provided context {context} and identify key information relevant to the claim {claim}.
#         Then, evaluate the evidence step by step to determine if the claim is true or false.
#         Finally, structure the response in the following format:
        

#         ### Analysis of Claim:

#         - Key Information from Context:  
#         [Summarize the key points from the context relevant to the claim.]

#         ### Step-by-Step Evaluation:

#         1. Evidence 1:  
#         - Observation: [Detail the first piece of evidence or data relevant to the claim.]  
#         - Reasoning: [Explain how this evidence supports or refutes the claim, or note any limitations.]

#         2. Evidence 2:  
#         - Observation: [Detail the second piece of evidence or data relevant to the claim.]  
#         - Reasoning: [Explain how this evidence supports or refutes the claim, or note any limitations.]

#         3. Additional Analysis (if needed):  
#         - [Integrate multiple pieces of evidence or consider other contextual factors for deeper reasoning.]

#         ### Conclusion:

#         - Claim Status: [State "Supported," "Refuted," or "Not Enough Information"]
#         - Language: [Specify the language of the claim and context, do not translate.]
#         - Date: [Specify the date of the claim or context, e.g., "YYYY-MM-DD."]
#         - Country: [Reason the country relevant to the claim and transform it to the country code (e.g., US, UK, CA), only show the country code.]
#         - URL: [Provide the URL of the source for reference.]
#     """
#     prompt = PromptTemplate.from_template(template)
#     formatted_prompt = prompt.format(context=documents, claim=query)
    
#     # 將 prompt 傳遞給 LLM
#     time_start = timeit.default_timer()
#     result = llm.invoke(formatted_prompt)
#     time_end = timeit.default_timer()
#     url_pattern = r"(https?://[^\s]+)"
    
#     # 使用正則表達式搜尋 result 中是否有符合的連結
#     match = re.search(url_pattern, result)

#     if match:
#         # 如果有找到，返回結果和連結
#         if analyzer_time == None:
#             return result, time_end - time_start
#         else:
#             return result, time_end - time_start + analyzer_time
#     else:
#         return False
    
# with open('APIKey.json', 'r') as f:
#     config = json.load(f)

# search_engine_id = config.get('search_engine_id')
# search_api_key = config.get('search_api_key')

# query = [
#     "Barack Obama said Donald Trump 'tried to kill' Mike Pence.",
# ]

# result = []
# model = "llama3"
# def main(query, model, search_api_key, search_engine_id):
#     urls = get_fact_check_url(query[0], search_api_key, search_engine_id)
#     content = get_fact_check_content(urls)
#     documents, analyzer_time = analyze_fact_check(content, model)
#     fact_check_result = fact_check(query[0], documents, model, analyzer_time)
#     return fact_check_result[0], fact_check_result[1]

# result = main(query, model, search_api_key, search_engine_id)

# result = {
#     "ID": 1,
#     "claim": query[0],
#     "result": result[0],
#     "time_taken": result[1]
# }
# with open('result.json', 'w') as f:
#     json.dump(result, f, indent=4)

# from langchain_chroma import Chroma
# from langchain_huggingface import HuggingFaceEmbeddings
# from datetime import datetime
# import os

# def is_vector_db_exist(persist_directory, embeddings):
#     try:
#         if os.path.exists(persist_directory):
#             vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
#             return vectordb
#         else:
#             return None
#     except Exception as e:
#         print(f"Error loading database at {persist_directory}: {e}")
#         return None

# def create_vector_db(persist_directory, embeddings):
#     try:
#         vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
#         return vectordb
#     except Exception as e:
#         print(f"Error creating database at {persist_directory}: {e}")
#         return None
    
# def query_vectordb(question, questionDB, answerDB):
#     try:
#         # 獲取資料庫中的文件
#         db_documents = questionDB.get()['documents']
#         db_size = len(db_documents)
#     except Exception as e:
#         print(f"Error retrieving vector database documents: {str(e)}")
#         return "Error accessing the vector database."
    
#     if db_size == 0:
#         return "The vector database is empty."
    
#     try:
#         # 執行相似度檢索，選擇最接近的 k 篇文章
#         retriever = questionDB.similarity_search_with_score(
#             query=question,
#             k=min(db_size, 4),
#         )
#     except Exception as e:
#         print(f"Error executing similarity search: {str(e)}")
#         return
    
#     if retriever[0][1] < 0.5:
#         top_document_metadata = retriever[0][0].metadata['id']
#     else:
#         return "No relevant documents found."
#     try:
#         # 使用 metadata 查找相應的答案
#         answer = answerDB.get(ids=[top_document_metadata])['documents'][0]
#         return answer
#     except Exception as e:
#         print(f"Error retrieving answer from database: {str(e)}")
#         return

# embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# query = [
#     "Barack Obama said Donald Trump 'tried to kill' Mike Pence.",
# ]
# today = datetime.today().strftime('%Y-%m-%d')

# questionDB = is_vector_db_exist(f"{today}-question-db", embeddings)
# answerDB = is_vector_db_exist(f"{today}-answer-db", embeddings)

# if not questionDB:
#     questionDB = create_vector_db(f"{today}-question-db", embeddings)
# if not answerDB:
#     answerDB = create_vector_db(f"{today}-answer-db", embeddings)

# query_result = query_vectordb(query[0], questionDB, answerDB)

# print(query_result)
# import json

# def test_data1():
#     with open('fact_check.json', 'r') as f:
#         data = json.load(f)
#     test = []  # 用於保留順序
#     count = 0
#     for i in data:
#         if count > 100:
#             break
#         for j in i:
#             label = j['claimReview'][0]['textualRating'].lower()
#             if label == 'true' or label == 'false':
#                 claim = j['claim']
#                 if label == 'true':
#                     label = 'Supported'
#                 else:
#                     label = 'Refuted'
#                 if claim not in test:
#                     test.append([claim, label])
#             else:
#                 continue
#         count += 1
#     return test


# def test_data1():
#     with open('fact_check.json', 'r') as f:
#         data = json.load(f)
#     test = []  # 用於保留順序
#     for i in data:
#         for j in i:
#             claim = j['claim']
#             label = j['claimReview'][0]['textualRating'].lower()
#             test.append([claim, label])
#     return test

# test1 = test_data1()



# def test_data2():
#     with open('result.json', 'r') as f:
#         data = json.load(f)
#     test = []
#     for i in data:
#         claim = i['claim']
#         label = i['claim_status']
#         test.append([claim, label])

#     return test

# test2 = test_data2()

# claim = []
# result = []
# for i in test1:
#     for j in test2:
#         if i[0] == j[0]:
#             if i[0] not in claim:
#                 claim.append(i[0])
#                 result.append([i[0], i[1], j[1]])

# for i in result:
#     test = i[1]
#     label = i[2]
#     print(test, label)

# import json

# def test_data():
#     with open('fact_check.json', 'r') as f:
#         data = json.load(f)
#     test = []
#     Claim = []
#     cnt = 100
#     for i in data:
#         for j in i:
#             claim = j['claim']
#             label = j['claimReview'][0]['textualRating'].lower()
#             if label == 'true':
#                 label = 'Supported'
#             elif label == 'false':
#                 label = 'Refuted'
#             else:
#                 continue
#             if claim not in Claim:
#                 Claim.append(claim)
#                 test.append({"claim":claim, "label":label})
#                 cnt -= 1
#         if cnt == 0:
#             break

#     return test

# test = test_data()



# with open('test.json', 'w') as f:
#     json.dump(test, f, indent=4)

# def test_data():
#     with open('test.json', 'r') as f:
#         data = json.load(f)
#     test = []  # 用於保留順序
#     for i in data:
#         test.append(i['claim'])
#     return test

# test = test_data()
# print(len(test))
# ----------------------------------------------------------------------------
# import json

# def test1_data():
#     with open('test.json', 'r') as f:
#         data = json.load(f)
#     return data

# def test2_data():
#     with open('result.json', 'r') as f:
#         data = json.load(f)
#     return data

# test1 = test1_data()
# test2 = test2_data()



# test = []
# for i in test1:
#     for j in test2:
#         if i['claim'] == j['claim']:
#             test.append([i['claim'], i['label'], j['claim_status']])

# test2=[]
# for i in test:
#     if i[1] == i[2]:
#         test2.append(i)
# print(len(test2)) 
#----------------------------------------------------------------------------

# import json

# def test_data():
#     with open('/home/user/talen-python/data/Climate_Fever.json', 'r') as f:
#         data = json.load(f)
#     test = []  # 用於保留順序
#     for i in data:
#         test.append(i['claim'])

#     return test

# test = test_data()

# for i in test:
#     print(i)
# import re
# with open('/home/user/talen-python/data/MultiFC.tsv') as f:
#     data = f.readlines()

# test = []

# def clean_text(text):
#     # 去除 \uXXXX 類型的 Unicode 字元
#     text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
#     text = re.sub(r'\"', '', text)
#     return text

# for i in data:
#     # tmp = {
#     #     "claimID": i.split('\t')[0],
#     #     "claim": i.split('\t')[1],
#     #     "label": i.split('\t')[2],
#     #     "claimURL": i.split('\t')[3],
#     #     "reason": i.split('\t')[4],
#     #     "category": i.split('\t')[5],
#     #     "speaker": i.split('\t')[6],
#     #     "checker": i.split('\t')[7],
#     #     "tags": i.split('\t')[8],
#     #     "claimEntities": i.split('\t')[9],
#     #     "articleTitle": i.split('\t')[10],
#     #     "publishDate": i.split('\t')[11],
#     #     "claimDate": i.split('\t')[12],
#     # }
#     tmp = {
#         "claimID": clean_text(i.split('\t')[0]),
#         "claim": clean_text(i.split('\t')[1]),
#         "label": clean_text(i.split('\t')[2]),
#         "claimURL": clean_text(i.split('\t')[3]),
#         "reason": clean_text(i.split('\t')[4]),
#         "category": clean_text(i.split('\t')[5]),
#         "speaker": clean_text(i.split('\t')[6]),
#         "checker": clean_text(i.split('\t')[7]),
#         "tags": clean_text(i.split('\t')[8]),
#         "claimEntities": clean_text(i.split('\t')[9]),
#         "articleTitle": clean_text(i.split('\t')[10]),
#         "publishDate": clean_text(i.split('\t')[11]),
#         "claimDate": clean_text(i.split('\t')[12]),
#     }

#     test.append(tmp)

# for i in test:
#     print(i)

# import json

# with open('/home/user/talen-python/data/MultiFC.json', 'w') as f:
#     json.dump(test, f, indent=4)


#去除特殊符號
#"\"Six out of 10 of the highest unemployment rates are also in so-called right to work states.\""
#去除多餘""符號


# def clean_text(text):
#     # 去除 \uXXXX 類型的 Unicode 字元
#     text = re.sub(r'\\u[0-9a-fA-F]{4}', '', text)
#     text = re.sub(r'\"', '', text)
#     return text



# with open('fact_check.json', 'w') as f:
#     json.dump(test, f, indent=4, ensure_ascii=False)


# import json
# with open('/home/user/talen-python/data/MultiFC.json') as f:
#     data = json.load(f)

# test = []
# for i in data:
#     claim = i['claim']
#     label = i['label'].lower()
#     if label == "true":
#         test.append({"claim":claim,"label":"Supported"})
#     elif label == "false":
#         test.append({"claim":claim,"label":"Refuted"})
#     # elif label == 2:
#     #     test.append({"claim":claim,"label":"Not Enough Information"})
#     else:
#         continue

# with open('test3.json', 'w') as f:
#     json.dump(test, f, indent=4, ensure_ascii=False)

# import json
# with open('test3.json', 'r') as f:
#     data = json.load(f)

# list1=[]
# list2=[]
# list3=[]
# cnt = 0
# for i in data:
#     cnt+=1
#     if cnt<3248:
#         list1.append(i)
#     elif cnt > 3248 and cnt < 6496:
#         list2.append(i)
#     else:
#         list3.append(i)

# with open('test4.json', 'w') as f:
#     json.dump(list1, f, indent=4, ensure_ascii=False)

# with open('test5.json', 'w') as f:
#     json.dump(list2, f, indent=4, ensure_ascii=False)

# with open('test6.json', 'w') as f:
#     json.dump(list3, f, indent=4, ensure_ascii=False)

# import json
# with open('fact_check2.json', 'r') as f:
#     data1 = json.load(f)


# with open('fact_check3.json', 'r') as f:
#     data2 = json.load(f)


# with open('fact_check4.json', 'r') as f:
#     data3 = json.load(f)

# if isinstance(data1, list) and isinstance(data2, list) and isinstance(data3, list):
#     merged_data = data1 + data2 + data3

# with open('fact_check5.json', 'w') as f:
#     json.dump(merged_data, f, indent=4, ensure_ascii=False)

# with open('fact_check.json', 'r') as f:
#     # json.dump(merged_data, f, indent=4, ensure_ascii=False)
#     data1 = json.load(f)
# print(len(data1))
# with open('fact_check1.json', 'r') as f:
#     # json.dump(merged_data, f, indent=4, ensure_ascii=False)
#     data1 = json.load(f)
# print(len(data1))
# with open('fact_check5.json', 'r') as f:
#     # json.dump(merged_data, f, indent=4, ensure_ascii=False)
#     data1 = json.load(f)
# print(len(data1))

# import json
# def test_data(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     test = []  # 用於保留順序
#     for i in data:
#         for j in i:
#             if j['claim'] not in test:
#                 test.append(j['claim'])
#     return test
# path1 = '/home/user/talen-python/google fact check data/Climate_Fever.json'
# path2 = '/home/user/talen-python/google fact check data/AVeriTeC.json'
# path3 = '/home/user/talen-python/google fact check data/MultiFC.json'
# print(test_data(path1))
# print(len(test_data(path2)))
# print(len(test_data(path3)))

# import json

# test1 = []
# test2 = []
# test3 = []
# with open('result.json', 'r') as f:
#     data = json.load(f)

# for i in data:
#     test1.append([i['claim'],i['claim_status']])

# with open('/home/user/talen-python/google fact check data/Climate_Fever.json', 'r') as f:
#     data = json.load(f)
# cnt = 0
# for i in data:
#     if cnt >=100:
#         break
#     for j in i:
#         if j['claim'] not in test2:
#             test2.append(j['claim'])
#             test3.append([j['claim'],j['label']])
#             break
#     cnt+=1

# for i in test1:
#     for j in test3:
#         if i[0] == j[0]:
#             if i[1] == j[1]:
#                 print(i)
#                 print(j)
#                 print("-------------------------------------------")























# import requests
# import json
# import time
# import re
# def clean_claim_text(claim):
#     # 使用正則表達式，只保留字母（包括大小寫）和空格
#     cleaned_claim = re.sub(r'[^a-zA-Z0-9\s]', '', claim)
#     # 去掉多餘的空格，確保格式整潔
#     cleaned_claim = re.sub(r'\s+', ' ', cleaned_claim).strip()
#     return cleaned_claim


# def get_fact_check_url(query, api_key, search_engine_id, num_results=5):
#     query = clean_claim_text(query)
#     url = (
#         f"https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}&num={num_results}"
#     )
#     try:
#         response = requests.get(url, timeout=5)
#         response.raise_for_status()  # 確保請求成功
#         results = response.json()

#         if 'items' in results:
#             fact_check_urls = [item['link'] for item in results['items']]
#             return fact_check_urls
#         else:
#             print("No items found in the search results.")
#             return []
#     except requests.RequestException as e:
#         print(f"Error during API request: {e}")
#         return []
    
# path1 = '/home/user/talen-python/Climate_Fever.json'

# with open('APIKey.json', 'r') as f:
#     config = json.load(f)

# search_engine_id = config.get('search_engine_id')
# search_api_key = config.get('search_api_key')


# with open(path1,'r') as f:
#     data = json.load(f)

# test = []
# for i in data:
#     test.append(i)

# result = []
# for i in test:
#     if i['urls'] != []:
#         result.append(i)
#         continue
#     else:
#         time.sleep(1)
#         print(i)
#         urls = get_fact_check_url(i['claim'], search_api_key, search_engine_id)
#         tmp = {
#             "claim":i['claim'],
#             "urls":urls
#         }
#         result.append(tmp)

# with open('Climate_Fever1.json', 'w') as f:
#     json.dump(result, f, indent=4, ensure_ascii=False)

# import json
# with open('Climate_Fever.json', 'r') as f:
#     data = json.load(f)

# with open('Climate_Fever1.json', 'r') as f:
#     data1 = json.load(f)
# cnt=0
# for i in range(len(data)):
#     if data[i] != data1[i]:
#         cnt+=1

# print(cnt)
# for i in data:
#     for j in i:
#         claim = j['claim']
#         if claim not in test:
#             test.append(j['claim'])
#             break

# import json
# with open('/home/user/talen-python/google fact check data/Climate_Fever.json','r') as f:
#     data = json.load(f)

# with open('result.json', 'r') as f:
#     data1 = json.load(f)
# test = []
# for i in data1:
#     for j in data:
#         if i['claim'] == j[0][]
# import json
# with open('AVeriTeC1.json', 'r') as f:
#     data = json.load(f)

# test = []
# test1 = []
# test2 = []
# cnt = 0
# for i in data:
#     if cnt < 630:
#         test.append(i)
#     elif cnt >= 630 and cnt < 1260:
#         test1.append(i)
#     else:
#         test2.append(i)
#     cnt+=1

# with open('AVeriTeC2.json', 'w') as f:
#     json.dump(test, f, indent=4, ensure_ascii=False)

# with open('AVeriTeC3.json', 'w') as f:
#     json.dump(test1, f, indent=4, ensure_ascii=False)

# with open('AVeriTeC4.json', 'w') as f:
#     json.dump(test2, f, indent=4, ensure_ascii=False)

# test = 1

# while test ==1:
#     print(test)
#     test+=1
# import json
# with open('/home/user/talen-python/result.json', 'r') as f:
#     data = json.load(f)
# print(len(data))
# test = []
# test1 = []
# test2 = []
# cnt=0
# for i in data:
#     if cnt < 1429:
#         test.append(i)
#     elif cnt >= 1429 and cnt < 2858:
#         test1.append(i)
#     else:
#         test2.append(i)
#     cnt+=1

# with open('MultiFC2.json', 'w') as f:
#     json.dump(test, f, indent=4, ensure_ascii=False)
# with open('MultiFC3.json', 'w') as f:
#     json.dump(test1, f, indent=4, ensure_ascii=False)
# with open('MultiFC4.json', 'w') as f:
#     json.dump(test2, f, indent=4, ensure_ascii=False)

import json
from langchain_ollama import OllamaLLM
from langchain.prompts import PromptTemplate


with open('/home/user/talen-python/Climate_Fever/Climate_Fever_content.json', 'r') as f:
    data = json.load(f)

with open('/home/user/talen-python/Climate_Fever/Climate_Fever.json', 'r') as f:
    data1 = json.load(f)

print(len(data))
for i in data1:
    flag = 1
    for j in data:
        if i['claim']==j:
            flag = 0
            break
    if flag == 1:
        print(i,"FUCK")


# model = "llama3"

# for i in data['Donald Trump delivered the largest tax cuts in American history.']:
#     llm = OllamaLLM(model=model)
#     # 定義 PromptTemplate
#     template = """
#     Write a concise summary of the following:
#     {context}
#     """
#     prompt = PromptTemplate.from_template(template)
#     # 格式化 prompt
#     formatted_prompt = prompt.format(context=i)
#     # 執行 LLM
#     result = llm.invoke(formatted_prompt)

#     print(result)
#     print("-----------------------------------------------")