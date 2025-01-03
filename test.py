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

from langchain_community.document_loaders import WebBaseLoader

# 初始化 WebBaseLoader
loader = WebBaseLoader('https://tfc-taiwan.org.tw/articles/11337')

# 載入數據
data = loader.load()

# 輸出數據的前幾行

print(data)
