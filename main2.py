import requests
import json
import os
import re
from fake_useragent import UserAgent

# 初始化 UserAgent 物件
ua = UserAgent()
os.environ['USER_AGENT'] = ua.random
from langchain_community.document_loaders import WebBaseLoader
from langchain.prompts import PromptTemplate
import requests
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
import timeit
from fp.fp import FreeProxy
from datetime import datetime
import urllib3
import signal
from time import sleep
import threading

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")




def is_vector_db_exist(persist_directory, embeddings):
    try:
        if os.path.exists(persist_directory):
            vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
            return vectordb
        else:
            return None
    except Exception as e:
        print(f"Error loading database at {persist_directory}: {e}")
        return None

def create_vector_db(persist_directory, embeddings):
    try:
        vectordb = Chroma(embedding_function=embeddings, persist_directory=persist_directory)
        return vectordb
    except Exception as e:
        print(f"Error creating database at {persist_directory}: {e}")
        return None
    
def store_to_vectordb(query,result, questionDB, answerDB, extracted_data=None):
    if not result:
        return
    try:
        # 將答案存入 answerDB，並記錄 token 作為 metadata
        answer_doc = Document(page_content=result,metadata={
            "Language": str(extracted_data.get("Language")),
            "Date": str(extracted_data.get("Date")),
            "Country": str(extracted_data.get("Country")),
            "URL": str(extracted_data.get("URL"))
        })
        token = answerDB.add_documents([answer_doc])[0]
        # 將問題存入 questionDB，並記錄 token 作為 metadata
        question_doc = Document(page_content=query, metadata={
            "id": token,
            "Language": str(extracted_data.get("Language")),
            "Date": str(extracted_data.get("Date")),
            "Country": str(extracted_data.get("Country")),
            "URL": str(extracted_data.get("URL"))
        })
        questionDB.add_documents([question_doc])
        
    except Exception as e:
        print(f"Error storing result to database: {e}")


def query_vectordb(question, questionDB, answerDB):
    try:
        # 獲取資料庫中的文件
        db_documents = questionDB.get()['documents']
        db_size = len(db_documents)
    except Exception as e:
        print(f"Error retrieving vector database documents: {str(e)}")
        return "Error accessing the vector database."
    
    if db_size == 0:
        return "The vector database is empty."
    
    try:
        # 執行相似度檢索，選擇最接近的 k 篇文章
        retriever = questionDB.similarity_search_with_score(
            query=question,
            k=min(db_size, 4),
        )
    except Exception as e:
        print(f"Error executing similarity search: {str(e)}")
        return
    
    if retriever[0][1] < 0.5:
        top_document_metadata = retriever[0][0].metadata['id']
    else:
        return "No relevant documents found."
    try:
        # 使用 metadata 查找相應的答案
        answer = answerDB.get(ids=[top_document_metadata])['documents'][0]
        return answer
    except Exception as e:
        print(f"Error retrieving answer from database: {str(e)}")
        return


def get_fact_check_url(query, api_key, search_engine_id, num_results=5):
    url = (
        f'https://www.googleapis.com/customsearch/v1?q={query}&key={api_key}&cx={search_engine_id}&num={num_results}'
    )
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()  # 確保請求成功
        results = response.json()

        if 'items' in results:
            fact_check_urls = [item['link'] for item in results['items']]
            return fact_check_urls
        else:
            print("No items found in the search results.")
            return []
    except requests.RequestException as e:
        print(f"Error during API request: {e}")
        return []
    

def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    result = [None]  # To store the function result
    exception = [None]  # To capture any exception

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)  # Wait for execution to complete

    if thread.is_alive():  # Check if the function timed out
        print("Function execution timed out.")
        return None
    if exception[0]:  # Check for any exceptions
        print(f"Error during execution: {exception[0]}")
        return None
    return result[0]

def web_loader(url, verify_ssl=False):
    try:
        proxy = FreeProxy(rand=True, timeout=3).get()  # Get a random proxy
        loader = WebBaseLoader(
            url,
            proxies={"http": proxy, "https": proxy},
            verify_ssl=verify_ssl,
        )
        docs = loader.load()  # Load the content
        return docs
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return None

def get_fact_check_content(urls, max_retries=3):
    if not urls:
        print("No URLs provided.")
        return []

    print('-' * 50)
    print('Searching relevant info...')
    fact_check_content = []

    for i, url in enumerate(urls, start=1):
        for retries in range(max_retries):
            docs = run_with_timeout(web_loader, args=(url,))
            if docs:  # Successfully fetched content
                print(f"[{i}/{len(urls)}] Successfully fetched content from {url}")
                fact_check_content.append([docs, url])
                break
            else:  # Retry failed
                print(f"[{i}/{len(urls)}] Attempt {retries + 1} failed. Retrying...")
        else:  # Exceeded max retries
            print(f"[{i}/{len(urls)}] Failed to load content from {url} after {max_retries} attempts.")

    print('Searching completed.')
    print('-' * 50)
    return fact_check_content
    
def analyze_fact_check(fact_check_content, model):
    if fact_check_content == None:
        return("There are no fact check content.")
    info = []
    time_start = timeit.default_timer()
    for i in fact_check_content:
        # 初始化 LLM
        llm = OllamaLLM(model=model)
        # 定義 PromptTemplate
        template = """
        Write a concise summary of the following:
        {context}
        """
        prompt = PromptTemplate.from_template(template)
        # 格式化 prompt
        formatted_prompt = prompt.format(context=i[0])
        # 執行 LLM
        result = llm.invoke(formatted_prompt)
        # 將結果與 URL 結合
        info.append(f"{result}, url: {i[1]}")
    time_end = timeit.default_timer()
    # 將結果轉為 Document 格式
    documents = [Document(page_content=item) for item in info]

    return documents, time_end - time_start


def fact_check(query, documents, model,analyzer_time=None):
    if not documents:
        return
    llm = OllamaLLM(model=model)
    # 定義 prompt
    template = """
        You are a professional fact-checker tasked with evaluating the following claim.
        Let's break down the evidence and reasoning step by step.
        First, analyze the provided context {context} and identify key information relevant to the claim {claim}.
        Then, evaluate the evidence step by step to determine if the claim is true or false.
        Finally, structure the response in the following format:
        

        ### Analysis of Claim:

        - Key Information from Context:  
        [Summarize the key points from the context relevant to the claim.]

        ### Step-by-Step Evaluation:

        1. Evidence 1:  
        - Observation: [Detail the first piece of evidence or data relevant to the claim.]  
        - Reasoning: [Explain how this evidence supports or refutes the claim, or note any limitations.]

        2. Evidence 2:  
        - Observation: [Detail the second piece of evidence or data relevant to the claim.]  
        - Reasoning: [Explain how this evidence supports or refutes the claim, or note any limitations.]

        3. Additional Analysis (if needed):  
        - [Integrate multiple pieces of evidence or consider other contextual factors for deeper reasoning.]

        ### Conclusion:

        - Claim Status: [State "Supported," "Refuted," or "Not Enough Information"]
        - Language: [Specify the language of the claim and context, do not translate.]
        - Date: [Specify the date of the claim or context, e.g., "YYYY-MM-DD."]
        - Country: [Reason the country relevant to the claim and transform it to the country code (e.g., US, UK, CA), only show the country code.]
        - URL: [Provide the URL of the source for reference.]
    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(context=documents, claim=query)
    
    # 將 prompt 傳遞給 LLM
    time_start = timeit.default_timer()
    result = llm.invoke(formatted_prompt)
    time_end = timeit.default_timer()
    url_pattern = r"(https?://[^\s]+)"
    
    # 使用正則表達式搜尋 result 中是否有符合的連結
    match = re.search(url_pattern, result)

    if match:
        # 如果有找到，返回結果和連結
        if analyzer_time == None:
            return result, time_end - time_start
        else:
            return result, time_end - time_start + analyzer_time
    else:
        return False
    
patterns = {
    "Claim Status": [r"Claim Status:\s*([^\n]+)", r"\*\*Claim Status\*\*:\s*([^\n]+)"],
    "Language": [r"Language:\s*([^\(\n]+)", r"\*\*Language\*\*:\s*([^\(\n]+)"],
    "Date": [r"Date:\s*([^\(\n]+)", r"\*\*Date\*\*:\s*([^\(\n]+)"],
    "Country": [r"Country:\s*([A-Z]+)", r"\*\*Country\*\*:\s*([A-Z]+)"],
    "URL": [r"URL:\s*(https?://[^\s]+)", r"https?://[^\s]+"]
}

def extract_data(response, patterns):
    result = {}

    # 檢查 response 是否為字串
    if not isinstance(response, str):
        print(f"Skipping non-string response: {response} (type: {type(response)})")
        return None
    
    for key, regex_list in patterns.items():
        if key == "URL":
            matches = []
            for regex in regex_list:
                matches.extend(re.findall(regex, response))
            # 使用 set 去重，然後轉回 list
            result[key] = list(set(matches)) if matches else None
        else:
            for regex in regex_list:
                match = re.search(regex, response)
                if match:
                    result[key] = match.group(1).strip()
                    break
            if key not in result:
                result[key] = None
    return result

def main(query, model, search_api_key, search_engine_id):
    # urls = get_fact_check_url(query, search_api_key, search_engine_id)
    urls = query[1]
    if urls == []:
        return "No relevant fact-checking articles found."
    content = get_fact_check_content(urls)
    if content == None:
        print("Failed to get fact check content.")
        return None
    documents, analyzer_time = analyze_fact_check(content, model)
    fact_check_result = fact_check(query[0], documents, model, analyzer_time)
    while fact_check_result == False:
        fact_check_result = fact_check(query[0], documents, model, analyzer_time)
    return fact_check_result[0], fact_check_result[1]

# def test_data(path):
#     with open(path, 'r') as f:
#         data = json.load(f)
#     test = []  # 用於保留順序
#     cnt = 0
#     for i in data:
#         if cnt >= 100:
#             break
#         for j in i:
#             if j['claim'] not in test:
#                 test.append(j['claim'])
#         cnt+=1
#     return test

def test_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

path1 = '/home/user/talen-python/Climate_Fever/Climate_Fever_content.json'


    
with open('APIKey.json', 'r') as f:
    config = json.load(f)

search_engine_id = config.get('search_engine_id')
search_api_key = config.get('search_api_key')


today = datetime.today().strftime('%Y-%m-%d')

urls = test_data(path1)

save_result = []
query = []
for i in urls:
    if i['urls'] != []:
        query.append([i['claim'],i['urls']])

model = "llama3"

# questionDB = is_vector_db_exist(f"{today}-question-db", embeddings)
# answerDB = is_vector_db_exist(f"{today}-answer-db", embeddings)

# if not questionDB:
#     questionDB = create_vector_db(f"{today}-question-db", embeddings)
# if not answerDB:
#     answerDB = create_vector_db(f"{today}-answer-db", embeddings)




for i in query:
    print(f"Processing claim: {i[0]}")
    # query_result = query_vectordb(i[0], questionDB, answerDB)
    # if query_result != "No relevant documents found." and query_result != "The vector database is empty.":
    #     print("Result found in database:")
    #     print(query_result)
    # else:
    #     print("No relevant documents found in the database. Searching online...")
    result = main(i, model, search_api_key, search_engine_id)
    if result == None:
        print("No relevant fact-checking articles found.")
        continue
    extract_data_result = extract_data(result[0], patterns)
    tmp = {
        "claim": i[0],
        "result": result[0],
        "claim_status": extract_data_result["Claim Status"],
        "language": extract_data_result["Language"],
        "date": extract_data_result["Date"],
        "country": extract_data_result["Country"],
        "url": extract_data_result["URL"],
        "time_taken": result[1]
    }
    print("Result:")
    print(tmp['result'])
    # store_to_vectordb(i[0], result[0], questionDB, answerDB, extracted_data=extract_data_result)
    save_result.append(tmp)

with open('result.json', 'w') as f:
    json.dump(save_result, f, indent=4, ensure_ascii=False)

