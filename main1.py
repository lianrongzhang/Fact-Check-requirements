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
from datetime import datetime
import json
import re
import timeit
from fp.fp import FreeProxy
import urllib3
import signal

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


def get_fact_check_content(urls, max_retries=3):
    if not urls:
        print("No URLs provided.")
        return []

    print('-' * 50)
    print('Searching relevant info...')
    fact_check_content = []

    for i, url in enumerate(urls, start=1):
        retries = 0
        while retries < max_retries:
            try:
                # 嘗試獲取代理
                proxy = FreeProxy(rand=True, timeout=3).get()
                print(f"[{i}/{len(urls)}] Attempt {retries + 1}: Using proxy: {proxy}")
                # 加載網頁內容
                docs = run_with_timeout(web_loader, args=(url,), kwargs={"proxies": {"http": proxy, "https": proxy}})
                fact_check_content.append([docs, url])
                break  # 如果成功，跳出重試循環
            except Exception as e:
                print(f"[{i}/{len(urls)}] Attempt {retries + 1} failed for {url}: {e}")
                retries += 1

        if retries == max_retries:
            print(f"[{i}/{len(urls)}] Failed to load {url} after {max_retries} attempts.")

    print('Searching completed.')
    print('-' * 50)
    return fact_check_content

def web_loader(url, proxies=None, verify_ssl=False):
    try:
        loader = WebBaseLoader(url, proxies=proxies, verify_ssl=verify_ssl)
        docs = loader.load()
        return docs
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return None


def analyze_fact_check(fact_check_content, model):
    if not fact_check_content:
        return
    info = []
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
        time_start = timeit.default_timer()
        result = llm.invoke(formatted_prompt)
        time_end = timeit.default_timer()
        # 將結果與 URL 結合
        info.append(f"{result}, url: {i[1]}")

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

def main(query, model, search_api_key, search_engine_id):
    today = datetime.today().strftime('%Y-%m-%d')
    questionDB = is_vector_db_exist(f"{today}-question-db", embeddings)
    answerDB = is_vector_db_exist(f"{today}-answer-db", embeddings)
    fact_check_urls = get_fact_check_url(query, search_api_key, search_engine_id)
    if fact_check_urls == None:
        return None
    fact_check_content = get_fact_check_content(fact_check_urls)
    documents, analyzer_time = analyze_fact_check(fact_check_content, model)
    fact_check_result = fact_check(query[0], documents, model, analyzer_time)
    while fact_check_result == False:
        fact_check_result = fact_check(query[0], documents, model, analyzer_time)
    extracted_data = extract_data(fact_check_result[0], patterns)
    store_to_vectordb(query,fact_check_result[0], questionDB, answerDB, extracted_data)
    return fact_check_result[0], fact_check_result[1], extracted_data


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

def timeout_handler(signum, frame):
    raise TimeoutError()

def run_with_timeout(func, args=(), kwargs={}, timeout=15):
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)
    try:
        result = func(*args, **kwargs)
    except TimeoutError:
        result = None
    finally:
        signal.alarm(0)
    return result

def MultiFC():
    # 開啟並讀取 JSON 檔案
    file_path = "/home/user/talen-python/data/AVeriTeC.json"  # 替換為你的 JSON 檔案路徑
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)  # 將 JSON 資料讀取並轉換為 Python 字典或列表
    except FileNotFoundError:
        print(f"找不到檔案：{file_path}")
    except json.JSONDecodeError as e:
        print(f"解析 JSON 檔案時發生錯誤：{e}")
    test = []
    for i in data:
        test.append(i["claim"])
    return test

    
# query = MultiFC()
# query = "Barack Obama said Donald Trump 'tried to kill' Mike Pence"
# query = "Kas kaitsevägi lasi õhku äsja 12 miljoni eest renoveeritud viadukti?"
# query = "Old clip of cargo ship fire falsely linked to Houthi attacks"
# query = "Vietăți marine care nu există"
# query = "nee, dieselauto’s zijn niet even milieuvriendelijk als elektrische wagens"
# query = "nee, deze video toont niet hoe een man stembiljetten voor Trump vernietigt"
# query = "het klopt dat je beter je neus ophaalt dan hem te snuiten"
# query = "حزب الله يستهدف مواقع إسرائيل العسكرية في جنوب لبنان"
# query = "The non-partisan Congressional Budget Office concluded ObamaCare will cost the U.S. more than 800,000 jobs."
# query = "中東和烏克蘭戰死的美軍棺木回國"

query = [
    "Barack Obama said Donald Trump 'tried to kill' Mike Pence",
]

today = datetime.today().strftime('%Y-%m-%d')

with open('APIKey.json', 'r') as f:
    config = json.load(f)

search_engine_id = config.get('search_engine_id')
search_api_key = config.get('search_api_key')

questionDB = is_vector_db_exist(f"{today}-question-db", embeddings)
answerDB = is_vector_db_exist(f"{today}-answer-db", embeddings)

if not questionDB:
    questionDB = create_vector_db(f"{today}-question-db", embeddings)
if not answerDB:
    answerDB = create_vector_db(f"{today}-answer-db", embeddings)

model = "llama3"
result = []
for i in query:
    print("Query:")
    print(i)
    # print(query_result)
    query_result = query_vectordb(i, questionDB, answerDB)
    if query_result != "No relevant documents found." and query_result != "The vector database is empty.":
        print("Result from local database:")
        print(query_result)
    else:
        print(query_result)
        try:
            main_result = main(i[0], model, search_api_key, search_engine_id)
            if main_result == None:
                print("No results from Google search.")
            else:
                while main_result == False:
                    main_result = main(i, model, search_api_key, search_engine_id)
                
                extract_data_result = extract_data(main_result[0], patterns)
                tmp = {
                    "Claim": i,
                    "Result": main_result[0],
                    "Claim Status": extract_data_result.get("Claim Status"),
                    "Language": extract_data_result.get("Language"),
                    "Date": extract_data_result.get("Date"),
                    "Country": extract_data_result.get("Country"),
                    "URL": extract_data_result.get("URL"),
                    "Time taken": main_result[1]
                }
                result.append(tmp)
                print("Result from Google search:")
                print(main_result[0])
                print(f"Time taken: {main_result[1]:.2f} seconds")
                print()
        except Exception as e:
            print(f"Failed to get result from Google search: {e}")

with open('result.json', 'w') as f:
    json.dump(result, f, indent=4)







def google_fact_check(query, API_KEY):
    # Base URL for Fact Check API
    url = 'https://factchecktools.googleapis.com/v1alpha1/claims:search'

    # Parameters for the API request, including languageCode
    params = {
        'query': query,
        'languageCode': 'all',  # Specify language code here (e.g., 'en' for English, 'zh' for Chinese)
        'key': API_KEY,
        'page_size': 5
    }
    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()
        if 'claims' in data:
            processed_claims = []
            seen_claims = set()  # To keep track of unique claims

            for index, claim in enumerate(data['claims']):
                claim_text = claim.get('text')
                # Check if the claim has already been seen
                if claim_text not in seen_claims:
                    seen_claims.add(claim_text)  # Add claim to the set

                    # Create a dictionary for each claim with relevant details
                    claim_info = {
                        'ID': index + 1,
                        'Claim': claim_text,
                        'Claimant': claim.get('claimant'),
                        'Claim Date': claim.get('claimDate'),
                        'Claim Review': []
                    }

                    # Process each review
                    for review in claim.get('claimReview', []):
                        review_info = {
                            'Publisher': review.get('publisher', {}).get('name'),
                            'Site': review.get('publisher', {}).get('site'),
                            'Review Date': review.get('reviewDate'),
                            'Title': review.get('title'),
                            'Rating': review.get('textualRating'),
                            'URL': review.get('url')
                        }
                        claim_info['Claim Review'].append(review_info)

                    processed_claims.append(claim_info)
            return [Document(page_content=str(item)) for item in processed_claims]
        else:
            return None
    else:
            print(f"Error: {response.status_code}, {response.json()}")