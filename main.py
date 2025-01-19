import os
import re
from fake_useragent import UserAgent

# 初始化 UserAgent 物件
ua = UserAgent()
os.environ['USER_AGENT'] = ua.random

from langchain_community.document_loaders import WebBaseLoader
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
import requests
from langchain_ollama import OllamaLLM
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from datetime import datetime
from evaluate import load
from langchain_community.retrievers import TFIDFRetriever
from sklearn.metrics.pairwise import cosine_similarity
import re

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
    
def get_fact_check_url(query):
    API_KEY = ''
    SEARCH_ENGINE_ID = ''
    query = query
    url = f'https://www.googleapis.com/customsearch/v1?q={query}&key={API_KEY}&num=5&cx={SEARCH_ENGINE_ID}'

    fact_check_urls = []
    response = requests.get(url)
    results = response.json()
    if 'items' in results:
        for item in results['items']:
            fact_check_urls.append(item['link'])
        return fact_check_urls
    else:
        return None

def get_fact_check_content(urls):
    if urls == None:
        return None
    print('-'*50)
    print('searching relavant info...')
    fact_check_content = []
    for i in urls:
        loader = WebBaseLoader(i)
        docs = loader.load()
        fact_check_content.append([docs, i])
    print('searching completed.')
    print('-'*50)
    return fact_check_content

def analyze_fact_check(fact_check_content):
    if not fact_check_content:
        return
    info = []
    for i in fact_check_content:
        # 初始化 LLM 和 Prompt
        llm = OllamaLLM(model="llama3")
        # Define prompt
        prompt = ChatPromptTemplate.from_messages([
            ("system", "Write a concise summary of the following:\\n\\n{context}"),
        ])

        # 建立處理鏈
        chain = create_stuff_documents_chain(llm, prompt)
        # Instantiate chain

        # Invoke chain
        result = chain.invoke({"context": i[0]})
        
        info.append(f"{result}, url: {i[1]}")

    documents = [Document(page_content=item) for item in info]

    return documents


def fact_check(query, documents):
    if not documents:
        return
    llm = OllamaLLM(model="llama3")
    # 定義 prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", """
            First, analyze the provided context {context} and identify key information relevant to the claim {claim}.
            Then, evaluate the evidence step by step to determine if the claim is true or false.
            Finally, structure the response in the following format:
            Generate answers based on the input language.

            ### Analysis of Claim:

            - **Key Information from Context**:  
            [Summarize the key points from the context relevant to the claim.]

            ### Step-by-Step Evaluation:

            1. **Evidence 1**:  
            - **Observation**: [Detail the first piece of evidence or data relevant to the claim.]  
            - **Reasoning**: [Explain how this evidence supports or refutes the claim, or note any limitations.]

            2. **Evidence 2**:  
            - **Observation**: [Detail the second piece of evidence or data relevant to the claim.]  
            - **Reasoning**: [Explain how this evidence supports or refutes the claim, or note any limitations.]

            3. **Additional Analysis** (if needed):  
            - [Integrate multiple pieces of evidence or consider other contextual factors for deeper reasoning.]

            ### Conclusion:

            - Claim Status: [State "Supported," "Refuted," or "Not Enough Information"]
            - Language: [Specify the language of the claim and context, do not translate.]
            - Date: [Specify the date of the claim or context, e.g., "YYYY-MM-DD."]
            - Country: [Reason the country relevant to the claim and transform it to the country code (e.g., US, UK, CA), only show the country code.]
            - URL: [Provide the URL of the source for reference.]
        """),
        ("assistant", "Let's break down the evidence and reasoning step by step:")
    ])
    # 建立處理鏈
    chain = create_stuff_documents_chain(llm, prompt)
    # 調用鏈
    result = chain.invoke({"context": documents, "claim": query})
    url_pattern = r"(https?://[^\s]+)"
    
    # 使用正則表達式搜尋 result 中是否有符合的連結
    match = re.search(url_pattern, result)

    if match:
        # 如果有找到，返回結果和連結
        return result
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

def google_fact_check(query):
    API_KEY = ''

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

def main(query):
    today = datetime.today().strftime('%Y-%m-%d')
    questionDB = is_vector_db_exist(f"{today}-question-db", embeddings)
    answerDB = is_vector_db_exist(f"{today}-answer-db", embeddings)
    fact_check_urls = get_fact_check_url(query)
    if fact_check_urls == None:
        return None
    fact_check_content = get_fact_check_content(fact_check_urls)
    documents = analyze_fact_check(fact_check_content)
    result = fact_check(query, documents)
    extracted_data = extract_data(result, patterns)
    store_to_vectordb(query,result, questionDB, answerDB, extracted_data)
    return result

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

    
# Main query execution
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


today = datetime.today().strftime('%Y-%m-%d')

questionDB = is_vector_db_exist(f"{today}-question-db", embeddings)
answerDB = is_vector_db_exist(f"{today}-answer-db", embeddings)

if questionDB and answerDB:
    print("Databases exist.")
else:
    print("Databases don't exist. Creating new databases.")
    if not questionDB:
        questionDB = create_vector_db(f"{today}-question-db", embeddings)
    if not answerDB:
        answerDB = create_vector_db(f"{today}-answer-db", embeddings)

query_result = query_vectordb(query, questionDB, answerDB)
# print(query_result)
if query_result != "No relevant documents found." and query_result != "The vector database is empty.":
    print("Result from local database:")
    print(query_result)
else:
    print(query_result)
    # 使用 Google Fact Check 進行查詢
    google_result = google_fact_check(query)
    if google_result:
        # 如果 Google Fact Check 有結果，進行處理
        print("Result from Google Fact Check:")
        result = fact_check(query, google_result)
        print(result)
        try:
            extracted_data = extract_data(result, patterns)
        except Exception as e:
            print(f"Failed to extract data from the result: {e}")
        # 嘗試將結果儲存到向量資料庫
        try:
            vector_store = store_to_vectordb(query,result, questionDB, answerDB, extracted_data)
        except Exception as e:
            print(f"Failed to store result to vector database: {e}")

    else:
        print("No results from Google Fact Check.")
        try:
            main_result = main(query)
            if main_result == None:
                print("No results from Google search.")
            else:
                while main_result == False:
                    main_result = main(query)
                print("Result from Google search:")
                print(main_result)
        except Exception as e:
            print(f"Failed to get result from Google search: {e}")
