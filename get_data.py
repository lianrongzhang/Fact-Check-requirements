import os
from fake_useragent import UserAgent

# 初始化 UserAgent 物件
ua = UserAgent()
os.environ['USER_AGENT'] = ua.random
from langchain_community.document_loaders import WebBaseLoader
from fp.fp import FreeProxy
import threading
import json
import urllib3

# 禁用 InsecureRequestWarning 警告
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

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

def main(query):
    urls = query[1]
    if urls == []:
        return "No relevant fact-checking articles found."
    
    content = get_fact_check_content(urls)
    if content == None:
        print("Failed to get fact check content.")
        return None
    
    return content

with open('AVeriTeC/AVeriTeC.json', 'r') as f:
    data = json.load(f)

query = []
save_result = []
for i in data:
    if i['urls'] != []:
        query.append([i['claim'],i['urls']])

for i in query:
    print(f"Processing claim: {i[0]}")
    result = main(i)
    save_result.append(result)

with open('AVeriTeC/AVeriTeC_content.json', 'w') as f:
    json.dump(save_result, f, indent=4, ensure_ascii=False)