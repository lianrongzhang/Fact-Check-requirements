import os
import argparse
from fake_useragent import UserAgent
from langchain_community.document_loaders import WebBaseLoader
from fp.fp import FreeProxy
import threading
import json
import urllib3

# Disable InsecureRequestWarning warnings
urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

# Initialize UserAgent object
ua = UserAgent()
os.environ['USER_AGENT'] = ua.random

def run_with_timeout(func, args=(), kwargs={}, timeout=10):
    result = [None]
    exception = [None]

    def target():
        try:
            result[0] = func(*args, **kwargs)
        except Exception as e:
            exception[0] = e

    thread = threading.Thread(target=target)
    thread.daemon = True
    thread.start()
    thread.join(timeout)

    if thread.is_alive():
        print("Function execution timed out.")
        return None
    if exception[0]:
        print(f"Error during execution: {exception[0]}")
        return None
    return result[0]

def web_loader(url, verify_ssl=False):
    try:
        proxy = FreeProxy(rand=True, timeout=3).get()
        loader = WebBaseLoader(
            url,
            proxies={"http": proxy, "https": proxy},
            verify_ssl=verify_ssl,
        )
        docs = loader.load()
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
            if docs:
                print(f"[{i}/{len(urls)}] Successfully fetched content from {url}")
                fact_check_content.append([docs, url])
                break
            else:
                print(f"[{i}/{len(urls)}] Attempt {retries + 1} failed. Retrying...")
        else:
            print(f"[{i}/{len(urls)}] Failed to load content from {url} after {max_retries} attempts.")

    print('Searching completed.')
    print('-' * 50)
    return fact_check_content

def main(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    query = []
    save_result = []
    for i in data:
        if i['urls']:
            query.append([i['claim'], i['urls']])

    for i in query:
        print(f"Processing claim: {i[0]}")
        result = get_fact_check_content(i[1])
        save_result.append(result)

    with open(output_path, 'w') as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process fact-checking claims.")
    parser.add_argument(
        "--input_path",
        type=str,
        required=True,
        help="Path to the input JSON file containing claims and URLs."
    )
    parser.add_argument(
        "--output_path",
        type=str,
        required=True,
        help="Path to save the output JSON file with fact-check content."
    )

    args = parser.parse_args()

    main(args.input_path, args.output_path)
