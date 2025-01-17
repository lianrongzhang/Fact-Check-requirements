import os
import argparse
from fake_useragent import UserAgent
from langchain_community.document_loaders import WebBaseLoader
from fp.fp import FreeProxy
import threading
import json
import urllib3
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document  # Assuming you're using langchain for Document format


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
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers)
        
        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        plain_text = soup.get_text()  # Extract text and separate lines
        
        # Create a LangChain Document object
        doc = Document(
            page_content=plain_text,
            metadata={"source": url}  # Optionally, add the URL as metadata
        )
        return doc  # Return the document directly
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return None

def get_fact_check_content(urls, max_retries=10):
    if not urls:
        print("No URLs provided.")
        return []

    print('-' * 50)
    print('Searching relevant info...')
    fact_check_content = []

    for i, url in enumerate(urls, start=1):
        for retries in range(max_retries):
            doc = run_with_timeout(web_loader, args=(url,))
            if doc:
                print(f"[{i}/{len(urls)}] Successfully fetched content from {url}")
                fact_check_content.append(doc)
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
    save_result = dict()

    for i in data:
        if i['urls']:
            query.append([i['claim'], i['urls']])
    
    print(len(query))
    for i in query:
        print(f"Processing claim: {i[0]}")
        result = get_fact_check_content(i[1])
        save_result[i[0]] = result  # Store LangChain Document objects in the result

    # Save the LangChain documents as JSON with metadata (if desired)
    serialized_result = {
        claim: [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        for claim, docs in save_result.items()
    }

    with open(output_path, 'w') as f:
        json.dump(serialized_result, f, indent=4, ensure_ascii=False)

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
