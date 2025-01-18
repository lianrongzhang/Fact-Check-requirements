import os
import argparse
import json
import threading
import time
from fake_useragent import UserAgent
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document


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

def web_loader(url):
    try:
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers)

        # Parse the HTML using BeautifulSoup
        soup = BeautifulSoup(response.text, 'html.parser')
        plain_text = soup.get_text()  # Extract plain text content

        # Create a LangChain Document object
        doc = Document(
            page_content=plain_text,
            metadata={"source": url}  # Optionally add the URL to metadata
        )
        return doc  # Return the document directly
    except Exception as e:
        print(f"Error loading content from {url}: {e}")
        return None

def get_fact_check_content(urls, max_retries=10):
    if not urls:
        print("No URLs provided.")
        return [], []

    print('-' * 50)
    print('Searching relevant information...')
    fact_check_content = []
    failed_urls = []

    for i, url in enumerate(urls, start=1):
        for retries in range(max_retries):
            doc = run_with_timeout(web_loader, args=(url,))
            if doc:
                print(f"[{i}/{len(urls)}] Successfully fetched content from {url}")
                fact_check_content.append(doc)
                break
            else:
                print(f"[{i}/{len(urls)}] Attempt {retries + 1} failed. Retrying...")
            time.sleep(1)
        else:
            print(f"[{i}/{len(urls)}] Failed to load content from {url} after {max_retries} attempts.")
            fact_check_content.append(Document(page_content="", metadata={"source": url}))
            failed_urls.append(url)

    print('Search completed.')
    print('-' * 50)
    return fact_check_content, failed_urls

def retry_failed_urls(failed_urls, output_path):
    print("Retrying failed URLs...")
    retried_content = []
    for url in failed_urls:
        doc = run_with_timeout(web_loader, args=(url,))
        if doc:
            print(f"Successfully fetched content from {url} (Retry successful).")
            retried_content.append(doc)
        else:
            print(f"Failed to fetch content from {url} even after retrying.")
            retried_content.append(Document(page_content="", metadata={"source": url}))

    # Save retried results immediately
    with open(output_path, 'a') as f:
        for doc in retried_content:
            json.dump({"page_content": doc.page_content, "metadata": doc.metadata}, f, ensure_ascii=False)
            f.write("\n")

    return retried_content

def main(input_path, output_path):
    with open(input_path, 'r') as f:
        data = json.load(f)

    query = []
    save_result = dict()
    failed_urls = []

    for i in data:
        if i['urls']:
            query.append([i['claim'], i['urls']])

    print(len(query))
    for i in query:
        print(f"Processing claim: {i[0]}")
        result, failed = get_fact_check_content(i[1])
        save_result[i[0]] = result  # Store LangChain Document objects in the result
        failed_urls.extend(failed)

    # Save intermediate results
    serialized_result = {
        claim: [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        for claim, docs in save_result.items()
    }
    with open(output_path, 'w') as f:
        json.dump(serialized_result, f, indent=4, ensure_ascii=False)

    if failed_urls:
        retry_failed_urls(failed_urls, output_path)

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