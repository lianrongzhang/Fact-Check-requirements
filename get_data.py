import os
import argparse
import json
import requests
from bs4 import BeautifulSoup
from langchain.schema import Document
from fake_useragent import UserAgent
from concurrent.futures import ThreadPoolExecutor, as_completed

# Initialize UserAgent object
ua = UserAgent()
os.environ['USER_AGENT'] = ua.random

def fetch_url(url, timeout=10):
    """Fetch content from a URL and return a LangChain Document object."""
    try:
        headers = {'User-Agent': ua.random}
        response = requests.get(url, headers=headers, timeout=timeout)
        response.raise_for_status()  # Raise HTTPError for bad responses
        soup = BeautifulSoup(response.text, 'html.parser')
        plain_text = soup.get_text(strip=True)  # Extract clean text
        
        # Create a LangChain Document
        return Document(page_content=plain_text, metadata={"source": url})
    except requests.RequestException as e:
        print(f"Error fetching {url}: {e}")
        return None

def fetch_urls_with_retries(urls, max_retries=3, timeout=10):
    """Fetch multiple URLs with retry logic."""
    results = []
    for url in urls:
        for attempt in range(max_retries):
            print(f"Fetching {url} (Attempt {attempt + 1}/{max_retries})...")
            doc = fetch_url(url, timeout=timeout)
            if doc:
                results.append(doc)
                break
            else:
                print(f"Retry {attempt + 1} failed for {url}.")
        else:
            print(f"Failed to fetch {url} after {max_retries} retries.")
    return results

def process_claims_parallel(claims, max_retries=3, timeout=10, max_workers=5):
    """Process claims and their URLs in parallel while preserving input order."""
    results = [None] * len(claims)  # Placeholder list to maintain order

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_index = {
            executor.submit(fetch_urls_with_retries, claim['urls'], max_retries, timeout): idx
            for idx, claim in enumerate(claims)
        }
        for future in as_completed(future_to_index):
            idx = future_to_index[future]
            try:
                results[idx] = {
                    "claim": claims[idx]['claim'],
                    "docs": future.result()
                }
            except Exception as e:
                print(f"Error processing claim '{claims[idx]['claim']}': {e}")
                results[idx] = {
                    "claim": claims[idx]['claim'],
                    "docs": []
                }

    # Convert list of results to dictionary
    ordered_results = {item['claim']: item['docs'] for item in results}
    return ordered_results

def save_results_to_file(results, output_path):
    """Save LangChain documents as JSON with metadata."""
    serialized_results = {
        claim: [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        for claim, docs in results.items()
    }
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(serialized_results, f, indent=4, ensure_ascii=False)
    print(f"Results saved to {output_path}")

def main(input_path, output_path, max_retries=3, timeout=10, max_workers=5):
    """Main entry point to process claims and fetch URLs."""
    with open(input_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    claims = [{"claim": item['claim'], "urls": item['urls']} for item in data if 'urls' in item and item['urls']]
    print(f"Loaded {len(claims)} claims from {input_path}.")

    # Process claims in parallel
    results = process_claims_parallel(claims, max_retries=max_retries, timeout=timeout, max_workers=max_workers)

    # Save results to file
    save_results_to_file(results, output_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process fact-checking claims.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to save output JSON file.")
    parser.add_argument("--max_retries", type=int, default=3, help="Maximum number of retries for failed requests.")
    parser.add_argument("--timeout", type=int, default=10, help="Request timeout in seconds.")
    parser.add_argument("--max_workers", type=int, default=5, help="Maximum number of threads for parallel processing.")
    args = parser.parse_args()

    main(args.input_path, args.output_path, args.max_retries, args.timeout, args.max_workers)
