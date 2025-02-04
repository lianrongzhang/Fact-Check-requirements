import re
import json
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema import Document
from langchain.output_parsers import PydanticOutputParser
from pydantic import BaseModel
import timeit
import threading
import argparse
import gc


def analyze_fact_check(fact_check_content, model):
    if fact_check_content is None:
        return "There are no fact check content."

    info = []
    time_start = timeit.default_timer()

    for i in fact_check_content[1]:
        # 初始化 LLM
        llm = OllamaLLM(model=model)

        # 定義 PromptTemplate
        template = """
        Write a summary of the following:
        {context}
        """
        prompt = PromptTemplate.from_template(template)

        # 格式化 prompt
        formatted_prompt = prompt.format(context=i)

        # 執行 LLM
        result = llm.invoke(formatted_prompt)

        # 將結果與 URL 結合
        info.append(f"{result}, url: {i['metadata']}")

        del llm
        gc.collect()

    time_end = timeit.default_timer()

    # 將結果轉為 Document 格式
    documents = [Document(page_content=item) for item in info]

    return documents, time_end - time_start

class FactCheckOutput(BaseModel):
    label: str
    language: str
    date: str
    country: str
    url: list[str]
    reasoning: str
    
def fact_check(query, documents, model, analyzer_time=None):
    if not documents:
        return

    llm = OllamaLLM(model=model)

    # 在此我們對原有的 prompt 進行調整，增加明確的思考過程說明
    template = """
    You are a professional fact-checker tasked with evaluating the following claim.
    Let's break down the evidence and reasoning step by step.
    First, analyze the provided context {context} and identify key information relevant to the claim {claim}.
    Then, evaluate the evidence step by step to determine if the claim is true or false.
    Ensure your response includes the URL(s) of the source(s) you used for evaluation.
    
    Finally, you MUST STRICTLY structure the one and only one response in the following JSON format and don't add anything excessive:

    {{
        "label": "[Supported|Refuted|Not Enough Information]",
        "language": "[Language of the claim and context]",
        "date": "[YYYY-MM-DD]",
        "country": "[Country Code with 2 letter only]",
        "url": ["URL1", "URL2", ...],
        "reasoning": "[Your complete reasoning process here in double quote]"
    }}

    Remember, output only the LEGAL JSON string mentioned above.

    """
    prompt = PromptTemplate.from_template(template)
    formatted_prompt = prompt.format(context=documents, claim=query)

    # 使用 PydanticOutputParser
    parser = PydanticOutputParser(pydantic_object=FactCheckOutput)

    time_start = timeit.default_timer()
    result = llm.invoke(formatted_prompt)
    time_end = timeit.default_timer()
    del llm
    gc.collect()
    try:
        start_index = result.index('{')
        end_index = result.index('}', start_index)
        result_json = result[start_index:end_index + 1]
        
        parsed_result = parser.parse(result_json)
        if analyzer_time is None:
            return result, parsed_result.model_dump(), time_end - time_start
        else:
            return result, parsed_result.model_dump(), time_end - time_start + analyzer_time
    except Exception as e:
        return None, None, 0
    
def run_with_timeout(func, args=(), kwargs={}, timeout=60):
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

def test_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data





if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch and process fact-checking claims.")
    parser.add_argument("--input_path", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--model", type=str, required=True, help="Path to input JSON file.")
    parser.add_argument("--output_path", type=str, required=True, help="Path to input JSON file.")
    
    args = parser.parse_args()

    input_path = args.input_path
    model = args.model
    output_path = args.output_path
    content = test_data(input_path)

    keys = list(content.keys())
    values = list(content.values())
    save_result = []

    for key, value in zip(keys, values):
        print(f"Processing claim: {key}")
        print("-----------------------------------------------------------------")

        documents, analyzer_time = analyze_fact_check([key, value], model)

        fact_check_result, parsing_result, final_time = run_with_timeout(fact_check,args=(key, documents, model, analyzer_time))
        retry_count = 0

        # 重試機制
        while fact_check_result == None:
            retry_count += 1
            if retry_count > 20:
                print("Failed to get a valid result after 20 attempts.")
                fact_check_result = None
                final_time = None
                break

            fact_check_result, parsing_result, final_time = run_with_timeout(fact_check,args=(key, documents, model, analyzer_time))

        if fact_check_result == None:
            print(f"Skipping claim: {key}")
            continue

        tmp = {
            "claim": key,
            "label": parsing_result['label'],
            "reasoning": parsing_result['reasoning'],
            "date" : parsing_result['date'],
            "country" : parsing_result['country'],
            "urls" : parsing_result['url'],
            "time_taken": final_time
        }

        print("Result:")
        print(json.dumps(parsing_result, indent=4, ensure_ascii=False))
        print("-----------------------------------------------------------------")
        save_result.append(tmp)
    # 保存結果
    with open(output_path, 'w') as f:
        json.dump(save_result, f, indent=4, ensure_ascii=False)