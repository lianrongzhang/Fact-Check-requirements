import re
import json
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain.schema import Document
import timeit


def analyze_fact_check(fact_check_content, model):
    if fact_check_content == None:
        return("There are no fact check content.")
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
        Ensure your response includes the URL(s) of the source(s) you used for evaluation.
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
        .
        .
        . (if needed)

        
        Additional Analysis:  
        - [Integrate multiple pieces of evidence or consider other contextual factors for deeper reasoning.]

        ### Conclusion:

        - Claim Status: [State "Supported," "Refuted," or "Not Enough Information"]
        - Language: [Specify the language of the claim and context, do not translate.]
        - Date: [Specify the date of the claim or context, e.g., "YYYY-MM-DD."]
        - Country: [Reason the country relevant to the claim and transform it to the country code (e.g., US, UK, CA), only show the country code.]
        - URL: [Provide the URL(s) of the source(s) for reference.]
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
        return False, None
    
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

patterns = {
    "Claim Status": [r"Claim Status:\s*([^\n]+)", r"\*\*Claim Status\*\*:\s*([^\n]+)"],
    "Language": [r"Language:\s*([^\(\n]+)", r"\*\*Language\*\*:\s*([^\(\n]+)"],
    "Date": [r"Date:\s*([^\(\n]+)", r"\*\*Date\*\*:\s*([^\(\n]+)"],
    "Country": [r"Country:\s*([A-Z]+)", r"\*\*Country\*\*:\s*([A-Z]+)"],
    "URL": [r"URL:\s*(https?://[^\s]+)", r"https?://[^\s]+"]
}
    
def test_data(path):
    with open(path, 'r') as f:
        data = json.load(f)
    return data

path1 = '/home/user/talen-python/Climate_Fever/Climate_Fever_content.json'

content = test_data(path1)

keys = list(content.keys())
values = list(content.values())
model = "mistral"
save_result = []
for key, value in zip(keys, values):
    print(f"Processing claim: {key}")
    print("-----------------------------------------------------------------")
    documents, analyzer_time = analyze_fact_check([key, value], model)
    
    fact_check_result, final_time = fact_check(key, documents, model, analyzer_time)
    retry_count = 0

    # 重试机制
    while fact_check_result == False:
        retry_count += 1
        if retry_count > 3:
            print("Failed to get a URL after 3 attempts.")
            fact_check_result = False
            final_time = None
            break
        fact_check_result, final_time = fact_check(key, documents, model, analyzer_time)

    if fact_check_result == False:
        print(f"Skipping claim: {key}")
        continue
    
    extract_data_result = extract_data(fact_check_result, patterns)
    tmp = {
        "claim": key,
        "result": fact_check_result,
        "claim_status": extract_data_result["Claim Status"],
        "language": extract_data_result["Language"],
        "date": extract_data_result["Date"],
        "country": extract_data_result["Country"],
        "url": extract_data_result["URL"],
        "time_taken": final_time
    }
    print("Result:")
    print(tmp['result'])
    print("-----------------------------------------------------------------")
    save_result.append(tmp)

# # 保存结果
with open('/home/user/talen-python/Climate_Fever/Climate_Fever_result3_mistral.json', 'w') as f:
    json.dump(save_result, f, indent=4, ensure_ascii=False)