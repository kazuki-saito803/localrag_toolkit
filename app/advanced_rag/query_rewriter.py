from langchain_ollama import OllamaLLM
from langchain import PromptTemplate

# 1. LLMのセットアップ
llm = OllamaLLM(model="llama3", stop=["<|eot_id|>"])

template = """
<|begin_of_text|>
<|start_header_id|>system<|end_header_id|>
You are a query rewriting assistant. Rewrite the user query into a more precise search query.
Answer in Japanese.
<|eot_id|>
<|start_header_id|>user<|end_header_id|>
Original query: "{user_query}"
<|eot_id|>
<|start_header_id|>assistant<|end_header_id|>
Rewritten query:
"""

prompt = PromptTemplate(input_variables=["user_query"], template=template)

def rewrite_query(user_query: str) -> str:
    formatted = prompt.format(user_query=user_query)
    response = llm(formatted)
    return response.strip()

if __name__ == '__main__':    
    # 使用例
    rw = rewrite_query("東京の首都を教えてください")
    print("Rewritten:", rw)