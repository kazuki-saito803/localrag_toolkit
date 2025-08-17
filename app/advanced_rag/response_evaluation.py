from typing import List, Dict
from llm_client import ask_llm
from elasticsearch_client import search_similar
from embedding import embed_texts

def evaluate_answer(query: str, answer: str, docs: List[Dict]) -> float:
    """
    回答の精度を LLM に評価させ、0〜1 のスコアで返す
    """
    context = "\n".join([f"{i+1}. {d['content']}" for i, d in enumerate(docs)])
    prompt = f"""
あなたは質問応答評価者です。
ユーザーの質問に対する以下の回答を、提供された情報に基づいて正確性・有用性の観点から評価してください。
スコアは0から1で返してください。

質問: {query}
提供情報:
{context}
回答:
{answer}
"""
    score_str = ask_llm(prompt)
    try:
        return float(score_str.strip())
    except:
        return 0.0  # パースできなければ低評価

if __name__ == '__main__':
    query = ["総理大臣の名前は？"]
    emb_result = embed_texts(query)
    docs = search_similar(emb_result[0], 3, "test")
    context = "\n".join([d["content"]] for d in docs)  # マージ
    prompt = f"以下の情報を参考に質問に答えてください:\n{context}\n質問: {query[0]}"  # マージ
    answer = ask_llm(prompt) 
    # score = evaluate_answer(query[0], answer, docs)
    # print("評価スコア:", score)