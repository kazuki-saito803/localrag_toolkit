from langchain_ollama import OllamaLLM
import json
from typing import List

# OllamaLLMの初期化
llm = OllamaLLM(model="llama3", stop=["<|eot_id|>"])

def reranking(query: str, candidates: List[dict], llm) -> List[dict]:
    """
    LLMを用いて候補のリストを再評価し、スコア順に並び替える
    """
    if not candidates:
        print("候補が空です")
        return []

    # 候補を文字列化
    candidate_texts = "\n".join([f"{i+1}. {c['content']}" for i, c in enumerate(candidates)])

    # プロンプト生成
    prompt = f"""
以下の検索結果候補を、ユーザーの検索クエリにどれだけ関連しているかを評価し、最も関連性の高い順に並び替えてください。

検索クエリ:
{query}

候補:
{candidate_texts}

出力はJSON形式で "index" と "score" をつけてください。
"""

    # LLMに問い合わせ
    response = llm.invoke(prompt)

    # デバッグ用: 生応答を表示
    print("LLM生応答:", response)

    if not response or response.strip() == "":
        print("LLMから応答がありません。元の候補を返します。")
        return candidates

    # JSONパースを例外処理
    try:
        reranked_indices = json.loads(response)
    except json.JSONDecodeError:
        print("LLM応答がJSON形式ではありません。元の候補を返します。")
        return candidates

    # LLMの評価順に候補を並べ替え
    reranked = []
    for item in reranked_indices:
        idx = item.get("index")
        score = item.get("score")
        if idx is not None and 0 <= idx < len(candidates):
            candidates[idx]["score"] = score
            reranked.append(candidates[idx])

    return reranked


if __name__ == '__main__':
    from elasticsearch_client import search_similar
    from embedding import embed_texts

    input_data = ["総理大臣の名前は？"]
    embed_data = embed_texts(input_data)
    
    # Elasticsearchで類似検索
    result = search_similar(embed_data[0], top_k=5, index_name="test")  # index_nameやtop_kは適宜変更
    print("検索結果:", result)

    # LLMで再評価
    reranked_result = reranking(input_data[0], result, llm)
    print("再評価後の結果:", reranked_result)