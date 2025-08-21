import numpy as np
from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from sentence_transformers import SentenceTransformer

# Elasticsearchの設定
ES_URL = "http://localhost:9200"
ES_COMPAT_VERSION = "8"
es = Elasticsearch(
    ES_URL,
    headers={
        "Accept": f"application/vnd.elasticsearch+json; compatible-with={ES_COMPAT_VERSION}",
        "Content-Type": f"application/vnd.elasticsearch+json; compatible-with={ES_COMPAT_VERSION}"
    }
)

# 埋め込みモデルの設定
model = SentenceTransformer("all-MiniLM-L6-v2")

# インデックス名
INDEX_NAME = "text_data04"

# インデックスのマッピング設定
mapping = {
    "mappings": {
        "properties": {
            "content": {"type": "text"},
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

# インデックスの存在確認
def index_exists(index_name: str) -> bool:
    try:
        es.indices.get(index=index_name)
        return True
    except NotFoundError:
        return False

# インデックスの作成
def create_index(index_name: str):
    if not index_exists(index_name):
        es.indices.create(index=index_name, body=mapping)
        print(f"✅ インデックス '{index_name}' を作成しました。")
    else:
        print(f"インデックス '{index_name}' は既に存在します。")

# ドキュメントの追加
def add_document(index_name: str, doc_id: str, content: str, embedding: list):
    doc = {"content": content, "embedding": embedding}
    es.index(index=index_name, id=doc_id, document=doc)
    print(f"ドキュメント {doc_id} をインデックス '{index_name}' に追加しました。")

# ベクトルの埋め込み生成
def embed_texts(texts: list) -> list:
    return model.encode(texts).tolist()

# ベクトル検索（cosine similarity）
def search_vector(query_vector: list, top_k: int = 3):
    query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": query_vector}
            }
        }
    }
    response = es.search(index=INDEX_NAME, query=query, size=top_k)
    return [{"content": hit["_source"]["content"], "score": hit["_score"]} for hit in response["hits"]["hits"]]

# BM25検索
def search_bm25(query_text: str, top_k: int = 3):
    query = {"match": {"content": query_text}}
    response = es.search(index=INDEX_NAME, query=query, size=top_k)
    return [{"content": hit["_source"]["content"], "score": hit["_score"]} for hit in response["hits"]["hits"]]

# キーワード検索
def search_keyword(keyword: str, top_k: int = 3):
    query = {
        "term": {
            "content.keyword": {
                "value": keyword
            }
        }
    }
    response = es.search(index=INDEX_NAME, query=query, size=top_k)
    return [{"content": hit["_source"]["content"], "score": hit["_score"]} for hit in response["hits"]["hits"]]

# ドキュメントの取得
def get_documents(index_name: str, size: int = 10):
    response = es.search(index=index_name, query={"match_all": {}}, size=size)
    return [{"content": hit["_source"]["content"]} for hit in response["hits"]["hits"]]

# メイン処理
if __name__ == "__main__":
    create_index(INDEX_NAME)

    # サンプルデータ
    texts = [
        "Elasticsearchは全文検索エンジンです。",
        "ベクトル検索は意味的な類似性を評価します。",
        "BM25は従来の検索手法です。",
        "キーワード検索は完全一致を重視します。",
        "Elasticsearchはスケーラブルな検索を提供します。",
        "ベクトル検索はAIによる意味理解を活用します。"
    ]

    # 埋め込みの生成とドキュメントの追加
    embeddings = embed_texts(texts)
    for i, (text, embedding) in enumerate(zip(texts, embeddings)):
        add_document(INDEX_NAME, str(i), text, embedding)

    # 検索例
    query = "ベクトル検索のメリットは何ですか？"
    query_embedding = embed_texts([query])[0]

    print("\n--- ベクトル検索 ---")
    vector_results = search_vector(query_embedding)
    for result in vector_results:
        print(f"スコア: {result['score']}, 内容: {result['content']}")

    print("\n--- BM25検索 ---")
    bm25_results = search_bm25(query)
    for result in bm25_results:
        print(f"スコア: {result['score']}, 内容: {result['content']}")

    print("\n--- キーワード検索 ---")
    keyword_results = search_keyword("ベクトル検索")
    for result in keyword_results:
        print(f"スコア: {result['score']}, 内容: {result['content']}")