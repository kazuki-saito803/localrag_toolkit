from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from embedding import embed_texts  # SentenceTransformerで埋め込み生成

ES_URL = "http://localhost:9200"
INDEX_NAME = "text_data_final"

# Elasticsearchクライアント初期化
es = Elasticsearch(ES_URL)

# インデックスマッピング
mapping = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ja_analyzer": {
                    "type": "custom",
                    "tokenizer": "kuromoji_tokenizer",
                    "filter": ["kuromoji_baseform", "kuromoji_part_of_speech", "cjk_width", "ja_stop"]
                }
            }
        }
    },
    "mappings": {
        "properties": {
            "content": {
                "type": "text",
                "analyzer": "ja_analyzer"
            },
            "embedding": {
                "type": "dense_vector",
                "dims": 384,
                "index": True,
                "similarity": "cosine"
            }
        }
    }
}

# インデックス作成
def create_index():
    if es.indices.exists(index=INDEX_NAME):
        print(f"インデックス '{INDEX_NAME}' は既に存在します。")
    else:
        es.indices.create(index=INDEX_NAME, body=mapping)
        print(f"✅ インデックス '{INDEX_NAME}' を作成しました。")

# ドキュメント追加
def add_document(doc_id, content, embedding):
    doc = {"content": content, "embedding": embedding}
    es.index(index=INDEX_NAME, id=doc_id, document=doc)
    print(f"ドキュメント {doc_id} を追加")

# 全文検索（BM25と同じ match クエリ）
def search_fulltext(query_text, top_k=5):
    query = {"match": {"content": {"query": query_text}}}
    resp = es.search(index=INDEX_NAME, query=query, size=top_k)
    return [{"content": hit["_source"]["content"], "score": hit["_score"]} for hit in resp["hits"]["hits"]]

# BM25検索（match クエリ）
def search_bm25(query_text, top_k=5):
    return search_fulltext(query_text, top_k)  # BM25も同じ match クエリ

# ベクトル検索（cosine similarity）
def search_vector(embedding, top_k=5):
    query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": embedding}
            }
        }
    }
    resp = es.search(index=INDEX_NAME, query=query, size=top_k)
    return [{"content": hit["_source"]["content"], "score": hit["_score"]} for hit in resp["hits"]["hits"]]

# --- 実行例 ---
if __name__ == "__main__":
    create_index()

    # サンプルデータ
    texts = [
        "総理大臣の名前は齋藤一樹です。",
        "今日の昼ごはんはカツ丼でした。",
        "好きな食べ物はお寿司です。",
        "趣味はサッカー観戦です。",
        "東京都に住んでいます。",
        "生まれは岩手県一関市です。"
    ]

    embeddings = embed_texts(texts)

    # ドキュメント追加
    for i, (text, emb) in enumerate(zip(texts, embeddings)):
        add_document(str(i), text, emb)

    # --- 全文検索 ---
    print("\n--- 全文検索 ---")
    for r in search_fulltext("齋藤一樹"):
        print(r["score"], r["content"])

    # --- BM25検索 ---
    print("\n--- BM25検索 ---")
    for r in search_bm25("昼ごはん"):
        print(r["score"], r["content"])

    # --- ベクトル検索 ---
    print("\n--- ベクトル検索 ---")
    query_text = ["今日の昼ごはんは？"]
    query_emb = embed_texts(query_text)[0]
    for r in search_vector(query_emb):
        print(r["score"], r["content"])