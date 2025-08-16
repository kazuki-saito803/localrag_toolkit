from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError

ES_COMPAT_VERSION = "8"  # Elasticsearch 8系なので8を指定
ES_URL = "http://localhost:9200"

# Elasticsearchクライアント初期化（互換モード対応）
es = Elasticsearch(
    ES_URL,
    headers={
        "Accept": f"application/vnd.elasticsearch+json; compatible-with={ES_COMPAT_VERSION}",
        "Content-Type": f"application/vnd.elasticsearch+json; compatible-with={ES_COMPAT_VERSION}"
    }
)


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


def index_exists(index_name: str) -> bool:
    try:
        es.indices.get(index=index_name)
        return True
    except NotFoundError:
        return False
    except Exception as e:
        print(f"インデックス存在チェック中に予期せぬエラーが発生しました: {e}")
        raise


# インデックス作成を index_name パラメータ対応に
def create_index(index_name: str):
    if not index_exists(index_name):
        try:
            es.indices.create(index=index_name, body=mapping)
            print(f"✅ インデックス '{index_name}' を作成しました。")
        except Exception as e:
            print(f"❌ インデックス作成時にエラーが発生しました: {e}")
            raise
    else:
        print(f"インデックス '{index_name}' は既に存在します。")

# ドキュメント追加を index_name パラメータ対応に
def add_document(index_name: str, id: str, content: str, embedding: list):
    doc = {"content": content, "embedding": embedding}
    try:
        es.index(index=index_name, id=id, document=doc)
        print(f"ドキュメント {id} をインデックス '{index_name}' に追加しました。")
    except Exception as e:
        print(f"ドキュメント追加時にエラーが発生しました: {e}")
        raise

# 入力クエリとベクトルDBに格納されたIndexの類似度を計算
def search_similar(embedding: list, top_k: int = 3, index_name: str = "test"):
    query = {
        "script_score": {
            "query": {"match_all": {}},
            "script": {
                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                "params": {"query_vector": embedding}
            }
        }
    }
    try:
        resp = es.search(index=index_name, query=query, size=top_k)
        print(f"Elasticsearch response: {resp}")
        hits = resp["hits"]["hits"]
        return [hit["_source"]["content"] for hit in hits]
    except Exception as e:
        print(f"類似検索時にエラーが発生しました: {e}")
        raise

if __name__ == '__main__':
    from embedding import embed_texts
    
    content = ["総理大臣の名前は齋藤一樹です。", "今日の天気は晴れです。", "お寿司は美味しいです。", "コーヒーを飲むとカフェインの効果で眠気が覚めます。", "夏は冷たいものが食べたくなります。"]
    # create_index("test")  # インデックスを作成
    documents_vector = embed_texts(content)
    for i, (doc, vector) in enumerate(zip(content, documents_vector)):
        add_document("test", str(i), doc, vector)
    query_vector = embed_texts(["今日の天気は？"])[0]
    similar_docs = search_similar(query_vector)
    print(similar_docs)