from elasticsearch import Elasticsearch
from elasticsearch.exceptions import NotFoundError
from chunk import chunking

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

def get_documents(index_name: str, size: int = 10):
    """
    指定インデックスの最初の `size` 件のドキュメントを取得して表示
    """
    try:
        resp = es.search(
            index=index_name,
            query={"match_all": {}},  # 全件取得
            size=size
        )
        hits = resp["hits"]["hits"]
        documents = [hit["_source"] for hit in hits]  # _source にドキュメント内容が入る
        # for i, doc in enumerate(documents, 1):
            # print(f"{i}: {doc}")
        return documents
    except Exception as e:
        print(f"ドキュメント取得時にエラー: {e}")
        return []

def get_document_count(index_name):
    try:
        count_result = es.count(index=index_name)
        # print(count_result)
        # print('---------------')
        doc_count = count_result["count"]
        # print(f"インデックス '{index_name}' のドキュメント数: {doc_count}")
    except Exception as e:
        print(f"ドキュメント数取得時にエラー: {e}")

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

def add_document(index_name: str, id: str, content: str, embedding: list):
    if index_exists(index_name):
        doc = {"content": content, "embedding": embedding}
        try:
            es.index(index=index_name, id=id, document=doc)
            print(f"ドキュメント {id} をインデックス '{index_name}' に追加しました。")
        except Exception as e:
            print(f"ドキュメント追加時にエラーが発生しました: {e}")
            raise
    else:
        print(f"インデックス '{index_name}' は存在しません。")

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
        hits = resp["hits"]["hits"]

        # content と score の両方を返す
        results = [
            {
                "content": hit["_source"]["content"],
                "score": hit["_score"]  # 類似度スコア
            }
            for hit in hits
        ]
        return results

    except Exception as e:
        print(f"類似検索時にエラーが発生しました: {e}")
        raise

def search_bm25(query_text: str, top_k: int = 5, index_name: str = "test"):
    """
    BM25による全文検索
    :param query_text: 検索クエリ文字列
    :param top_k: 上位k件を返す
    :param index_name: 検索対象のインデックス
    :return: [{'content': ..., 'score': ...}, ...]
    """
    bm25_query = {
        "match": {
            "content": {
                "query": query_text
            }
        }
    }
    
    try:
        resp = es.search(index=index_name, query=bm25_query, size=top_k)
        hits = resp["hits"]["hits"]

        results = [
            {
                "content": hit["_source"]["content"],
                "score": hit["_score"]  # BM25スコア
            }
            for hit in hits
        ]
        return results
    except Exception as e:
        print(f"BM25検索時にエラーが発生しました: {e}")
        raise
    
def search_keyword(keyword: str, top_k: int = 5, index_name: str = "test"):
    """
    キーワード検索（完全一致に近い検索）
    :param keyword: 検索キーワード
    :param top_k: 上位k件を返す
    :param index_name: 検索対象のインデックス
    :return: [{'content': ..., 'score': ...}, ...]
    """
    keyword_query = {
        "term": {
            "content.keyword": {  # content フィールドの keyword サブフィールドを使用
                "value": keyword
            }
        }
    }

    try:
        resp = es.search(index=index_name, query=keyword_query, size=top_k)
        hits = resp["hits"]["hits"]

        results = [
            {
                "content": hit["_source"]["content"],
                "score": hit["_score"]  # キーワード検索スコア
            }
            for hit in hits
        ]
        return results
    except Exception as e:
        print(f"キーワード検索時にエラーが発生しました: {e}")
        raise