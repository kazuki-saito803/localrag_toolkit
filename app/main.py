from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from app.schemas import IndexRequest, QueryRequest, QueryResponse
from app.embedding import embed_texts
from app.elasticsearch_client import create_index, add_document, index_exists, search_similar, es
from app.llm_client import ask_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # 起動時に特定の固定インデックスを作る場合はここで処理可能
    # ただし動的インデックス作成ならここは空でも良い
    yield


app = FastAPI(lifespan=lifespan)


@app.post("/index")
def index_docs(req: IndexRequest):
    # インデックス名必須チェック（念のため）
    if not req.index_name:
        raise HTTPException(status_code=400, detail="index_name は必須です")

    # インデックス存在チェック＆作成
    try:
        if not index_exists(req.index_name):
            create_index(req.index_name)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"インデックス作成に失敗しました: {e}")

    # 埋め込み計算＆ドキュメント登録
    embeddings = embed_texts(req.documents)
    for i, (doc, emb) in enumerate(zip(req.documents, embeddings)):
        add_document(index_name=req.index_name, id=f"doc_{i}", content=doc, embedding=emb)

    return {"message": f"インデックス '{req.index_name}' に {len(req.documents)} 件のドキュメントを登録しました"}


@app.post("/query", response_model=QueryResponse)
def query_answer(req: QueryRequest):
    q_emb = embed_texts([req.question])[0]
    docs = search_similar(q_emb)
    context = "\n".join(docs)
    prompt = f"以下の情報を参考に質問に答えてください:\n{context}\n質問: {req.question}"
    answer = ask_llm(prompt)
    return QueryResponse(answer=answer, docs=docs)


@app.get("/get_index")
def get_index(index_name: str):
    """
    指定したインデックス名のドキュメントを最大 size 件取得して返すエンドポイント
    """
    try:
        count = es.count(index=index_name)["count"]
        if count > 10000:
            count = 10000
        print(f"{index_name} のドキュメント件数は {count} 件です。")
        resp = es.search(
            index=index_name,
            query={"match_all": {}},
            size=count
        )
        hits = resp["hits"]["hits"]
        contents = [hit["_source"]["content"] for hit in hits]
        return {"index_name": index_name, "documents": contents}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"インデックス取得失敗: {e}")