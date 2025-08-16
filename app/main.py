from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query
from schemas import IndexRequest, QueryRequest, QueryResponse
from embedding import embed_texts
from elasticsearch_client import create_index, add_document, index_exists, search_similar, es
from llm_client import ask_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
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
    docs = search_similar(q_emb, req.top_k, req.index_name)
    context = "\n".join(docs)
    prompt = f"以下の情報を参考に質問に答えてください:\n{context}\n質問: {req.question}"
    answer = ask_llm(prompt)
    return QueryResponse(answer=answer, docs=docs)


@app.get("/get_index")
def get_index(index_name: str):
    try:
        count = es.count(index=index_name)["count"]
        # ドキュメント取得数の上限まで達した場合は10000を返す。
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
    


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)