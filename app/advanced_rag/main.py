from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException, Query, FastAPI, UploadFile, File, Form
from schemas import IndexRequest, QueryRequest, QueryResponse
from embedding import embed_texts
from elasticsearch_client import create_index, add_document, index_exists, search_similar, es
from llm_client import ask_llm
from chunk import chunking
from typing import List
from fastapi.middleware.cors import CORSMiddleware

from reranking import reranking
from query_rewriter import rewrite_query
from response_evaluation import evaluate_answer


@asynccontextmanager
async def lifespan(app: FastAPI):
    yield

app = FastAPI(lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # ReactのURL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/index")
async def index_docs(
    index_name: str = Form(...),  # フォームから取得
    documents: List[UploadFile] = File(...)  # ファイルを複数受け取る
):
    all_chunks = []
    for uploaded_file in documents:
        content_bytes = await uploaded_file.read()
        text = content_bytes.decode("utf-8")
        chunks = chunking(text, chunk_size=50, overlap=10)
        all_chunks.extend(chunks)

    embeddings = embed_texts(all_chunks)
    for i, (chunk, emb) in enumerate(zip(all_chunks, embeddings)):
        add_document(index_name=index_name, id=f"chunk_{i}", content=chunk, embedding=emb)

    return {"message": f"インデックス '{index_name}' に {len(all_chunks)} 件のチャンクを登録しました"}

@app.post("/query", response_model=QueryResponse)
def query_answer(req: QueryRequest):
    q_emb = embed_texts([req.question])[0]  # 入力テキストをベクトル化
    docs = search_similar(q_emb, req.top_k, req.index_name)  # 類似度を計算して取得
    rerank_result = reranking(req.question, docs) # 計算結果を再評価
    context = "\n".join([d["content"] for d in rerank_result])  # マージ
    rewrited_query = rewrite_query(req.question)  # クエリを書き換え
    prompt = f"以下の情報を参考に質問に答えてください:\n{context}\n質問: {rewrited_query}"  # マージ
    answer = ask_llm(prompt)  # 回答生成
    return QueryResponse(answer=answer, docs=docs) # return


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