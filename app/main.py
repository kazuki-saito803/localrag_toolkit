from contextlib import asynccontextmanager
from fastapi import FastAPI
from app.schemas import IndexRequest, QueryRequest, QueryResponse
from app.embedding import embed_texts
from app.elasticsearch_client import create_index, add_document, search_similar
from app.llm_client import ask_llm


@asynccontextmanager
async def lifespan(app: FastAPI):
    # アプリ起動時に実行（startup の代わり）
    create_index()
    yield
    # アプリ終了時に実行（必要なクリーンアップ処理があればここに）


app = FastAPI(lifespan=lifespan)


@app.post("/index")
def index_docs(req: IndexRequest):
    embeddings = embed_texts(req.documents)
    for i, (doc, emb) in enumerate(zip(req.documents, embeddings)):
        add_document(id=f"doc_{i}", content=doc, embedding=emb)
    return {"message": f"{len(req.documents)}件のドキュメントを登録しました"}


@app.post("/query", response_model=QueryResponse)
def query_answer(req: QueryRequest):
    q_emb = embed_texts([req.question])[0]
    docs = search_similar(q_emb)
    context = "\n".join(docs)
    prompt = f"以下の情報を参考に質問に答えてください:\n{context}\n質問: {req.question}"
    answer = ask_llm(prompt)
    return QueryResponse(answer=answer)