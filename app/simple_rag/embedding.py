from sentence_transformers import SentenceTransformer

# Hugging Faceの埋め込み用公開モデルを指定
## all-MiniLM-L6-v2は384 次元の密ベクトルに変換するモデル
model = SentenceTransformer("all-MiniLM-L6-v2")

# Listでテキストを受け取って, 埋め込みを行ったListを返す
def embed_texts(texts: list)->list:
    return model.encode(texts).tolist()