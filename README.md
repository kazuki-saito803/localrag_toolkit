# localrag_toolkit

このリポジトリは、ローカル環境で動かす RAG API のサンプル実装です。  
FastAPI を入口にして、Elasticsearch を検索基盤、SentenceTransformers を埋め込み生成、Ollama を回答生成に使います。

## 構成

- `app/simple_rag`
  基本的な RAG パイプラインです。
- `app/advanced_rag`
  基本構成に加えて、再ランキング、クエリ書き換えなどを試している拡張版です。
- `tests`
  `pytest` で実行できる自動テストです。

## RAG の流れ

### 1. インデックス登録

`/index` にファイルをアップロードすると、次の流れで処理します。

1. アップロードファイルを UTF-8 で読み込む
2. テキストをチャンク分割する
3. 各チャンクを埋め込みベクトルに変換する
4. Elasticsearch のインデックスを作成し、チャンクと埋め込みを保存する

### 2. 質問応答

`/query` に質問を送ると、次の流れで処理します。

1. 質問文を埋め込みベクトルに変換する
2. Elasticsearch で類似チャンクを検索する
3. 検索結果を文脈としてまとめる
4. Ollama に文脈付きプロンプトを送り、回答を生成する

`advanced_rag` ではこの途中に以下を追加しています。

- 検索結果の再ランキング
- 質問文の書き換え

## セットアップ

### Elasticsearch を起動

永続化したい場合は先に volume を作成します。

```bash
docker volume create elasticsearch_data
```

次に Elasticsearch を Docker で起動します。

```bash
docker run -d --name elasticsearch-8.2.2 \
  -p 9200:9200 -p 9300:9300 \
  -e "discovery.type=single-node" \
  -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
  -v elasticsearch_data:/usr/share/elasticsearch/data \
  docker.elastic.co/elasticsearch/elasticsearch:8.2.2
```

### Ollama を起動

```bash
ollama serve --port 11434
```

使用するモデルを取得します。

```bash
ollama pull llama3
```

### API サーバーを起動

`simple_rag` を起動する場合:

```bash
cd app/simple_rag
python main.py
```

`advanced_rag` を起動する場合:

```bash
cd app/advanced_rag
python main.py
```

## API

### `POST /index`

複数ファイルをアップロードしてインデックスを作成します。  
このエンドポイントは `multipart/form-data` を受け取ります。

例:

```bash
curl -X POST http://127.0.0.1:8000/index \
  -F "index_name=my_index" \
  -F "documents=@sample1.txt" \
  -F "documents=@sample2.txt"
```

### `POST /query`

質問を送り、検索結果を使って回答を生成します。

```bash
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "総理大臣の名前は誰ですか？",
    "top_k": 3,
    "index_name": "my_index"
  }'
```

### `GET /get_index`

指定インデックスに保存されているチャンク内容を確認します。

```bash
curl "http://127.0.0.1:8000/get_index?index_name=my_index"
```

## ファイル説明

### 共通の主要ファイル

- `main.py`
  FastAPI のエンドポイント定義です。
- `chunk.py`
  テキストをチャンク分割します。
- `embedding.py`
  `all-MiniLM-L6-v2` を使って埋め込みを生成します。
- `elasticsearch_client.py`
  インデックス作成、ドキュメント追加、類似検索を行います。
- `llm_client.py`
  Ollama API を呼び出して回答を生成します。
- `schemas.py`
  リクエストとレスポンスの Pydantic モデルです。

### `advanced_rag` の追加ファイル

- `reranking.py`
  LLM を使って検索結果の順位を並び替えます。
- `query_rewriter.py`
  ユーザーの質問を検索向けに書き換えます。
- `response_evaluation.py`
  回答の妥当性を評価する実験用モジュールです。

## テスト

`pytest` でテストを実行できます。

```bash
./venv/bin/pytest -q
```

## 補足

- Elasticsearch は `http://localhost:9200`
- Ollama は `http://localhost:11434`
- 埋め込み次元は `384`
- CORS は `http://localhost:3000` を許可しています
