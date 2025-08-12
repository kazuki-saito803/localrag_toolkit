## はじめに
このリポジトリはAPIでRAGを構成するプロジェクトです。
## 準備
1. ボリュームを作成
    ```bash
    docker volume create elasticsearch_data
    ```
1. Elastic SearchをDockerコンテナで立ち上げる。
    以下のコマンドを実行。
    ```bash
    docker run -d --name elasticsearch-8.2.2 \
    -p 9200:9200 -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "ES_JAVA_OPTS=-Xms1g -Xmx1g" \
    -v elasticsearch_data:/usr/share/elasticsearch/data \
    docker.elastic.co/elasticsearch/elasticsearch:8.2.2
    ```
1. Fast APIサーバーを立ち上げる。
    ```bash
    uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
    ```
1. Ollamaサーバーを立ち上げる
    ```bash
    ollama serve --port 11434
    ```
1. APIリクエストを送る
    Index作成
    ```bash
    curl -X POST http://localhost:8000/index \
    -H "Content-Type: application/json" \
    -d '{
    "index_name": "my_index",
    "documents": [
        "日本の内閣総理大臣は齋藤一樹です。",
        "これは二つ目の文章です。",
        "これは三つ目の文章です。",
        "AI技術は日々進化しています。",
        "昨日の天気はとても良かったです。",
        "プログラミング学習は楽しいです。"
    ]
    }'
    ```
    LLM呼び出し
    ```bash
    curl -X POST http://localhost:8000/query \
    -H "Content-Type: application/json" \
    -d '{"question": "日本の総理大臣は誰ですか？"}'
    ```