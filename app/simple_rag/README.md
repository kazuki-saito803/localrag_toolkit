## simple_ragとは
このディレクトリではFast APIによってローカル環境でRAG機能を実装するPythonプログラムを提供。
 
## 各ファイルの説明
- chunk.py  
    入力に与えられたテキストをチャンク分割するための関数を定義するファイル
    - chunking関数  
        引数で与えられたtextをchunk_sizeおよびoverlapで指定された値を元にしてchunks(文字列を分割したリスト)を作成し返す関数  
        #### <引数>  
        - text:チャンク分割する文字列  
        - chunk_size:チャンク分割する文字数  
        - overlap:チャンク間で重複させる文字数  
        #### <戻り値>  
        - 文字列を分割してリスト化したオブジェクト
- elasticsearch_client.py  
    elasticsearchのコンテナへアクセスするための関数を定義するファイル  
    - get_documents関数  
        引数で与えられたIndex内にあるドキュメント情報をsizeで指定した数だけ取得する関数
        #### <引数>  
        - index_name:Elasticsearchに登録されておりDocument情報を取得する対象のIndex  
        - size:ドキュメントをどのくらい取得するか指定  
        #### <戻り値>  
        - Index内に格納されているドキュメント情報のリスト
    - get_document_count関数  
        対象Index内のドキュメント数を確認する関数  
        #### <引数>  
        - index_name:ドキュメント取得対象のIndex  
        #### <戻り値>  
        - ドキュメント数  
    - index_exists関数  
        引数で指定されたIndexがElastic Search上に存在するか確認する関数
        #### <引数>  
        - index_name:検索対象のIndex名  
        #### <戻り値>  
        - 存在する場合はTrue、存在しない場合はFalseの真偽値
    - create_index関数  
        Elasticsearchのコンテナ内に新しくIndexを作成する関数
        #### <引数>  
        - index_name:作成するIndex名  
        #### <戻り値>  
        - なし
    - add_document関数  
        #### <引数>  
        指定したIndexに対してdocumentを追加する関数
        - index_name:ドキュメントを追加するIndex名  
        - id:ドキュメントに付与する一意のID  
        - content:ベクトル化したドキュメントの元情報  
        - embedding:contentをベクトル化した情報  
        #### <戻り値>  
        - なし
    - search_similar関数  
        引数で与えられた埋め込み文字列の情報とIndexに格納された情報の類似度を計算して類似度の高い情報を返す関数  
        #### <引数>  
        - embedding:検索元の埋め込み文字情報  
        - top_k:類似度の高い順で取得するドキュメント数  
        - index_name:検索対象のIndex  
        #### <戻り値>  
        - 類似度の高い順に取得したドキュメント情報と類似度の辞書を要素としたリスト  
    - search_keyword関数  
        引数で受け取ったクエリとBD内の文章をキーワード検索する関数  
        #### <引数>  
        - keyword:クエリ  
        - top_k:類似度の高い順で何個のチャンクを取得するか  
        - index_name:検索対象のインデックス名  
        #### <戻り値>  
        - 検索結果と類似度スコア  
    - search_bm25関数
        引数で受け取ったクエリとBD内の文章をBM25に基づき類似度検索する関数
        #### <引数>  
        - query_text:クエリ  
        - top_k:類似度の高い順で何個のチャンクを取得するか  
        - index_name:検索対象のインデックス名  
        #### <戻り値>  
        - 検索結果と類似度スコア  
- embedding.py  
    Hugging Faceで提供されている埋め込みモデルを呼び出して文字列を多次元のベクトル情報に変換する処理を定義するファイル
    - embed_texts関数  
        #### <引数>  
        - texts:ベクトル化するテキスト情報
        #### <戻り値>  
        - 引数で受け取った文字列をそれぞれベクトル化したリスト  
- llm_client.py  
    ローカルPC上で立ち上げたollamaサーバーと通信するための関数を定義するファイル
    - ask_llm関数
        #### <引数>  
        - prompt:llmによる回答生成を行う際のプロンプト  
        #### <戻り値>  
        - llmからの回答  
- main.py
    Fast APIにより上記各関数を用いながらそれぞれのエンドポイント毎にRAGに必要なロジックを定義したファイル  
    - index_docs関数  
        引数で指定したIndexに対してDocumentを追加。Indexがない場合は作成  
        #### <引数>  
        - index_name:ドキュメント登録および作成するIndex名  
        - documents:Indexに追加するドキュメント情報  
        #### <戻り値>  
        - ドキュメント登録終了メッセージ
    - query_answer関数  
        #### <引数>  
        - question:ユーザーからの質問  
        - top_k:検索類似度の上位何ドキュメントを取得するか  
        - index_name:検索対象のIndex名  
        #### <戻り値>  
        - 回答と検索結果の情報を格納した辞書
    - get_index関数  
        引数で指定したIndex内に格納されているドキュメントを取得する関数  
        #### <引数>  
        - index_name:ドキュメント情報取得対象のIndex名  
        #### <戻り値>  
        - インデックス名とドキュメント情報を格納した辞書  
## API機能詳細
### インデックス登録機能  
- パス: /index  
- メソッド: POST  
- 処理の流れ:  
    1. ドキュメント数に応じて非同期的なファイルの読み込み, UTF-8によるデコード, チャンク分割の適用, チャンクの統合を実施
    1. embed_texts関数でチャンク化した文字列のベクトル化
    1. add_document関数によるIndexの作成
    1. インデックス作成成功のレスポンスを返す
### 回答生成機能  
- パス: /query  
- メソッド: POST  
- 処理の流れ:  
    1. embed_texts関数による入力文字列のベクトル化
    1. search_similar関数による類似度検索
    1. joinメソッドで取得した文字列のマージ
    1. 3で作成した文字列とユーザーからの質問を元にask_llm関数で回答の取得
    1. QueryRersponseの型に従ってレスポンスを返す
### インデックス確認機能  
- パス: /get_index  
- メソッド: GET  
- 処理の流れ:  
    1. countメソッドを元にドキュメント数を取得
    1. 取得ドキュメントの上限を設定
    1. searchメソッドを実行してIndex内のドキュメント情報を取得
    1. Index名とドキュメント情報を辞書化してレスポンスとして返す