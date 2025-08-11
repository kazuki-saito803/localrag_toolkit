import subprocess
import json
from fastapi import HTTPException

def ask_llm(prompt: str) -> str:
    try:
        # 'ollama chat' コマンドをサブプロセスで呼び出し
        process = subprocess.Popen(
            ["ollama", "chat", "gpt-oss:20b", "--json"],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        stdout, stderr = process.communicate(input=prompt.encode("utf-8"))

        # エラーチェック
        if process.returncode != 0:
            raise HTTPException(status_code=500, detail=f"Ollama CLI error: {stderr.decode()}")

        # JSONレスポンスのパース
        resp = json.loads(stdout)

        # 応答メッセージの抽出
        return resp.get("message", {}).get("content", "")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM問い合わせ失敗: {str(e)}")