import requests
from fastapi import HTTPException

def ask_llm(prompt: str) -> str:
    try:
        url = "http://localhost:11434/api/generate"  # OllamaのAPIエンドポイント
        payload = {
            "model": "llama3",
            "prompt": prompt,
            "stream": False 
        }

        resp = requests.post(url, json=payload)

        # HTTPエラーチェック
        if resp.status_code != 200:
            raise HTTPException(status_code=500, detail=f"Ollama API error: {resp.text}")

        # JSONパース
        data = resp.json()

        # 応答テキスト取得
        return data.get("response", "")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"LLM問い合わせ失敗: {str(e)}")