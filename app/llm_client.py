import requests
from fastapi import HTTPException

def ask_llm(prompt: str) -> str:
    url = "http://localhost:11434/api/chat/completions"  # Ollamaの例
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "gpt-oss:20b",
        "messages": [{"role": "user", "content": prompt}]
    }
    resp = requests.post(url, json=data, headers=headers)
    if resp.status_code == 200:
        return resp.json()['choices'][0]['message']['content']
    else:
        raise HTTPException(status_code=500, detail="LLM問い合わせ失敗")