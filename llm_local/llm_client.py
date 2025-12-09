import requests
from typing import Dict

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:instruct"  # или любая другая установленная модель


def generate(prompt: str) -> Dict:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    return {
        "response": data.get("response", ""),
        "raw": data
    }


def main():
    print("Локальная LLM (Ollama). Пиши prompt, пустая строка — выход.")
    while True:
        prompt = input("> ")
        if not prompt.strip():
            break
        result = generate(prompt)
        print(result["response"])
        print("-" * 40)

if __name__ == "__main__":
    main()
