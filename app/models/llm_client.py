# app/models/llm_client.py

import requests
from typing import Dict, Any

from langfuse import observe

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:instruct"


@observe(name="ollama_generate", as_type="generation")
def generate_llm(prompt: str) -> Dict[str, Any]:
    """
    Вызывает локальную LLM через Ollama.
    Декорирован @observe, поэтому:
    - Langfuse сам логирует вход (prompt),
    - выход (response),
    - время выполнения,
    - ошибку (если будет).
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except Exception as e:
        return {
            "response": "",
            "raw": None,
            "ok": False,
            "error": f"Request error: {e}",
        }

    if response.status_code != 200:
        return {
            "response": "",
            "raw": None,
            "ok": False,
            "error": f"LLM HTTP {response.status_code}: {response.text}",
        }

    try:
        data = response.json()
    except Exception as e:
        return {
            "response": "",
            "raw": None,
            "ok": False,
            "error": f"JSON parse error: {e}",
        }

    text = data.get("response", "")

    return {
        "response": text,
        "raw": data,
        "ok": True,
        "error": None,
    }
