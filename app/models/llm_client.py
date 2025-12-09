# app/models/llm_client.py

import requests
from typing import Dict, Any

from app.observability.langfuse_client import langfuse  # noqa: F401

from langfuse import observe

OLLAMA_URL = "http://127.0.0.1:11434/api/generate"
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
        response = requests.post(
            OLLAMA_URL,
            json=payload,
            timeout=180,
            proxies={"http": None, "https": None},
            headers={"Content-Type": "application/json"},
        )
    except requests.exceptions.Timeout as e:
        return {
            "response": "",
            "raw": None,
            "ok": False,
            "error": f"Timeout error: Модель загружается слишком долго. Попробуйте запустить модель заранее: ollama run {MODEL_NAME}",
        }
    except requests.exceptions.ConnectionError as e:
        return {
            "response": "",
            "raw": None,
            "ok": False,
            "error": f"Connection error: Не удалось подключиться к Ollama. Убедитесь, что Ollama запущен: ollama serve",
        }
    except Exception as e:
        return {
            "response": "",
            "raw": None,
            "ok": False,
            "error": f"Request error: {e}",
        }

    if response.status_code != 200:
        error_msg = f"LLM HTTP {response.status_code}"
        if response.status_code == 502:
            error_msg += " (Bad Gateway) - Ollama сервер не запущен или недоступен. Убедитесь, что Ollama запущен: ollama serve"
        elif response.text:
            error_msg += f": {response.text[:200]}"
        return {
            "response": "",
            "raw": None,
            "ok": False,
            "error": error_msg,
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
