# app/models/llm_client.py

import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "phi3:instruct"  # убедись, что такая модель есть в `ollama list`


def generate_llm(prompt: str):
    """
    Вызывает локальную LLM через Ollama.
    Делает запрос к /api/generate и возвращает словарь.

    Если Ollama отвечает ошибкой (502/500/404/и т.п.),
    мы НЕ бросаем исключение, а возвращаем JSON с описанием ошибки.
    Это важно для стабильной работы API и тестов.
    """
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=120)
    except Exception as e:
        # Ошибка сети / соединения — возвращаем описательный ответ
        return {
            "response": "",
            "error": f"Request error: {e}",
            "ok": False,
        }

    # Не используем response.raise_for_status(), чтобы не падать по HTTPError
    if response.status_code != 200:
        return {
            "response": "",
            "error": f"LLM HTTP {response.status_code}: {response.text}",
            "ok": False,
        }

    # Нормальный кейс: 200 OK
    try:
        data = response.json()
    except Exception as e:
        return {
            "response": "",
            "error": f"JSON parse error: {e}",
            "ok": False,
        }

    # У Ollama обычно есть поле "response" с текстом ответа
    return {
        "response": data.get("response", ""),
        "raw": data,
        "ok": True,
    }
