# agent/agent.py

from langfuse import observe
from app.models.llm_client import generate_llm


@observe(name="agent_process")
def agent_process(text: str) -> str:
    """
    Простой ИИ-агент на основе локальной LLM через Ollama.

    Делает 3 шага:
    1. Определяет тональность текста.
    2. Делает краткое резюме.
    3. Формирует ответ пользователю от лица компании.
    """
    prompt = f"""
Ты — интеллектуальный ассистент по работе с пользователями.

Проанализируй следующий текст и выполни 3 шага:

1) Кратко определи тональность (позитивная / нейтральная / негативная).
2) Дай краткое резюме (1–2 предложения).
3) Сформулируй вежливый и полезный ответ пользователю от лица компании.

Отвечай структурированно и по делу.

Текст:
\"\"\"{text}\"\"\"
"""

    data = generate_llm(prompt)

    if isinstance(data, dict):
        if data.get("response"):
            return data["response"]
        if data.get("error"):
            return f"LLM error in agent: {data['error']}"
        return str(data)

    return str(data)
