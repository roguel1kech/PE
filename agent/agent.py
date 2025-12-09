from langchain.llms import Ollama

llm = Ollama(model="phi3:instruct")

def agent_process(text: str):
    prompt = f"""
Ты — интеллектуальный агент. 
Проанализируй текст и выполни 3 шага:

1. Определи тональность.
2. Сформулируй краткую суть текста.
3. Дай полезный ответ пользователю.

Текст: {text}
"""

    return llm(prompt)
