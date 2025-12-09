# Sentiment Analysis API + Local LLM Agent

Проект выполнен в рамках **Серии заданий 2**.  
Цель — реализовать полноценный ML-сервис на основе готовых моделей и создать API для взаимодействия.

## 📌 Функциональность

- Анализ тональности текста (`/predict`)
- API для локальной LLM (Ollama) (`/generate`)
- Интеллектуальный агент на базе LLM
- Автоматические тесты pytest
- Полностью изолированная архитектура (FastAPI)

---

## 🧠 Используемые технологии

- Python 3.11  
- FastAPI  
- Uvicorn  
- HuggingFace Transformers  
- Ollama (локальная LLM `phi3:instruct`)  
- PyTest  

---

## 📁 Структура проекта

```
app/
  main.py
  llm_api.py
  models/
    sentiment_model.py
    llm_client.py
agent/
  agent.py
tests/
  test_sentiment_api.py
  test_llm_api.py
  test_agent.py
requirements.txt
README.md
```

---

## 🚀 Запуск проекта

### 1. Создать виртуальное окружение

```bash
python -m venv .venv
.venv\Scripts\activate
```

### 2. Установить зависимости

```bash
pip install -r requirements.txt
```

### 3. Запустить ML API

```bash
uvicorn app.main:app --reload
```

API будет доступно по адресу:

```
http://127.0.0.1:8000/docs
```

### 4. Запустить LLM API

```bash
uvicorn app.llm_api:app --reload --port 8001
```

---

## 🤖 Запуск LLM (Ollama)

Убедитесь, что Ollama установлена:  
https://ollama.com/

Проверьте модель:

```bash
ollama pull phi3:instruct
```

---

## 🧪 Тестирование

Запуск всех тестов:

```bash
pytest -vv
```

Все тесты должны быть зелёными.

---

## 📌 Автор

Проект выполнен как часть задания «Серия 2» для изучения применимости готовых ML-моделей.
