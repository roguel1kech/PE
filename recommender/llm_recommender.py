import os
import re
from typing import List

from recommender.data import get_items_df, get_interactions_df
from app.models.llm_client import generate_llm

PROMPT_VARIANT = int(os.getenv("LLM_PROMPT_VARIANT", "1"))


def _get_history_and_candidates(user_id: int, candidate_item_ids: List[int]):
    """Общие данные для обоих вариантов промпта."""
    items = get_items_df()
    interactions = get_interactions_df()

    user_hist = interactions[interactions["user_id"] == user_id]
    if user_hist.empty:
        hist_str = "У пользователя ещё нет истории покупок."
    else:
        hist_items = items[items["item_id"].isin(user_hist["item_id"])]
        parts = []
        for _, row in hist_items.iterrows():
            parts.append(f"  • id={row.item_id} | {row.title} | {row.category} | {row.description}")
        hist_str = "\n".join(parts)

    cand_items = items[items["item_id"].isin(candidate_item_ids)]
    cand_lines = []
    for _, row in cand_items.iterrows():
        cand_lines.append(f"  • id={row.item_id} | {row.title} | {row.category} | {row.description}")
    cand_str = "\n".join(cand_lines)

    return hist_str, cand_str


def _build_prompt_v1(user_id: int, hist_str: str, cand_str: str) -> str:
    """Вариант 1: блочная структура с явными заголовками и примером."""
    return f"""# Роль
Ты — модуль переранжирования. 
На вход даны история взаимодействий пользователя и список кандидатов. 
Выведи id кандидатов в порядке убывания релевантности.

# История (user_id={user_id})
{hist_str}

# Кандидаты
Расставь по релевантности (сверху — самый подходящий):
{cand_str}

# Формат ответа
Одна строка: id через запятую, без текста. Пример: 5, 2, 1
"""


def _build_prompt_v2(user_id: int, hist_str: str, cand_str: str) -> str:
    """Вариант 2: пошаговая инструкция и явный шаблон ответа."""
    return f"""Переранжируй кандидаты по релевантности для пользователя.

История (user_id={user_id}):
{hist_str}

Кандидаты:
{cand_str}

Требование: выведи одну строку — id через запятую, от самого релевантного к менее релевантному. Без пояснений.
Пример формата: 5, 2, 1, 3, 6

Ответ:
"""


def build_llm_prompt(user_id: int, candidate_item_ids: List[int], variant: int = None) -> str:
    """
    Формирует промпт для LLM. Два наглядных варианта:
    - 1: блочная структура (# Роль, # История, # Кандидаты, # Формат, # Пример)
    - 2: пошаговая инструкция (Шаг 1–3) и явная строка «Твой ответ:»
    По умолчанию используется PROMPT_VARIANT (или LLM_PROMPT_VARIANT из env).
    """
    variant = variant if variant is not None else PROMPT_VARIANT
    hist_str, cand_str = _get_history_and_candidates(user_id, candidate_item_ids)

    if variant == 2:
        return _build_prompt_v2(user_id, hist_str, cand_str)
    return _build_prompt_v1(user_id, hist_str, cand_str)


def parse_item_ids_from_llm_response(text: str) -> List[int]:
    """
    Пытаемся вытащить числа из ответа LLM.
    Если LLM начала болтать — вычленим все числа и интерпретируем их как item_id.
    """
    nums = re.findall(r"\d+", text)
    return [int(n) for n in nums]


def llm_rerank(user_id: int, candidate_item_ids: List[int], top_k: int = 5) -> List[int]:
    """
    LLM-based рекоммендер: берёт кандидатов и просит LLM отранжировать их.

    По сути: two-stage pipeline:
      1) генерация candidate set (например, популярные)
      2) переранжирование LLM на основе текстовых описаний.
    """
    if not candidate_item_ids:
        return []

    prompt = build_llm_prompt(user_id, candidate_item_ids)
    data = generate_llm(prompt)

    # generate_llm уже возвращает dict {"response": str, "ok": bool, ...}
    resp_text = ""
    if isinstance(data, dict):
        resp_text = data.get("response", "") or data.get("error", "")
    else:
        resp_text = str(data)

    ids = parse_item_ids_from_llm_response(resp_text)
    # фильтруем только тех, кто в исходных кандидатах
    ids = [i for i in ids if i in candidate_item_ids]

    # Если LLM вернула что-то странное, fallback к исходным
    if not ids:
        return candidate_item_ids[:top_k]

    # ограничим top_k
    return ids[:top_k]
