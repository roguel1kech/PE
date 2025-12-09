from typing import List
import re

from recommender.data import get_items_df, get_interactions_df
from app.models.llm_client import generate_llm


def build_llm_prompt(user_id: int, candidate_item_ids: List[int]) -> str:
    """
    Формирует промпт для LLM:
    - краткая история пользователя
    - описание кандидатов
    - просьба вернуть список item_id в порядке релевантности
    """
    items = get_items_df()
    interactions = get_interactions_df()

    user_hist = interactions[interactions["user_id"] == user_id]
    hist_str = ""
    if user_hist.empty:
        hist_str = "У пользователя ещё нет истории покупок."
    else:
        hist_items = items[items["item_id"].isin(user_hist["item_id"])]
        parts = []
        for _, row in hist_items.iterrows():
            parts.append(f"- [{row.item_id}] {row.title} ({row.category}): {row.description}")
        hist_str = "Пользователь ранее взаимодействовал с:\n" + "\n".join(parts)

    cand_items = items[items["item_id"].isin(candidate_item_ids)]
    cand_lines = []
    for _, row in cand_items.iterrows():
        cand_lines.append(f"- [{row.item_id}] {row.title} ({row.category}): {row.description}")

    cand_str = "\n".join(cand_lines)

    prompt = f"""
Ты — рекомендательная система для магазина LOOP.

Тебе передана история пользователя и список кандидатов-товаров.
Твоя задача — упорядочить кандидатов по степени релевантности пользователю.

История пользователя (user_id={user_id}):
{hist_str}

Список кандидатов:
{cand_str}

Верни только список идентификаторов товаров (item_id) в порядке убывания релевантности.
Формат ответа: перечисли числа через запятую, без других комментариев.
Например: 5, 2, 1
"""
    return prompt


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
