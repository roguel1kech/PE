import requests

from .data import get_items_df, get_interactions_df
from .content_based import ContentBasedRecommender
from .collaborative import ItemBasedCollaborativeRecommender
from .heuristics import popular_items, category_based_recent
from .metrics import precision_at_k, recall_at_k
from .llm_recommender import llm_rerank, PROMPT_VARIANT

from app.observability.langfuse_client import langfuse

OLLAMA_URL = "http://127.0.0.1:11434/api/tags"  # лёгкий запрос для проверки


def _ollama_available() -> bool:
    try:
        r = requests.get(OLLAMA_URL, timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def main():
    items = get_items_df()
    interactions = get_interactions_df()

    user_id = 101
    print(f"Демо для user_id={user_id}")
    print("История пользователя:")
    print(interactions[interactions["user_id"] == user_id])

    relevant = set(interactions[interactions["user_id"] == user_id]["item_id"].tolist())

    print("\n=== Контентная модель ===")
    cb = ContentBasedRecommender()
    rec_cb = cb.recommend_for_user(user_id, top_k=5)
    print("Рекомендации:", rec_cb)
    print("Precision@5:", precision_at_k(rec_cb, relevant, k=5))
    print("Recall@5:", recall_at_k(rec_cb, relevant, k=5))

    print("\n=== Коллаборативная фильтрация (item-based) ===")
    cf = ItemBasedCollaborativeRecommender()
    rec_cf = cf.recommend_for_user(user_id, top_k=5)
    print("Рекомендации:", rec_cf)
    print("Precision@5:", precision_at_k(rec_cf, relevant, k=5))
    print("Recall@5:", recall_at_k(rec_cf, relevant, k=5))

    print("\n=== Эвристика: популярные ===")
    rec_pop = popular_items(top_k=5)
    print("Рекомендации:", rec_pop)
    print("Precision@5:", precision_at_k(rec_pop, relevant, k=5))
    print("Recall@5:", recall_at_k(rec_pop, relevant, k=5))

    print("\n=== Эвристика: по категории последнего ===")
    rec_cat = category_based_recent(user_id, top_k=5)
    print("Рекомендации:", rec_cat)
    print("Precision@5:", precision_at_k(rec_cat, relevant, k=5))
    print("Recall@5:", recall_at_k(rec_cat, relevant, k=5))

    print("\n=== LLM-based rerank (кандидаты = популярные) ===")
    print(f"(вариант промпта: {PROMPT_VARIANT}; для варианта 2: LLM_PROMPT_VARIANT=2)")
    if not _ollama_available():
        print("Подсказка: Ollama не запущен. Реренк будет использовать порядок кандидатов без LLM.")
        print("  Запустите в отдельном терминале: ollama serve   затем: ollama run phi3:instruct")
    base_candidates = popular_items(top_k=5)
    rec_llm = llm_rerank(user_id, base_candidates, top_k=5)
    print("Кандидаты (популярные):", base_candidates)
    print("Реренжированные LLM:", rec_llm)
    print("Precision@5:", precision_at_k(rec_llm, relevant, k=5))
    print("Recall@5:", recall_at_k(rec_llm, relevant, k=5))

    try:
        langfuse.flush()
        if hasattr(langfuse, "shutdown"):
            langfuse.shutdown()  # ждём завершения отправки, иначе процесс выйдет до ответа сервера → connection error
        print("\n✓ Данные отправлены в Langfuse")
    except Exception as e:
        print(f"\n⚠ Ошибка при отправке данных в Langfuse: {e}")

if __name__ == "__main__":
    main()
