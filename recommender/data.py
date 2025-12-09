from typing import List, Dict
import pandas as pd


def get_items_df() -> pd.DataFrame:
    """
    Таблица товаров (items) с простыми текстовыми описаниями
    для контентной модели.
    """
    data = [
        {"item_id": 1, "title": "Энергетик LOOP Classic", "category": "energy", "description": "Классический энергетик без сахара"},
        {"item_id": 2, "title": "Энергетик LOOP Berry", "category": "energy", "description": "Энергетик с ягодным вкусом"},
        {"item_id": 3, "title": "Фруктовая пастила LOOP яблоко", "category": "snack", "description": "Натуральная пастила из яблок"},
        {"item_id": 4, "title": "Фруктовая пастила LOOP клубника", "category": "snack", "description": "Пастила клубника без сахара"},
        {"item_id": 5, "title": "Мерч: худи LOOP", "category": "merch", "description": "Чёрное худи LOOP, размер oversize"},
        {"item_id": 6, "title": "Мерч: шапка LOOP", "category": "merch", "description": "Тёплая шапка с логотипом LOOP"},
    ]
    return pd.DataFrame(data)


def get_interactions_df() -> pd.DataFrame:
    """
    История взаимодействий user-item (простая: просмотры/покупки).
    rating = 1 означает 'лайк/покупка/просмотр'.
    """
    data = [
        {"user_id": 101, "item_id": 1, "rating": 1},
        {"user_id": 101, "item_id": 2, "rating": 1},
        {"user_id": 101, "item_id": 5, "rating": 1},
        {"user_id": 102, "item_id": 3, "rating": 1},
        {"user_id": 102, "item_id": 4, "rating": 1},
        {"user_id": 103, "item_id": 1, "rating": 1},
        {"user_id": 103, "item_id": 3, "rating": 1},
        {"user_id": 104, "item_id": 2, "rating": 1},
        {"user_id": 104, "item_id": 6, "rating": 1},
        {"user_id": 105, "item_id": 5, "rating": 1},
        {"user_id": 105, "item_id": 6, "rating": 1},
    ]
    return pd.DataFrame(data)


def get_all_data() -> Dict[str, pd.DataFrame]:
    return {
        "items": get_items_df(),
        "interactions": get_interactions_df(),
    }
