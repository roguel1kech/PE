from typing import List
import pandas as pd

from .data import get_items_df, get_interactions_df


def popular_items(top_k: int = 5) -> List[int]:
    """
    Базовый эвристический рекомендатель:
    рекомендует самые популярные товары (по числу взаимодействий).
    """
    interactions = get_interactions_df()
    pop = (
        interactions.groupby("item_id")["rating"]
        .count()
        .sort_values(ascending=False)
    )
    return pop.head(top_k).index.astype(int).tolist()


def category_based_recent(user_id: int, top_k: int = 5) -> List[int]:
    """
    Эвристика:
    - берём последнюю категорию, с которой взаимодействовал юзер
    - рекомендуем другие товары из этой категории
    """
    items = get_items_df()
    interactions = get_interactions_df()
    user_hist = interactions[interactions["user_id"] == user_id]
    if user_hist.empty:
        return popular_items(top_k=top_k)

    last_item_id = int(user_hist.iloc[-1].item_id)
    last_item = items[items["item_id"] == last_item_id].iloc[0]
    cat = last_item.category

    same_cat = items[
        (items["category"] == cat) &
        (items["item_id"] != last_item_id)
    ]

    if same_cat.empty:
        return popular_items(top_k=top_k)

    return same_cat["item_id"].head(top_k).astype(int).tolist()
