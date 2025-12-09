from typing import List
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

from .data import get_interactions_df, get_items_df


class ItemBasedCollaborativeRecommender:
    """
    Коллаборативная фильтрация на основе сходства товаров (item-based CF).

    Строим матрицу user-item (binary/ratings),
    считаем косинусное сходство между товарами,
    рекомендуем похожие на те, с которыми взаимодействовал пользователь.
    """

    def __init__(self):
        self.interactions = get_interactions_df()
        self.items = get_items_df()

        # user-item матрица
        user_item = self.interactions.pivot_table(
            index="user_id",
            columns="item_id",
            values="rating",
            fill_value=0,
        )
        self.user_ids = user_item.index.to_list()
        self.item_ids = user_item.columns.to_list()
        self.user_item_matrix = user_item.values  # shape: [n_users, n_items]

        self.item_sim_matrix = cosine_similarity(self.user_item_matrix.T)  # [n_items, n_items]

        self.item_id_to_index = {item_id: idx for idx, item_id in enumerate(self.item_ids)}
        self.index_to_item_id = {idx: item_id for item_id, idx in self.item_id_to_index.items()}

    def recommend_for_user(self, user_id: int, top_k: int = 5) -> List[int]:
        if user_id not in self.user_ids:
            return []

        u_idx = self.user_ids.index(user_id)
        user_vector = self.user_item_matrix[u_idx]  # [n_items]

        scores = self.item_sim_matrix.dot(user_vector)

        seen = set(self.interactions[self.interactions["user_id"] == user_id]["item_id"].tolist())
        ranked_indices = np.argsort(scores)[::-1]

        result = []
        for idx in ranked_indices:
            item_id = self.index_to_item_id[idx]
            if item_id in seen:
                continue
            if scores[idx] <= 0:
                continue
            result.append(int(item_id))
            if len(result) >= top_k:
                break

        return result
