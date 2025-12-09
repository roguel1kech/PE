from typing import List
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from .data import get_items_df, get_interactions_df


class ContentBasedRecommender:
    """
    Контентно-ориентированная рекомендация:
    на основе текстовых описаний товаров строим TF-IDF,
    вычисляем косинусное сходство между товарами.
    """

    def __init__(self):
        self.items = get_items_df()
        self.items["text"] = (
            self.items["title"].fillna("") + " " +
            self.items["category"].fillna("") + " " +
            self.items["description"].fillna("")
        )
        self.vectorizer = TfidfVectorizer()
        self.item_tfidf = self.vectorizer.fit_transform(self.items["text"])
        self.item_id_to_index = {row.item_id: idx for idx, row in self.items.iterrows()}

    def recommend_similar_items(self, item_id: int, top_k: int = 5) -> List[int]:
        """
        Рекомендует товары, похожие на данный item_id.
        Возвращает список item_id.
        """
        if item_id not in self.item_id_to_index:
            return []

        idx = self.item_id_to_index[item_id]
        item_vec = self.item_tfidf[idx]
        sims = cosine_similarity(item_vec, self.item_tfidf)[0]
        similar_indices = sims.argsort()[::-1]
        result = []
        for i in similar_indices:
            if i == idx:
                continue
            result.append(int(self.items.iloc[i].item_id))
            if len(result) >= top_k:
                break
        return result

    def recommend_for_user(self, user_id: int, top_k: int = 5) -> List[int]:
        """
        Простейший вариант: берём последний просмотренный/купленный товар
        и ищем похожие на него.
        """
        interactions = get_interactions_df()
        user_hist = interactions[interactions["user_id"] == user_id]
        if user_hist.empty:
            return []
        last_item_id = int(user_hist.iloc[-1].item_id)
        return self.recommend_similar_items(last_item_id, top_k=top_k)
