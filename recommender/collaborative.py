"""
collaborative.py — User-based and item-based collaborative filtering.

Uses the `surprise` library for matrix factorization (SVD) and KNN-based methods.
"""

from __future__ import annotations

from pathlib import Path
import pickle
from typing import List, Tuple

import pandas as pd
from surprise import SVD, Dataset, Reader, KNNBasic
from surprise.model_selection import cross_validate


class CollaborativeFilter:
    """
    Trains SVD and KNN models on a user-item ratings matrix and
    provides top-N recommendations for a given user.
    """

    def __init__(self, algorithm: str = "svd") -> None:
        if algorithm == "svd":
            self.model = SVD(
                n_factors=100,
                n_epochs=20,
                lr_all=0.005,
                reg_all=0.02,
                random_state=42,
            )
        elif algorithm == "knn_user":
            self.model = KNNBasic(
                k=40,
                sim_options={"name": "cosine", "user_based": True},
            )
        elif algorithm == "knn_item":
            self.model = KNNBasic(
                k=40,
                sim_options={"name": "cosine", "user_based": False},
            )
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}")

        self._trainset = None
        self._all_items: List[int] = []

    def fit(self, ratings_df: pd.DataFrame) -> "CollaborativeFilter":
        """
        Train the model on a DataFrame with columns: userId, movieId, rating.
        """
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
        self._trainset = data.build_full_trainset()
        self._all_items = list(self._trainset.all_items())
        self.model.fit(self._trainset)
        return self

    def recommend(self, user_id: int, top_n: int = 10, exclude_seen: bool = True) -> List[Tuple[int, float]]:
        """
        Return top-N (movieId, predicted_rating) tuples for the given user.
        """
        if self._trainset is None:
            raise RuntimeError("Model not yet trained. Call fit() first.")

        seen = set()
        if exclude_seen and user_id in self._trainset._raw2inner_id_users:
            inner_uid = self._trainset.to_inner_uid(user_id)
            seen = {self._trainset.to_raw_iid(iid) for iid, _ in self._trainset.ur[inner_uid]}

        predictions = [
            (self._trainset.to_raw_iid(iid), self.model.predict(user_id, self._trainset.to_raw_iid(iid)).est)
            for iid in self._all_items
            if self._trainset.to_raw_iid(iid) not in seen
        ]
        predictions.sort(key=lambda x: x[1], reverse=True)
        return predictions[:top_n]

    def save(self, path: str | Path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path) -> "CollaborativeFilter":
        with open(path, "rb") as f:
            return pickle.load(f)

    def cross_validate(self, ratings_df: pd.DataFrame, n_splits: int = 5) -> dict:
        """Run k-fold cross-validation and return RMSE/MAE."""
        reader = Reader(rating_scale=(0.5, 5.0))
        data = Dataset.load_from_df(ratings_df[["userId", "movieId", "rating"]], reader)
        results = cross_validate(
            self.model, data, measures=["RMSE", "MAE"],
            cv=n_splits, verbose=False,
        )
        return {
            "rmse": results["test_rmse"].mean(),
            "mae": results["test_mae"].mean(),
        }
