"""
hybrid.py — Hybrid recommender blending collaborative and content-based signals.
"""

from __future__ import annotations

from typing import List
import pandas as pd

from .collaborative import CollaborativeFilter
from .content_based import ContentBasedFilter


class HybridRecommender:
    """
    Combines collaborative filtering and content-based filtering via
    weighted score blending.

    For users with sufficient rating history (>= min_ratings_for_cf),
    the hybrid uses both CF and CB. For new/cold-start users, it falls
    back to content-based only.
    """

    def __init__(
        self,
        cf_model: CollaborativeFilter,
        cb_model: ContentBasedFilter,
        movies_df: pd.DataFrame,
        alpha: float = 0.7,
        min_ratings_for_cf: int = 5,
    ) -> None:
        self._cf = cf_model
        self._cb = cb_model
        self._movies = movies_df.set_index("movieId")
        self._alpha = alpha
        self._min_ratings = min_ratings_for_cf

    def recommend_for_user(
        self,
        user_id: int,
        user_ratings: dict[int, float],
        top_n: int = 10,
    ) -> pd.DataFrame:
        """
        Generate top-N recommendations for a user.

        Args:
            user_id:       Numeric user ID.
            user_ratings:  Dict of {movieId: rating} the user has already rated.
            top_n:         Number of recommendations to return.

        Returns:
            DataFrame with movieId, title, hybrid_score.
        """
        use_cf = len(user_ratings) >= self._min_ratings

        if use_cf:
            cf_recs = dict(self._cf.recommend(user_id, top_n=top_n * 3))
        else:
            cf_recs = {}

        # Content-based: aggregate similarity to highly-rated movies
        liked = [mid for mid, r in user_ratings.items() if r >= 4.0]
        cb_scores: dict[int, float] = {}

        for movie_id in liked:
            if movie_id not in self._movies.index:
                continue
            title = self._movies.loc[movie_id, "title"]
            try:
                similar = self._cb.recommend(title, top_n=top_n * 2)
                for _, row in similar.iterrows():
                    mid = row["movieId"]
                    if mid not in user_ratings:
                        cb_scores[mid] = cb_scores.get(mid, 0) + row["similarity_score"]
            except ValueError:
                continue

        # Normalize scores to [0, 1]
        def normalize(d: dict) -> dict:
            if not d:
                return d
            mx = max(d.values())
            return {k: v / mx for k, v in d.items()} if mx > 0 else d

        cf_norm = normalize(cf_recs)
        cb_norm = normalize(cb_scores)

        all_ids = set(cf_norm) | set(cb_norm)
        alpha = self._alpha if use_cf else 0.0

        blended = {
            mid: alpha * cf_norm.get(mid, 0) + (1 - alpha) * cb_norm.get(mid, 0)
            for mid in all_ids
        }

        top_ids = sorted(blended, key=blended.get, reverse=True)[:top_n]
        rows = []
        for mid in top_ids:
            if mid in self._movies.index:
                rows.append({
                    "movieId": mid,
                    "title": self._movies.loc[mid, "title"],
                    "hybrid_score": round(blended[mid], 4),
                })

        return pd.DataFrame(rows)
