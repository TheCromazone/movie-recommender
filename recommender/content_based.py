"""
content_based.py — Content-based filtering using TF-IDF and cosine similarity.

Builds a "soup" of movie metadata (genres, cast, director, keywords) and
computes pairwise cosine similarity to find similar titles.
"""

from __future__ import annotations

from typing import List
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class ContentBasedFilter:
    """
    Recommends movies similar to a given title based on metadata similarity.
    """

    def __init__(self, max_features: int = 5000, ngram_range=(1, 2)) -> None:
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words="english",
        )
        self._sim_matrix: np.ndarray | None = None
        self._movie_index: pd.Series | None = None
        self._movies_df: pd.DataFrame | None = None

    def _build_soup(self, row: pd.Series) -> str:
        """Combine metadata fields into a single text string."""
        parts = []
        if "genres" in row and pd.notna(row["genres"]):
            parts.append(str(row["genres"]).replace("|", " ").replace("-", " "))
        if "keywords" in row and pd.notna(row["keywords"]):
            parts.append(str(row["keywords"]))
        if "cast" in row and pd.notna(row["cast"]):
            # Take top 3 cast members, remove spaces from names
            cast_list = str(row["cast"]).split(",")[:3]
            parts.append(" ".join(c.strip().replace(" ", "") for c in cast_list))
        if "director" in row and pd.notna(row["director"]):
            parts.append(str(row["director"]).replace(" ", ""))
        # Repeat title tokens to boost title match
        if "title" in row and pd.notna(row["title"]):
            parts.append(str(row["title"]) * 2)
        return " ".join(parts).lower()

    def fit(self, movies_df: pd.DataFrame) -> "ContentBasedFilter":
        """
        Build the TF-IDF matrix and cosine similarity index.

        Args:
            movies_df: DataFrame with at least 'movieId' and 'title' columns.
                       Optionally: 'genres', 'keywords', 'cast', 'director'.
        """
        self._movies_df = movies_df.reset_index(drop=True)
        self._movie_index = pd.Series(
            self._movies_df.index, index=self._movies_df["title"].str.lower()
        )

        soup = self._movies_df.apply(self._build_soup, axis=1)
        tfidf_matrix = self._vectorizer.fit_transform(soup)

        # Compute cosine similarity (chunk for memory efficiency)
        self._sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)
        return self

    def recommend(self, title: str, top_n: int = 10) -> pd.DataFrame:
        """
        Return top-N movies most similar to the given title.

        Args:
            title:  Movie title string (case-insensitive).
            top_n:  Number of recommendations to return.

        Returns:
            DataFrame with 'movieId', 'title', and 'similarity_score'.
        """
        if self._sim_matrix is None:
            raise RuntimeError("Model not yet fitted. Call fit() first.")

        title_lower = title.lower().strip()
        if title_lower not in self._movie_index:
            # Fuzzy fallback
            matches = [t for t in self._movie_index.index if title_lower in t]
            if not matches:
                raise ValueError(f"Title '{title}' not found in the dataset.")
            title_lower = matches[0]

        idx = self._movie_index[title_lower]
        if isinstance(idx, pd.Series):
            idx = idx.iloc[0]

        sim_scores = list(enumerate(self._sim_matrix[idx]))
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [s for s in sim_scores if s[0] != idx][:top_n]

        movie_indices = [s[0] for s in sim_scores]
        scores = [s[1] for s in sim_scores]

        result = self._movies_df.iloc[movie_indices][["movieId", "title"]].copy()
        result["similarity_score"] = scores
        return result.reset_index(drop=True)
