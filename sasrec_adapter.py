"""
SASRec adapter: Wrap the SASRec model into a unified interface that Lusifer can call.

Note: This is a minimum runnable "placeholder implementation" to quickly connect the input and output process in the Lusifer framework. 
You can replace it with the actual SASRec training code and checkpoint.
"""

from typing import Dict, Iterable, List

import numpy as np
import pandas as pd


class DummySASRecRecommender:
    """
    Placeholder SASRec recommender, just score the candidates based on the global average rating of the items.
    You can replace its internal implementation with the actual SASRec inference logic.
    """

    def __init__(self, rating_df: pd.DataFrame):
        """
        :param rating_df: Movielens training set rating table, at least contains 'movie_id', 'rating'
        """
        self.rating_df = rating_df
        # estimated the average rating of each item, as a simple baseline
        self.item_mean = (
            rating_df.groupby("movie_id")["rating"].mean().to_dict()
            if "movie_id" in rating_df.columns
            else {}
        )
        self.global_mean = float(rating_df["rating"].mean()) if "rating" in rating_df.columns else 3.0

    def score(
        self,
        user_id: int,
        user_history: List[int],
        candidates: Iterable[int],
    ) -> Dict[int, float]:
        """
        Given a user and a candidate set, output a score for each candidate.

        in the actual SASRec version, you should:
        1. map user_history and candidates to internal item ids;
        2. call the SASRec model for forward, get the score for each candidate;
        3. map the score back to movie_id.
        """
        scores: Dict[int, float] = {}
        for mid in candidates:
            mid = int(mid)
            scores[mid] = float(self.item_mean.get(mid, self.global_mean))
        return scores


def score_candidates_with_sasrec(
    sasrec_model: DummySASRecRecommender,
    *,
    user_id: int,
    rating_df: pd.DataFrame,
    candidates: Iterable[int],
    history_size: int = 10,
    min_rating: int = 1,
    max_rating: int = 5,
) -> Dict[int, int]:
    """
    The unified "score candidates" interface, returning {movie_id: integer rating from 1 to 5}.
    This is consistent with the interface in pa_base_adapter, making it easy to switch between movielens1m_example.
    """
    candidates = list(candidates)
    if not candidates:
        return {}

    user_history = (
        rating_df.loc[rating_df["user_id"] == user_id]
        .sort_values("timestamp", ascending=False)["movie_id"]
        .head(history_size)
        .tolist()
    )

    raw_scores = sasrec_model.score(user_id=user_id, user_history=user_history, candidates=candidates)

    # map any real-valued score to [min_rating, max_rating], then round to an integer
    if raw_scores:
        values = np.array(list(raw_scores.values()), dtype=float)
        vmin, vmax = float(values.min()), float(values.max())
    else:
        vmin, vmax = 0.0, 0.0

    results: Dict[int, int] = {}
    for mid, s in raw_scores.items():
        if vmax > vmin:
            norm = (float(s) - vmin) / (vmax - vmin)
        else:
            norm = 0.5
        rating = min_rating + norm * (max_rating - min_rating)
        rating = max(min_rating, min(max_rating, rating))
        results[int(mid)] = int(round(rating))

    # if some candidates did not get a score, fall back to the global mean
    fallback = int(round((min_rating + max_rating) / 2))
    for mid in candidates:
        mid = int(mid)
        if mid not in results:
            results[mid] = fallback

    return results


