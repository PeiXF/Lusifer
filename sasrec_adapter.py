"""
SASRec adapter: Wrap the SASRec model into a unified interface that Lusifer can call.

Note: This is a minimum runnable "placeholder implementation" to quickly connect the input and output process in the Lusifer framework. 
You can replace it with the actual SASRec training code and checkpoint.
"""

from typing import Dict, Iterable, List

import logging
import os
import pickle

import numpy as np
import pandas as pd
# from sasrec_inference import SARSEC_inference

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


class SASRecRecommender:
    """
    Real SASRec model wrapper.

    Requires environment variable:
      SASREC_MODEL_DIR = directory containing meta.pkl and sasrec_weights/
    """

    def __init__(self, model_dir: str):
        self.model_dir = model_dir
        meta_path = os.path.join(model_dir, "meta.pkl")
        if not os.path.exists(meta_path):
            raise FileNotFoundError(f"SASRec meta file not found at '{meta_path}'")

        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        # mapping saved during training: external id <-> internal item_id
        self.item_id_for_extid: Dict = meta["item_id_for_extid"]
        self.extid_for_itemid: Dict = meta["external_id_for_item_id"]
        desc = meta["Sasrec_model_descritption"]

        self.model = SARSEC_inference(
            item_num=meta["num_items_training"],
            seq_max_len=desc["maxlen"],
            embedding_dim=desc["hidden_units"],
            attention_dim=desc["hidden_units"],
            conv_dims=[desc["hidden_units"], desc["hidden_units"]],
        )

        weights_path = os.path.join(model_dir, "sasrec_weights")
        self.model.load_weights(weights_path)
        logging.info(f"Loaded SASRec weights from: {weights_path}")

        # construct "full candidate set" from item_id 1 to item_num
        self.candidates = np.array([[i for i in range(1, self.model.item_num + 1)]])

    def _prepare_sequence(self, history_extids: List[int]) -> np.ndarray:
        """map external movie_id sequence to internal item_id, and pad to seq_max_len."""
        seq_ids = [
            self.item_id_for_extid[e]
            for e in history_extids
            if e in self.item_id_for_extid
        ]
        if not seq_ids:
            return np.zeros((1, self.model.seq_max_len), dtype=np.int64)

        seq_ids = seq_ids[-self.model.seq_max_len :]
        padded = np.pad(
            seq_ids,
            (self.model.seq_max_len - len(seq_ids), 0),
            mode="constant",
        )
        return np.array([padded], dtype=np.int64)

    def score(
        self,
        user_id: int,
        user_history: List[int],
        candidates: Iterable[int],
    ) -> Dict[int, float]:
        """
        Return the raw score (model logits) for each candidate, keyed by the external movie_id.
        Here we assume that the external id during training SASRec is consistent with the movie_id in Lusifer.
        """
        input_seq = self._prepare_sequence(user_history)
        if not input_seq.any():
            return {}

        inputs = {"input_seq": input_seq, "candidate": self.candidates}
        logits = self.model.predict(inputs).numpy().flatten()  # shape: [item_num]

        scores: Dict[int, float] = {}
        for extid in candidates:
            extid = int(extid)
            if extid not in self.item_id_for_extid:
                continue
            internal_id = self.item_id_for_extid[extid]
            # internal_id starts from 1, corresponding to logits[internal_id - 1]
            scores[extid] = float(logits[internal_id - 1])
        return scores



def build_sasrec_model_or_dummy(rating_df: pd.DataFrame):
    """Build real SASRec model, otherwise fall back to Dummy implementation."""
    model_dir = os.getenv("SASREC_MODEL_DIR")
    if model_dir:
        try:
            logging.info(f"Initializing real SASRec model from '{model_dir}'")
            return SASRecRecommender(model_dir=model_dir)
        except Exception as exc:
            logging.warning(
                f"Failed to init SASRec model: {exc}. Falling back to DummySASRecRecommender."
            )
    else:
        logging.info("SASREC_MODEL_DIR not set, using DummySASRecRecommender.")
    return DummySASRecRecommender(rating_df=rating_df)

    
def score_candidates_with_sasrec(
    sasrec_model,
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

    raw_scores = sasrec_model.score(
        user_id=user_id, user_history=user_history, candidates=candidates
    )

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


