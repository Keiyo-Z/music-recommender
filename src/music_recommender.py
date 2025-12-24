import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List

def precision_k(actual: List, predicted: List, k: int = 10) -> float:
    """Precision@K"""
    if k <= 0:
        return 0.0

    predicted_k = predicted[:k]
    if len(predicted_k) == 0:
        return 0.0

    actual_set = set(actual)
    hit_count = len(set(predicted_k) & actual_set)
    return hit_count / k


def recall_k(actual: List, predicted: List, k: int = 10) -> float:
    """Recall@K"""
    if len(actual) == 0:
        return 0.0

    predicted_k = predicted[:k]
    actual_set = set(actual)
    hit_count = len(set(predicted_k) & actual_set)
    return hit_count / len(actual)


def ap_k(actual: List, predicted: List, k: int = 10) -> float:
    """Average Precision@K"""
    if len(actual) == 0:
        return 0.0

    actual_set = set(actual)
    score = 0.0
    hits = 0

    for i, p in enumerate(predicted[:k]):
        if p in actual_set:
            hits += 1
            score += hits / (i + 1)

    return score / min(len(actual), k)


def map_k(actuals: List[List], predicteds: List[List], k: int = 10) -> float:
    """Mean Average Precision@K"""
    if len(actuals) == 0:
        return 0.0

    ap_scores = [
        ap_k(actual, predicted, k) for actual, predicted in zip(actuals, predicteds)
    ]
    return sum(ap_scores) / len(ap_scores)

class ItemBasedCFRecommender:

    def __init__(self, df: pd.DataFrame):
        df = df.dropna(subset=["トラック名", "追加者"])

        df["user_id"] = df["追加者"].astype(str)
        df["item_id"] = df["トラック名"].astype(str)

        df["人気度"] = pd.to_numeric(df["人気度"], errors="coerce")

        self.df = df

        inter = df[["user_id", "item_id"]].copy()
        inter["value"] = 1

        user_item = inter.pivot_table(
            index="user_id",
            columns="item_id",
            values="value",
            aggfunc="max",
            fill_value=0,
        )
        self.user_item = user_item

        self.user_ids = list(user_item.index)
        self.item_ids = list(user_item.columns)
        self.user_index = {u: i for i, u in enumerate(self.user_ids)}
        self.item_index = {it: j for j, it in enumerate(self.item_ids)}

        # R: (n_users, n_items)
        self.R = user_item.values.astype(float)

        user_mean = self.R.mean(axis=1, keepdims=True)  # shape: (n_users, 1)
        R_user_centered = self.R - user_mean            # shape: (n_users, n_items)

        self.item_sim_matrix = cosine_similarity(R_user_centered.T)

        self.popularity_rank = (
            df.groupby("item_id")["人気度"]
              .mean()
              .sort_values(ascending=False)
              .index
              .tolist()
        )

    def predict_score(self, user_id: str, item_id: str, k_neighbors: int = 5) -> float:

        if user_id not in self.user_index or item_id not in self.item_index:
            return 0.0

        u_idx = self.user_index[user_id]
        i_idx = self.item_index[item_id]

        user_vector = self.R[u_idx]  # shape: (n_items,)
        rated_idx = np.where(user_vector > 0)[0]
        if len(rated_idx) == 0:
            return 0.0

        sims = self.item_sim_matrix[i_idx]  # shape: (n_items,)

        neighbor_idx = rated_idx[np.argsort(sims[rated_idx])[::-1]][:k_neighbors]

        weighted_sum = 0.0
        sim_sum = 0.0
        for j in neighbor_idx:
            r_uj = self.R[u_idx, j]  # 0/1
            if r_uj <= 0:
                continue
            sim_ij = sims[j]
            weighted_sum += sim_ij * r_uj
            sim_sum += abs(sim_ij)

        if sim_sum == 0.0:
            return 0.0

        score = weighted_sum / sim_sum

        return float(np.clip(score, 0.0, 1.0))

    def recommend(self, user_id: str, top_k: int = 5, k_neighbors: int = 5) -> List[str]:

        if user_id not in self.user_index:
            return self.popularity_rank[:top_k]

        u_idx = self.user_index[user_id]
        user_vector = self.R[u_idx]

        scores = []
        for j, item_id in enumerate(self.item_ids):
            if user_vector[j] > 0:
                continue
            s = self.predict_score(user_id, item_id, k_neighbors)
            scores.append((item_id, s))

        scores.sort(key=lambda x: x[1], reverse=True)
        recs = [it for it, _ in scores[:top_k]]

        if len(recs) < top_k:
            already = set(recs) | set(
                self.item_ids[j] for j, v in enumerate(user_vector) if v > 0
            )
            for it in self.popularity_rank:
                if it not in already:
                    recs.append(it)
                if len(recs) >= top_k:
                    break

        return recs


def load_tracks_fixed(path: str) -> pd.DataFrame:
    """
    Load track list from either TSV (tab-separated) or CSV (comma-separated).
    - First tries tab-separated.
    - If the header does not match expected columns, falls back to comma-separated parsing.
    """

    expected_cols = [
        "トラック名",
        "アーティスト名",
        "アルバム名",
        "アルバムアーティスト名",
        "トラックの長さ（ミリ秒）",
        "人気度",
        "追加者",
        "追加日時",
    ]

    try:
        df_tsv = pd.read_csv(path, sep="\t", encoding="utf-8-sig", engine="python")
        if all(col in df_tsv.columns for col in expected_cols):
            return df_tsv[expected_cols]
    except Exception:
        pass

    with open(path, "r", encoding="utf-8-sig", errors="ignore") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]

    data_lines = lines[1:]

    rows = []
    for line in data_lines:
        parts = line.split(",")
        if len(parts) < 8:
            parts = parts + [""] * (8 - len(parts))
        elif len(parts) > 8:
            head = parts[:7]
            tail = ",".join(parts[7:])
            parts = head + [tail]
        rows.append(parts)

    df = pd.DataFrame(rows, columns=expected_cols)
    return df

if __name__ == "__main__":
    csv_path = "D:/OneDrive - 奈良先端科学技術大学院大学/Y1後期/ソフトウェアシステム開発/music-recommender/data/sample_musiclist.txt"

    df = load_tracks_fixed(csv_path)
    # print(df.head())
    # print(df.columns)

    recommender = ItemBasedCFRecommender(df)

    target_user = "8"
    recommended_tracks = recommender.recommend(target_user, top_k=5, k_neighbors=5)

    print(f"Recommendation for user {target_user} based on Item-based CF:")
    for rank, track in enumerate(recommended_tracks, start=1):
        row = df[df["トラック名"] == track].iloc[0]
        artist = row["アーティスト名"]
        album = row["アルバム名"]
        print(f"{rank}. {track} / {artist} ({album})")
