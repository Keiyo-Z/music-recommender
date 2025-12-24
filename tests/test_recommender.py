from pathlib import Path
from src.music_recommender import load_tracks_fixed, ItemBasedCFRecommender

def test_recommend_known_and_unknown_user():
    df = load_tracks_fixed(str(Path("data/sample_musiclist.txt")))
    rec = ItemBasedCFRecommender(df)

    # 既知ユーザ（1）は推薦が top_k 件返る
    r1 = rec.recommend("1", top_k=3, k_neighbors=2)
    assert isinstance(r1, list)
    assert len(r1) == 3

    # 未知ユーザは人気度順フォールバック
    r_unknown = rec.recommend("999", top_k=3, k_neighbors=2)
    assert len(r_unknown) == 3
    # フォールバックが popularity_rank から来ていることを軽く確認
    assert r_unknown[0] in rec.popularity_rank

def test_predict_score_range():
    df = load_tracks_fixed(str(Path("data/sample_musiclist.txt")))
    rec = ItemBasedCFRecommender(df)

    # 存在するユーザとアイテムでスコアが0~1
    any_user = rec.user_ids[0]
    any_item = rec.item_ids[0]
    s = rec.predict_score(any_user, any_item, k_neighbors=2)
    assert 0.0 <= s <= 1.0