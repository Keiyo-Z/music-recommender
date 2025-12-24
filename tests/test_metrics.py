from src.music_recommender import precision_k, recall_k, ap_k, map_k

def test_precision_k_basic():
    actual = ["a", "b"]
    predicted = ["a", "c", "d"]
    assert precision_k(actual, predicted, k=2) == 0.5  # a

def test_recall_k_basic():
    actual = ["a", "b", "c", "d"]
    predicted = ["a", "x", "b"]
    assert recall_k(actual, predicted, k=3) == 0.5  # a,b out of 4

def test_ap_k():
    actual = ["a", "b"]
    predicted = ["a", "x", "b"]
    expected = (1.0 + 2/3) / 2
    assert abs(ap_k(actual, predicted, k=3) - expected) < 1e-9

def test_map_k():
    actuals = [["a"], ["b", "c"]]
    predicteds = [["a", "x"], ["c", "b"]]
    score = map_k(actuals, predicteds, k=2)
    assert 0.0 <= score <= 1.0