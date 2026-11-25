import numpy as np
import scipy.sparse as sp

from truyen_reco.evaluation import metrics
from truyen_reco.models.collaborative import ALS
from truyen_reco.models.content_based import FacetModel


def blocky_matrix(n_users=60, n_items=40, seed=0):
    """Hai nhóm người đọc tách bạch — ALS phải tách lại được."""
    rng = np.random.default_rng(seed)
    matrix = np.zeros((n_users, n_items))
    half_u, half_i = n_users // 2, n_items // 2
    matrix[:half_u, :half_i] = rng.random((half_u, half_i)) < 0.5
    matrix[half_u:, half_i:] = rng.random((n_users - half_u, n_items - half_i)) < 0.5
    return sp.csr_matrix(matrix)


def test_als_recovers_group_structure():
    matrix = blocky_matrix()
    model = ALS(factors=8, iterations=12).fit(matrix)
    scores = model.user_factors @ model.item_factors.T
    half_u, half_i = 30, 20
    # Người nhóm 1 phải được chấm điểm cao hơn ở các truyện nhóm 1
    assert scores[:half_u, :half_i].mean() > scores[:half_u, half_i:].mean()
    assert scores[half_u:, half_i:].mean() > scores[half_u:, :half_i].mean()


def test_als_factors_are_finite():
    model = ALS(factors=8, iterations=5).fit(blocky_matrix())
    assert np.isfinite(model.user_factors).all()
    assert np.isfinite(model.item_factors).all()


def test_missing_facet_scores_nan_not_zero():
    """Truyện thiếu ảnh bìa phải trả NaN để tầng trên chuẩn hoá lại trọng số,
    chứ không phải 0 — nếu không nó bị phạt oan như thể nét vẽ rất khác."""
    vectors = np.zeros((4, 3))
    vectors[0] = [1.0, 0.0, 0.0]
    vectors[1] = [0.0, 1.0, 0.0]
    model = FacetModel({"art": vectors}, {10: 0, 11: 1, 12: 2, 13: 3})
    profile = model.profile([0])
    scores = model.score(profile)
    assert np.isfinite(scores["art"][0])
    assert np.isnan(scores["art"][2])
    assert np.isnan(scores["art"][3])


def test_coherence_higher_for_tight_group():
    vectors = np.array([[1.0, 0.0], [0.99, 0.14], [0.0, 1.0], [-1.0, 0.0]])
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    model = FacetModel({"art": vectors}, {i: i for i in range(4)})
    tight = model.coherence([0, 1])["art"]
    spread = model.coherence([0, 3])["art"]
    assert tight > spread


def test_ndcg_rewards_ranking_order():
    relevant = {"a", "b"}
    good = metrics.ndcg_at_k(["a", "b", "c", "d"], relevant, 4)
    bad = metrics.ndcg_at_k(["c", "d", "a", "b"], relevant, 4)
    assert good > bad
    assert good == 1.0


def test_creator_concentration_detects_repeats():
    same = metrics.creator_concentration(["x", "y"], {"x": ("A",), "y": ("A",)})
    different = metrics.creator_concentration(["x", "y"], {"x": ("A",), "y": ("B",)})
    assert same > different
    assert different == 0.0


def test_cover_budget_is_split_across_countries():
    """Ngân sách ảnh bìa không được dồn hết cho manga Nhật.

    Xếp theo độ phổ biến toàn cục thì KR/CN bị đẩy khỏi hạn mức, và mặt nét vẽ
    chỉ phủ một nửa danh mục.
    """
    import pandas as pd

    from truyen_reco.features.art import select_for_covers

    rows = ([{"media_id": i, "country": "JP", "popularity": 100_000 - i}
             for i in range(200)]
            + [{"media_id": 1000 + i, "country": "KR", "popularity": 500 - i}
               for i in range(100)]
            + [{"media_id": 2000 + i, "country": "CN", "popularity": 50 - i}
               for i in range(50)])
    picked = select_for_covers(pd.DataFrame(rows), max_items=70)
    mix = picked["country"].value_counts()
    assert set(mix.index) == {"JP", "KR", "CN"}
    assert mix["KR"] >= 15 and mix["CN"] >= 5
