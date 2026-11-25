"""Láng giềng item-item: ba tính chất, một trong đó từng bị nan phá."""

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

from truyen_reco.models.knn import top_k_neighbours, user_item_matrix


def toy():
    # 1 và 2 luôn đi cùng nhau (6 người); 3 và 4 chỉ gặp nhau 1 lần.
    rows, cols = [], []
    for user in range(6):
        rows += [user, user]
        cols += [0, 1]
    rows += [6, 6]
    cols += [2, 3]
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32),
                              (rows, cols)), shape=(7, 5))


def test_khong_bao_gio_co_nan_ke_ca_khi_shrink_bang_0():
    """`shrink=0` từng cho 0/0 = nan ở mọi cặp không đồng xuất hiện, rồi nan
    chiếm chỗ trong top-k và bị lọc, làm mô hình có ít láng giềng hơn k."""
    for shrink in (0.0, 1.0, 20.0):
        similarity = top_k_neighbours(toy(), k=3, block=2, shrink=shrink)
        assert not np.isnan(similarity.data).any(), f"shrink={shrink} sinh nan"


def test_ha_trong_so_keo_cap_mong_xuong_chu_khong_doi_cap_day():
    no_shrink = top_k_neighbours(toy(), k=4, shrink=0.0).toarray()
    shrunk = top_k_neighbours(toy(), k=4, shrink=20.0).toarray()
    # Cặp (0,1): 6 người đọc chung -> cosine 1,0, hạ còn 6/26.
    # float32 nên so xấp xỉ: 6/6 ra 0,99999994, không phải 1,0 chẵn.
    assert no_shrink[0, 1] == pytest.approx(1.0, abs=1e-6)
    assert shrunk[0, 1] < no_shrink[0, 1]
    # Cặp (2,3): 1 người chung, cũng cosine 1,0 nếu không hạ — đây chính là
    # trường hợp significance weighting tồn tại để xử lý.
    assert no_shrink[2, 3] == pytest.approx(1.0, abs=1e-6)
    assert shrunk[2, 3] < shrunk[0, 1], (
        "cặp 1 người chung phải bị kéo xuống dưới cặp 6 người chung")


def test_duong_cheo_bang_0_va_khong_vuot_qua_k():
    similarity = top_k_neighbours(toy(), k=2, block=2, shrink=1.0)
    dense = similarity.toarray()
    assert np.allclose(np.diag(dense), 0.0)
    assert (np.count_nonzero(dense, axis=1) <= 2).all()


def test_dung_ma_tran_tu_bang_tuong_tac():
    frame = pd.DataFrame({"user": ["a", "a", "b"], "media_id": [10, 11, 10]})
    matrix = user_item_matrix(frame, {"a": 0, "b": 1}, {10: 0, 11: 1})
    assert matrix.shape == (2, 2) and matrix.nnz == 3
    # Khoá lạ bị bỏ, không làm hỏng hình dạng.
    frame2 = pd.DataFrame({"user": ["a", "z"], "media_id": [10, 99]})
    assert user_item_matrix(frame2, {"a": 0}, {10: 0}).nnz == 1
