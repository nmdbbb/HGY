"""BPR-MF phải học được THỨ TỰ, và phải không tự lừa mình bằng mẫu âm trùng."""

import numpy as np
from scipy import sparse

from truyen_reco.models.bpr import BPRMF


def toy_matrix():
    """Hai nhóm người đọc hai nhóm truyện rời nhau — thứ tự đúng phải học được."""
    rows, cols = [], []
    for user in range(20):
        group = range(0, 5) if user < 10 else range(5, 10)
        for item in group:
            rows.append(user)
            cols.append(item)
    return sparse.csr_matrix((np.ones(len(rows), dtype=np.float32),
                              (rows, cols)), shape=(20, 10))


def test_hoc_dung_thu_tu_tren_du_lieu_hai_nhom():
    model = BPRMF(factors=8, iterations=60, batch_size=32, seed=0)
    model.fit(toy_matrix())
    for user, own, other in ((0, range(0, 5), range(5, 10)),
                             (15, range(5, 10), range(0, 5))):
        scores = model.scores_for(user)
        assert scores[list(own)].mean() > scores[list(other)].mean(), (
            f"người {user}: điểm nhóm của mình phải cao hơn nhóm kia")


def test_khong_lay_mau_am_trung_voi_muc_da_doc():
    """Người đọc HẾT danh mục trừ một mục: mẫu âm hợp lệ chỉ còn đúng mục đó.

    Nếu khâu lấy lại mẫu bị bỏ thì phần lớn bước gradient sẽ ghép một mục dương
    với chính một mục dương khác, tức dạy mô hình rằng thứ tự giữa hai mục nó
    đều thích là một điều cần tối ưu — sai dấu một cách im lặng.
    """
    n_items = 12
    cols = list(range(n_items - 1))
    matrix = sparse.csr_matrix(
        (np.ones(len(cols), dtype=np.float32), ([0] * len(cols), cols)),
        shape=(1, n_items))
    model = BPRMF(factors=4, iterations=25, batch_size=8, seed=1).fit(matrix)
    scores = model.scores_for(0)
    assert scores[n_items - 1] < scores[:n_items - 1].mean(), (
        "mục duy nhất chưa đọc phải bị xếp dưới trung bình các mục đã đọc")


def test_tai_lap_duoc_theo_seed():
    a = BPRMF(factors=4, iterations=5, batch_size=16, seed=7).fit(toy_matrix())
    b = BPRMF(factors=4, iterations=5, batch_size=16, seed=7).fit(toy_matrix())
    assert np.allclose(a.item_factors, b.item_factors)
