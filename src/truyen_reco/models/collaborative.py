"""Lọc cộng tác bằng ALS cho implicit feedback (Hu, Koren, Volinsky 2008).

Tự cài thay vì dùng thư viện ngoài: không phụ thuộc phần mở rộng C++, chạy
được ở mọi máy trong nhóm, và kiểm soát được từng bước.

Bài toán là implicit: người đọc không chấm điểm hầu hết truyện họ đọc, nên
không quan sát được "ghét", chỉ quan sát được "có đọc" với độ tin cậy khác nhau.
"""

import numpy as np
import scipy.sparse as sp


class ALS:
    def __init__(self, factors=64, regularization=0.05, iterations=15,
                 alpha=20.0, seed=42):
        self.factors = factors
        self.regularization = regularization
        self.iterations = iterations
        self.alpha = alpha
        self.seed = seed
        self.user_factors = None
        self.item_factors = None

    def _solve(self, fixed, matrix):
        """Cập nhật một phía, giữ phía kia cố định."""
        n_rows = matrix.shape[0]
        out = np.zeros((n_rows, self.factors), dtype=np.float64)
        gram = fixed.T @ fixed + self.regularization * np.eye(self.factors)
        indptr, indices, data = matrix.indptr, matrix.indices, matrix.data

        for row in range(n_rows):
            start, end = indptr[row], indptr[row + 1]
            if start == end:
                continue
            cols = indices[start:end]
            confidence = data[start:end]
            block = fixed[cols]
            # (Y'Y + Y'(C-I)Y + λI) x = Y'C p,  với p = 1 ở mọi mục quan sát được
            left = gram + block.T @ ((confidence - 1.0)[:, None] * block)
            right = (confidence[:, None] * block).sum(axis=0)
            out[row] = np.linalg.solve(left, right)
        return out

    def fit(self, user_item):
        """user_item: csr_matrix (user x item), giá trị là trọng số tin cậy thô."""
        confidence = user_item.copy().astype(np.float64)
        confidence.data = 1.0 + self.alpha * confidence.data

        rng = np.random.default_rng(self.seed)
        n_users, n_items = confidence.shape
        self.user_factors = rng.normal(0, 0.01, (n_users, self.factors))
        self.item_factors = rng.normal(0, 0.01, (n_items, self.factors))

        item_user = confidence.T.tocsr()
        for _ in range(self.iterations):
            self.user_factors = self._solve(self.item_factors, confidence)
            self.item_factors = self._solve(self.user_factors, item_user)
        return self

    def score_users(self, user_indices):
        return self.user_factors[user_indices] @ self.item_factors.T

    def item_similarity(self, item_index, top_k=50):
        vectors = self.item_factors
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        normalized = vectors / np.maximum(norms, 1e-9)
        scores = normalized @ normalized[item_index]
        best = np.argpartition(-scores, min(top_k, len(scores) - 1))[:top_k]
        return best[np.argsort(-scores[best])]


def build_matrix(train, user_index, item_index, weight_column="confidence"):
    rows = train["user"].map(user_index)
    cols = train["media_id"].map(item_index)
    mask = rows.notna() & cols.notna()
    matrix = sp.csr_matrix(
        (train.loc[mask, weight_column].astype(float).values,
         (rows[mask].astype(int).values, cols[mask].astype(int).values)),
        shape=(len(user_index), len(item_index)),
    )
    matrix.sum_duplicates()
    return matrix


class PopularityModel:
    """Mốc so sánh bắt buộc. Trong hệ gợi ý, một mô hình thua popularity là
    mô hình không dùng được, nên luôn báo cáo nó cạnh mọi kết quả khác."""

    def __init__(self):
        self.scores = None

    def fit(self, user_item):
        self.scores = np.asarray(user_item.sum(axis=0)).ravel()
        return self

    def score(self, n_users=1):
        return np.tile(self.scores, (n_users, 1))
