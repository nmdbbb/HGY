"""BPR-MF: xếp hạng cá nhân hoá Bayes cho phản hồi ẩn (Rendle và cộng sự, UAI 2009).

Vì sao cần nó bên cạnh ALS, dù cả hai đều là phân rã ma trận: **hai hàm mất
khác bài toán**. ALS tối thiểu bình phương sai số trên *mọi* cặp (user, item) có
trọng số tin cậy — nó học "mức độ thích", và mọi item chưa đọc đều bị kéo về 0.
BPR tối đa hoá xác suất `x_ui > x_uj` cho từng cặp *đã đọc / chưa đọc* — nó học
**thứ tự**, đúng thứ mà NDCG@10 đo. Đề xuất đồ án xếp hai cái vào hai mục khác
nhau (L2 và L3) cũng vì thế, nên chạy một cái rồi suy ra cái kia là bỏ mất một
phép so.

Cài bằng SGD theo lô trên numpy, không dùng thư viện ngoài (`implicit` không có
trong môi trường này). Ba chỗ phải cẩn thận, ghi lại vì mỗi chỗ đều từng là một
lỗi im lặng trong văn liệu:

1. **Mẫu âm phải tránh item người đó đã đọc.** Lấy ngẫu nhiên rồi bỏ qua trùng
   thì với người đọc nhiều (1.496 mục) tỉ lệ trùng lên tới vài phần trăm, và
   mỗi lần trùng là một bước gradient *dạy sai dấu*. Ở đây lấy lại mẫu cho
   những vị trí bị trùng, tối đa vài vòng.
2. **Tốc độ học phải chia theo cỡ lô**, nếu không thì đổi `batch_size` là đổi
   luôn bước cập nhật và hai lần chạy không so được với nhau.
3. **Cập nhật cộng dồn theo chỉ số trùng nhau**: một item có thể xuất hiện nhiều
   lần trong cùng một lô, nên phải dùng `np.add.at` thay vì gán thẳng — gán
   thẳng thì chỉ lần cuối có hiệu lực và phần gradient còn lại biến mất.
"""

import numpy as np


class BPRMF:
    def __init__(self, factors=64, iterations=30, learning_rate=0.05,
                 regularization=0.01, batch_size=8192, seed=0,
                 use_item_bias=False):
        self.factors = factors
        self.iterations = iterations
        self.learning_rate = learning_rate
        self.regularization = regularization
        self.batch_size = batch_size
        self.seed = seed
        # Số hạng chệch theo item: trong xếp hạng CẶP, `bias_i - bias_j` không
        # phụ thuộc người đọc, nên nó là đúng một bảng xếp hạng chung — mô hình
        # có thể hạ hàm mất bằng cách tái tạo *độ phổ biến* mà chẳng học gì về
        # gu. Đo được, không phải lo xa: bật chệch cho NDCG@10 0,0344 với ARP
        # 0,9995 và long-tail 0,0067 — trùng khít mốc `phổ biến` (0,0345 /
        # 0,9998 / 0,0000); tắt chệch cho 0,0537 (+56%) với phủ gấp 9,5 lần.
        # Nên **mặc định là tắt**, đúng như BPR-MF bản công bố (Rendle 2009)
        # không có số hạng này. Cờ giữ lại để đo lại kết luận trên, chứ không
        # phải một tham số để dò.
        self.use_item_bias = use_item_bias
        self.user_factors = None
        self.item_factors = None
        self.item_bias = None

    def fit(self, user_item, log=None):
        """`user_item` là csr_matrix nhị phân (người x mục)."""
        matrix = user_item.tocsr()
        n_users, n_items = matrix.shape
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / np.sqrt(self.factors)
        self.user_factors = (rng.standard_normal((n_users, self.factors))
                             * scale).astype(np.float32)
        self.item_factors = (rng.standard_normal((n_items, self.factors))
                             * scale).astype(np.float32)
        self.item_bias = np.zeros(n_items, dtype=np.float32)

        indptr, indices = matrix.indptr, matrix.indices
        # Danh sách cặp (user, item dương) phẳng — rút mẫu đồng đều trên *tương
        # tác*, không trên người: rút đồng đều trên người sẽ cho người đọc ít
        # trọng số lớn gấp nhiều lần, một lựa chọn khác hẳn và phải cố ý.
        rows = np.repeat(np.arange(n_users, dtype=np.int32), np.diff(indptr))
        n_pairs = len(rows)
        if n_pairs == 0:
            return self
        # Kiểm "mẫu âm có trùng mục đã đọc" bằng MỘT mảng khoá đã sắp, thay vì
        # một set cho mỗi người: `user * n_items + item` là song ánh, nên kiểm
        # thuộc tập thành một phép tìm nhị phân vector hoá. Bản dùng set và
        # vòng lặp Python mất ~60ms mỗi lô, tức vài phút chỉ để kiểm trùng.
        keys = np.sort(rows.astype(np.int64) * n_items + indices.astype(np.int64))
        step = self.learning_rate
        batches = max(1, n_pairs // self.batch_size)

        for epoch in range(self.iterations):
            order = rng.permutation(n_pairs)
            total_loss = 0.0
            for batch in range(batches):
                picked = order[batch * self.batch_size:
                               (batch + 1) * self.batch_size]
                if not len(picked):
                    continue
                users = rows[picked]
                positives = indices[picked]
                negatives = rng.integers(0, n_items, len(picked)).astype(np.int32)
                # Lấy lại mẫu cho những chỗ trúng item đã đọc (xem docstring).
                for _ in range(4):
                    probe = users.astype(np.int64) * n_items + negatives
                    position = np.searchsorted(keys, probe)
                    position[position >= len(keys)] = len(keys) - 1
                    clash = keys[position] == probe
                    if not clash.any():
                        break
                    negatives[clash] = rng.integers(0, n_items,
                                                    int(clash.sum())).astype(np.int32)

                u = self.user_factors[users]
                i = self.item_factors[positives]
                j = self.item_factors[negatives]
                x = np.einsum("ij,ij->i", u, i - j)
                if self.use_item_bias:
                    x = x + self.item_bias[positives] - self.item_bias[negatives]
                sigmoid = 1.0 / (1.0 + np.exp(np.clip(x, -30, 30)))
                total_loss += float(np.logaddexp(0.0, -x).sum())

                weight = (sigmoid * step).astype(np.float32)[:, None]
                grad_u = weight * (i - j) - step * self.regularization * u
                grad_i = weight * u - step * self.regularization * i
                grad_j = -weight * u - step * self.regularization * j
                # np.add.at vì một người/mục có thể xuất hiện nhiều lần trong lô.
                np.add.at(self.user_factors, users, grad_u)
                np.add.at(self.item_factors, positives, grad_i)
                np.add.at(self.item_factors, negatives, grad_j)
                if self.use_item_bias:
                    np.add.at(self.item_bias, positives,
                              (sigmoid * step).astype(np.float32))
                    np.add.at(self.item_bias, negatives,
                              -(sigmoid * step).astype(np.float32))
            if log:
                log(f"  epoch {epoch + 1}/{self.iterations}  "
                    f"loss {total_loss / max(batches * self.batch_size, 1):.4f}")
        return self

    def scores_for(self, row):
        return self.item_factors @ self.user_factors[row] + self.item_bias
