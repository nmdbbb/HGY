"""Láng giềng item-item trên ma trận thưa, có hạ trọng số cặp mỏng.

Tách ra khỏi script vì nó là một **mô hình**, không phải một bước chạy: hai
script dùng nó (`run_protocol_baselines.py`, `tune_protocol.py`) và nó có một
chỗ dễ sai cần test giữ.

## Chỗ dễ sai: `shrink = 0` từng tạo ra nan

Hệ số hạ trọng số là `n_chung / (n_chung + shrink)` — significance weighting mà
đề xuất đồ án nêu ở mục 5.2: hai truyện chỉ có vài người đọc chung có thể cho
cosine bằng 1,0 mà chẳng nói gì, nên nhân thêm hệ số này để kéo chúng về 0.

Với `shrink = 0` thì cặp **không** đồng xuất hiện cho `0 / 0 = nan`, và nan lan
đúng theo đường tệ nhất: `np.argpartition(-line, k)` không có thứ tự xác định
với nan nên nan có thể chiếm chỗ trong k phần tử đầu, rồi bị lọc ở bước
`line[picked] > 0` (vì `nan > 0` là False). Kết quả là mô hình có **ít láng
giềng hơn k** một cách không kiểm soát, không có cảnh báo nào ngoài một dòng
RuntimeWarning dễ bỏ qua — và `shrink = 0` lại chính là cấu hình mà tập
validation chọn.

Sửa bằng phép chia có điều kiện: cặp `n_chung = 0` cho hệ số 0, đúng về mặt
nghĩa (không có người đọc chung thì tương đồng bằng 0).
"""

import numpy as np
from scipy import sparse


def user_item_matrix(frame, user_index, item_index, weight_column=None):
    """Ma trận nhị phân (người x mục) từ bảng tương tác."""
    rows = frame["user"].map(user_index)
    cols = frame["media_id"].map(item_index)
    keep = rows.notna() & cols.notna()
    if weight_column is not None and weight_column in frame.columns:
        values = frame.loc[keep, weight_column].to_numpy(dtype=np.float32)
    else:
        values = np.ones(int(keep.sum()), dtype=np.float32)
    return sparse.csr_matrix(
        (values, (rows[keep].astype(np.int32), cols[keep].astype(np.int32))),
        shape=(len(user_index), len(item_index)))


def top_k_neighbours(matrix, k=200, block=512, shrink=20.0):
    """Cosine giữa các cột, giữ k láng giềng mỗi mục, có hạ trọng số cặp mỏng.

    Trả về csr_matrix (mục x mục), đường chéo bằng 0, không bao giờ chứa nan.
    """
    norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1
    norms[norms == 0] = 1.0
    n_items = matrix.shape[1]
    transposed = matrix.T.tocsr()
    indptr, indices, values = [0], [], []
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        counts = (transposed[start:stop] @ matrix).toarray()
        similarity = counts / (norms[start:stop, None] * norms[None, :])
        denominator = counts + shrink
        # Phép chia có điều kiện: 0/0 phải ra 0, không ra nan (xem docstring).
        factor = np.divide(counts, denominator, out=np.zeros_like(counts),
                           where=denominator > 0)
        similarity *= factor
        for row in range(stop - start):
            similarity[row, start + row] = 0.0
            line = similarity[row]
            if k < n_items:
                picked = np.argpartition(-line, k)[:k]
            else:
                picked = np.arange(n_items)
            picked = picked[line[picked] > 0]
            indices.append(picked.astype(np.int32))
            values.append(line[picked].astype(np.float32))
            indptr.append(indptr[-1] + len(picked))
    if not indices:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    return sparse.csr_matrix(
        (np.concatenate(values), np.concatenate(indices), np.array(indptr)),
        shape=(n_items, n_items))
