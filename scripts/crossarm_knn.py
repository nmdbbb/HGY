"""Phép thử quyết định cho câu "so hai hướng": chạy mô hình của NHÁNH GIAO THỨC
dưới GIAO THỨC của nhánh dữ liệu, trên CÙNG bộ Goodreads.

    python scripts/crossarm_knn.py --out-dir data/processed_gr

Vì sao cần. Hai nhánh cho hai dải số rất khác nhau — Session-KNN 0,1273 ở nhánh
giao thức so với hai tháp 0,0625 ở nhánh dữ liệu — nhưng chúng **không so trực
tiếp được**: khác định nghĩa dương (rating >= 4 so với mọi tương tác), khác
k-core (10/5 so với 5/5), khác phép chia (70/10/20 theo từng người so với cắt
theo thời điểm toàn cục), khác phép loại ứng viên (nhánh giao thức loại cả mục
cùng bộ). Cách duy nhất tôi có để nối hai dải là **tỉ lệ so với ALS**, vì ALS
chạy ở cả hai nhánh — nhưng tỉ lệ là một phép chuẩn hoá, không phải một phép đo.

Script này bỏ hẳn phép chuẩn hoá: lấy đúng Item-KNN và Session-KNN của nhánh
giao thức, cho chạy dưới **giao thức của nhánh dữ liệu** (cùng train/test, cùng
lát, cùng phép loại "đã đọc", cùng `metrics.evaluate`). Khác biệt giao thức bị
triệt tiêu hoàn toàn; thứ còn lại chỉ là họ mô hình.

Chốt trước khi chạy:

- Nếu lợi thế của họ láng giềng là **tính chất của họ mô hình trên dữ liệu này**,
  Item-KNN và Session-KNN phải hơn ALS (0,0734 ở quy mô khớp) và hơn hai tháp
  (0,0625).
- Nếu nó là **tính chất của giao thức** — chia 70/10/20 theo từng người đặt kỳ
  test ngay sát kỳ train, nên mọi thứ dựa vào "vừa đọc gì" được lợi — thì dưới
  giao thức cắt theo thời điểm toàn cục chúng phải rơi về ngang hoặc dưới ALS.

Điều kiện bác bỏ giả thuyết "họ mô hình": **Item-KNN <= ALS** dưới giao thức này.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.knn import top_k_neighbours  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_embedding import (  # noqa: E402
    build_histories, load_state, preprocess_slices)


def knn_recommender(similarity, reverse, k=10, length=None, decay=1.0):
    """Gợi ý theo láng giềng item-item, loại mục đã đọc — đúng như nhánh dữ liệu.

    `length=None` là Item-KNN (toàn hồ sơ, trọng số đều). `length=L` cắt còn L
    mục gần nhất; `history` đã được `build_histories` sắp theo thời gian nên
    "gần nhất" ở đây là gần nhất thật, không phải cuối danh sách ngẫu nhiên.
    """
    n_items = similarity.shape[0]

    def recommend(_user, history):
        if not len(history):
            return []
        positions = np.asarray(history, dtype=np.int64)
        used = positions[-length:] if length else positions
        weights = (np.power(decay, np.arange(len(used) - 1, -1, -1))
                   if decay != 1.0 else np.ones(len(used)))
        profile = sparse.csr_matrix(
            (weights.astype(np.float32),
             (np.zeros(len(used), dtype=np.int32), used.astype(np.int32))),
            shape=(1, n_items))
        values = np.asarray((profile @ similarity).todense()).ravel().astype(float)
        values[positions] = -np.inf
        if not np.isfinite(values).any():
            return []
        top = np.argpartition(-values, k)[:k]
        top = top[np.argsort(-values[top])]
        return [reverse[int(i)] for i in top if np.isfinite(values[int(i)])]
    return recommend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed_gr", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=1500)
    parser.add_argument("--neighbours", type=int, default=500)
    parser.add_argument("--shrink", type=float, default=0.0)
    parser.add_argument("--lengths", nargs="+", type=int, default=[2, 5])
    parser.add_argument("--decay", type=float, default=0.7)
    args = parser.parse_args()

    started = time.monotonic()
    state = load_state(args.out_dir)
    item_index = state["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    train, test, cold_users = state["train"], state["test"], state["cold_users"]
    history, seed, holdout = build_histories(train, cold_users, item_index)
    combined = dict(history)
    combined.update(seed)

    slices = dict(preprocess_slices(train, test))
    cold = cold_users[[m in holdout.get(u, set())
                       for u, m in zip(cold_users["user"], cold_users["media_id"])]]
    slices["new_user"] = cold
    print(f"{args.out_dir}: {len(item_index):,} mục | người đọc train "
          f"{len(history):,} | người đọc mới {len(seed):,}")

    # Ma trận người x mục dựng TỪ CHÍNH `history` chứ không từ bảng train, để
    # chắc chắn mô hình thấy đúng những gì bộ gợi ý được đưa lúc chấm điểm.
    rows, cols = [], []
    for position, items in enumerate(history.values()):
        rows.extend([position] * len(items))
        cols.extend(items)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), dtype=np.float32), (rows, cols)),
        shape=(len(history), len(item_index)))
    print(f"  ma trận {matrix.shape} | {matrix.nnz:,} ô "
          f"({time.monotonic() - started:.0f}s)", flush=True)
    similarity = top_k_neighbours(matrix, k=args.neighbours, shrink=args.shrink)
    print(f"  tương đồng item-item xong ({time.monotonic() - started:.0f}s)",
          flush=True)

    names = list(slices)
    print("\n" + "=" * 88)
    print(f"NDCG@{args.k} dưới GIAO THỨC NHÁNH DỮ LIỆU (cắt theo thời điểm, "
          f"chỉ loại mục đã đọc)")
    print("=" * 88)
    print(f"{'mô hình':<28}" + "".join(f"{s:>12}" for s in names))
    print("-" * (28 + 12 * len(names)))

    results = {}
    todo = [(f"Item-KNN (k={args.neighbours})", None, 1.0)]
    todo += [(f"Session-KNN (L={L}, giảm {args.decay})", L, args.decay)
             for L in args.lengths]
    for label, length, decay in todo:
        fn = knn_recommender(similarity, reverse, args.k, length, decay)
        row = {}
        line = f"{label:<28}"
        for name in names:
            got = metrics.evaluate(fn, slices[name], combined, k=args.k,
                                   max_users=args.max_users,
                                   catalog_size=len(item_index))
            row[name] = got
            line += f"{got['ndcg']:>12.4f}"
        results[label] = row
        print(line, flush=True)

    path = args.out_dir / "crossarm_knn.json"
    path.write_text(json.dumps(results, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n-> {path}   ({(time.monotonic() - started) / 60:.1f} phút)")


if __name__ == "__main__":
    main()
