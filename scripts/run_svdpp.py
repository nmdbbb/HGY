"""SVD++ trên bài dự đoán điểm — dòng cuối cùng còn thiếu của bậc L2.

    python scripts/run_svdpp.py --out-name svdpp.json

Dùng **đúng** phép chia và đúng bảng chỉ số của `run_ladder_gaps.py --part a`,
bằng cách gọi lại chính các hàm của nó, nên con số đặt thẳng cạnh năm dòng kia
được: trung bình toàn cục, trung bình theo truyện, trung bình theo người, chỉ
chệch, MF có chệch.

Công thức khác MF có chệch đúng một số hạng:

    r̂(u,i) = μ + b_u + b_i + q_i · ( p_u + |N(u)|^(−1/2) Σ_{j∈N(u)} y_j )

`N(u)` là tập truyện người đó **có chấm điểm** trong train. Số hạng thêm nói:
*việc bạn chấm truyện nào* đã là một tín hiệu, tách khỏi *bạn chấm mấy sao*.

Chỗ phải nghĩ là cách cập nhật `y`. Cài theo sách — mỗi rating cập nhật mọi
`y_j` với j ∈ N(u) — là 4,1 triệu rating x trung bình 41 mục = 169 triệu phép
cộng hàng mỗi epoch, không chạy nổi bằng numpy. Nhưng gradient trên `y_j` giống
hệt nhau với mọi j trong cùng N(u):

    ∂L/∂y_j = |N(u)|^(−1/2) · Σ_{(u,i) trong lô} err(u,i) · q_i    (với mọi j∈N(u))

nên gom theo người trước rồi rải một lần bằng **một phép nhân ma trận thưa**
`ΔY = Bᵀ G` với B là ma trận nhị phân người × truyện. Cùng phép cập nhật, mà
mỗi lô chỉ còn vài phép trên mảng. Đây là biến đổi đại số, không phải xấp xỉ.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data.protocol_peruser import split_per_user  # noqa: E402
from run_ladder_gaps import k_core_ratings, rating_errors  # noqa: E402


def implicit_matrix(train, n_users, n_items):
    """B nhị phân (người × truyện) cho N(u), kèm |N(u)|^(−1/2)."""
    rows = train["row"].to_numpy(np.int32)
    cols = train["col"].to_numpy(np.int32)
    matrix = sparse.csr_matrix(
        (np.ones(len(rows), np.float32), (rows, cols)),
        shape=(n_users, n_items))
    matrix.data[:] = 1.0
    sizes = np.asarray(matrix.sum(axis=1)).ravel().astype(np.float32)
    scale = np.where(sizes > 0, 1.0 / np.sqrt(np.maximum(sizes, 1.0)), 0.0)
    return matrix.astype(np.float32), scale.astype(np.float32)


def fit_svdpp(train, n_users, n_items, factors=32, epochs=15, lr=0.02,
              reg=0.05, batch=50_000, seed=0, verbose=False):
    rng = np.random.default_rng(seed)
    users = train["row"].to_numpy(np.int32)
    items = train["col"].to_numpy(np.int32)
    ratings = train["rating"].to_numpy(np.float32)
    mu = float(ratings.mean())
    bu = np.zeros(n_users, np.float32)
    bi = np.zeros(n_items, np.float32)
    P = rng.normal(0, 0.05, (n_users, factors)).astype(np.float32)
    Q = rng.normal(0, 0.05, (n_items, factors)).astype(np.float32)
    Y = np.zeros((n_items, factors), np.float32)
    B, scale = implicit_matrix(train, n_users, n_items)
    S = np.zeros((n_users, factors), np.float32)  # Σ_{j∈N(u)} y_j

    order = np.arange(len(ratings))
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            u, i, r = users[idx], items[idx], ratings[idx]
            implicit = scale[u][:, None] * S[u]
            pu = P[u] + implicit
            qi = Q[i]
            err = r - (mu + bu[u] + bi[i] + np.einsum("ij,ij->i", pu, qi))
            # Cùng lý do như trong `fit_biased_mf`: `np.add.at` cộng TỔNG các
            # cập nhật, nên phải chia cho số lần xuất hiện trong lô, nếu không
            # người chấm nhiều sẽ nhận một bước lớn gấp số rating của họ.
            scale_u = 1.0 / np.maximum(np.bincount(u, minlength=n_users)[u], 1)
            scale_i = 1.0 / np.maximum(np.bincount(i, minlength=n_items)[i], 1)
            np.add.at(bu, u, lr * scale_u * (err - reg * bu[u]))
            np.add.at(bi, i, lr * scale_i * (err - reg * bi[i]))
            np.add.at(P, u, lr * scale_u[:, None]
                      * (err[:, None] * qi - reg * P[u]))
            np.add.at(Q, i, lr * scale_i[:, None]
                      * (err[:, None] * pu - reg * qi))
            # Gom gradient theo NGƯỜI rồi rải sang y bằng một phép Bᵀ G.
            grad = np.zeros((n_users, factors), np.float32)
            np.add.at(grad, u, scale_u[:, None] * scale[u][:, None]
                      * err[:, None] * qi)
            touched = np.unique(u)
            delta = B[touched].T @ grad[touched]
            # Hệ số co ngót áp **một lần mỗi lô**, đúng quy ước mà
            # `fit_biased_mf` dùng cho P và Q. Giữ cùng quy ước là điều kiện
            # để hai dòng kết quả so được với nhau: khi đó SVD++ khác MF có
            # chệch đúng một thứ — số hạng ẩn — chứ không khác cả cách phạt.
            Y *= (1.0 - lr * reg)
            Y += lr * np.asarray(delta, dtype=np.float32)
            S = B @ Y
        if not np.isfinite(P).all() or not np.isfinite(Y).all():
            raise FloatingPointError(
                f"SVD++ phân kỳ ở epoch {epoch + 1} (f={factors}, reg={reg})")
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}", flush=True)
    return mu, bu, bi, P + scale[:, None] * S, Q


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--interactions",
                        default="data/processed_gr/interactions.parquet",
                        type=Path)
    parser.add_argument("--out", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--out-name", default="svdpp.json")
    parser.add_argument("--factors", type=int, nargs="+", default=[32, 64])
    parser.add_argument("--regs", type=float, nargs="+", default=[0.02, 0.05])
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--min-user-rating", type=int, default=5)
    parser.add_argument("--min-item-rating", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    started = time.monotonic()
    raw = pd.read_parquet(args.interactions,
                          columns=["user", "media_id", "score", "created_at"])
    rated = raw[raw["score"] > 0].copy()
    rated["rating"] = (rated["score"] / 20.0).astype(np.float32)
    rated = k_core_ratings(rated, args.min_user_rating, args.min_item_rating)
    train, val, test, _ = split_per_user(rated, min_items=args.min_user_rating)
    users = {u: n for n, u in enumerate(train["user"].unique())}
    items = {m: n for n, m in enumerate(train["media_id"].unique())}
    frames = {}
    for name, frame in (("train", train), ("val", val), ("test", test)):
        frame = frame.copy()
        frame["row"] = frame["user"].map(users)
        frame["col"] = frame["media_id"].map(items)
        frames[name] = frame.dropna(subset=["row", "col"]).astype(
            {"row": np.int32, "col": np.int32})
    print(f"  {len(rated):,} dòng có điểm | {len(users):,} người | "
          f"{len(items):,} truyện | train/val/test "
          f"{len(frames['train']):,}/{len(frames['val']):,}/"
          f"{len(frames['test']):,}  ({time.monotonic() - started:.0f}s)",
          flush=True)

    best = None
    for factors in args.factors:
        for reg in args.regs:
            clock = time.monotonic()
            fitted = fit_svdpp(frames["train"], len(users), len(items),
                               factors=factors, epochs=args.epochs, reg=reg,
                               seed=args.seed)
            mae, rmse = rating_errors(frames["val"], *fitted)
            print(f"  SVD++ f={factors:<3} reg={reg:<5} val MAE {mae:.4f} "
                  f"RMSE {rmse:.4f}  ({time.monotonic() - clock:.0f}s)",
                  flush=True)
            if best is None or rmse < best[0]:
                best = (rmse, factors, reg, fitted)
    _, factors, reg, fitted = best
    mae, rmse = rating_errors(frames["test"], *fitted)
    print(f"  -> val chọn f={factors}, reg={reg} | test MAE {mae:.4f} "
          f"RMSE {rmse:.4f}")

    path = args.out / args.out_name
    path.write_text(json.dumps(
        {"SVD++": {"mae": mae, "rmse": rmse, "factors": factors, "reg": reg,
                   "rows_rated": int(len(rated)), "users": len(users),
                   "items": len(items), "epochs": args.epochs}},
        ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  -> {path}")


if __name__ == "__main__":
    main()
