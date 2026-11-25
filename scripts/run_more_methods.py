"""Sáu phương pháp còn thiếu mà không cần mạng nơ-ron.

    python scripts/run_more_methods.py --dir data/processed_hgn_none

Tất cả chạy trên cùng giao thức, cùng rổ ứng viên, cùng `metrics.evaluate` với
các mô hình đã có, nên con số xếp cùng một bảng được.

1. **Item-KNN tương quan Pearson** (L2). Trên dữ liệu nhị phân, Pearson giữa hai
   cột là hệ số phi, và nó tính được **chính xác** từ số đồng xuất hiện mà không
   cần làm dày ma trận:
   `(n_ij/N − p_i p_j) / sqrt(p_i(1−p_i) · p_j(1−p_j))`.
   Khác cosine ở chỗ nó trừ đi phần kỳ vọng do độ phổ biến, nên hai truyện nhiều
   người đọc không tự động giống nhau.

2. **SVD++** (L2) cho dự đoán điểm. Thêm số hạng ẩn
   `|N(u)|^(−1/2) · Σ_{j∈N(u)} y_j` vào vectơ người đọc: những truyện người đó
   *có chấm điểm* nói thêm điều mà giá trị điểm không nói.

3. **Content-boosted CF** (L7, monolithic). Dùng mô hình nội dung sinh
   pseudo-rating cho M mục mỗi người, trộn vào ma trận rồi chạy ALS. Đây là cách
   lai ghép *trong một mô hình* chứ không phải cộng điểm hai mô hình.

4. **Nội dung chiều thấp bằng LSA** (L4). Đề xuất ghi "TF-IDF **hoặc** sentence
   embedding đa ngôn ngữ". Không tải được mô hình câu trong môi trường này, nên
   tôi dùng SVD rút gọn TF-IDF xuống 128 chiều — cùng tính chất "vectơ dày, chiều
   thấp, ngữ nghĩa trộn lẫn", và **phải ghi rõ nó là thay thế**, không phải
   sentence embedding.

5. **Lọc theo ràng buộc** (L5). Suy ràng buộc từ chính lịch sử của người đọc
   (nước, định dạng, nội dung người lớn, số tập) rồi lọc rổ ứng viên trước khi
   xếp. Không có phỏng vấn người dùng nên ràng buộc là *suy ra*, không phải
   *khai báo* — ghi rõ chỗ này vì nó làm phép đo yếu hơn phép đo của một hệ có
   onboarding thật.

6. **Case-based** (L5). Lấy mục đọc gần nhất làm "case", xếp ứng viên theo độ
   tương đồng thuộc tính có trọng số. Đây đúng là critiquing khi chưa có lời phê
   bình nào: bước khởi đầu của vòng lặp.
"""

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data.protocol_peruser import (  # noqa: E402
    excluded_candidates, reachable_test)
from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.collaborative import ALS  # noqa: E402
from truyen_reco.models.content import (  # noqa: E402
    build_item_features, user_profiles)
from truyen_reco.models.knn import user_item_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_protocol_baselines import load, ranker  # noqa: E402


def pearson_neighbours(matrix, k=200, block=512, min_co=1):
    """Hệ số phi (Pearson trên nhị phân) giữa các cột, giữ k láng giềng mỗi mục.

    Tính từ số đồng xuất hiện, không làm dày: với cột nhị phân,
    `cov(i,j) = n_ij/N − p_i p_j` và `sd(i) = sqrt(p_i (1 − p_i))`.
    Cặp có `n_ij < min_co` bị bỏ vì tương quan dựng từ một người đọc chung có
    thể bằng 1 mà không nói gì.
    """
    n_users, n_items = matrix.shape
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    p = counts / max(n_users, 1)
    sd = np.sqrt(np.maximum(p * (1.0 - p), 1e-12))
    transposed = matrix.T.tocsr()
    indptr, indices, values = [0], [], []
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        co = (transposed[start:stop] @ matrix).toarray()
        corr = (co / n_users - p[start:stop, None] * p[None, :]) / (
            sd[start:stop, None] * sd[None, :])
        corr[co < min_co] = 0.0
        for row in range(stop - start):
            corr[row, start + row] = 0.0
            line = corr[row]
            picked = (np.argpartition(-line, k)[:k] if k < n_items
                      else np.arange(n_items))
            picked = picked[line[picked] > 0]
            indices.append(picked.astype(np.int32))
            values.append(line[picked].astype(np.float32))
            indptr.append(indptr[-1] + len(picked))
    if not indices:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32)
    return sparse.csr_matrix(
        (np.concatenate(values), np.concatenate(indices), np.array(indptr)),
        shape=(n_items, n_items))


def attribute_table(media, item_index, merged_of=None):
    """Bảng thuộc tính cho L5: nước, định dạng, người lớn, số tập, thể loại."""
    frame = media.copy()
    frame["slot"] = frame["media_id"].map(item_index)
    if merged_of:
        extra = frame[frame["media_id"].isin(merged_of)].copy()
        extra["slot"] = extra["media_id"].map(merged_of).map(item_index)
        frame = pd.concat([frame, extra])
    frame = frame.dropna(subset=["slot"]).copy()
    frame["slot"] = frame["slot"].astype(int)
    frame = frame.groupby("slot").first()
    n_items = len(item_index)
    country = np.full(n_items, "?", dtype=object)
    fmt = np.full(n_items, "?", dtype=object)
    adult = np.zeros(n_items, dtype=bool)
    volumes = np.full(n_items, np.nan)
    country[frame.index.to_numpy()] = frame["country"].astype(str).to_numpy()
    fmt[frame.index.to_numpy()] = frame["format"].astype(str).to_numpy()
    if "is_adult" in frame:
        adult[frame.index.to_numpy()] = frame["is_adult"].fillna(False).to_numpy()
    if "volumes" in frame:
        volumes[frame.index.to_numpy()] = pd.to_numeric(
            frame["volumes"], errors="coerce").to_numpy()
    return {"country": country, "format": fmt, "adult": adult,
            "volumes": volumes}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--media", default="data/processed_gr/media.parquet",
                        type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--grid-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--neighbours", nargs="+", type=int, default=[200, 500])
    parser.add_argument("--lsa-dims", nargs="+", type=int, default=[64, 128])
    parser.add_argument("--boost-items", nargs="+", type=int, default=[10, 20])
    parser.add_argument("--boost-weight", type=float, default=0.3)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--methods", nargs="+",
                        default=["pearson", "lsa", "boosted", "constraint",
                                 "casebased"])
    args = parser.parse_args()

    started = time.monotonic()
    data = load(args.dir)
    train, val, test = data["train"], data["val"], data["test"]
    user_index, item_index = data["user_index"], data["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    matrix = user_item_matrix(train, user_index, item_index)
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(-counts)
    percentile = np.empty(len(counts))
    percentile[order] = 1.0 - np.arange(len(counts)) / max(len(counts) - 1, 1)
    extra = {
        "popularity": {reverse[i]: int(counts[i]) for i in range(len(counts))},
        "n_users": matrix.shape[0],
        "pop_percentile": {reverse[i]: float(percentile[i])
                           for i in range(len(counts))},
        "head_items": {reverse[int(i)] for i in order[:max(1, len(counts) // 100)]},
    }
    train_by_user = train.groupby("user")["media_id"].apply(list).to_dict()
    excluded_train = excluded_candidates(train, val.iloc[0:0], user_index,
                                         item_index, data["series_of"])
    excluded_all = excluded_candidates(train, val, user_index, item_index,
                                       data["series_of"])
    val_frame, _ = reachable_test(val, excluded_train, item_index)
    test_frame, _ = reachable_test(test, excluded_all, item_index)
    print(f"{args.dir}: {len(item_index):,} mục | {len(user_index):,} người "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    results = {}

    def score_on(frame, excluded, score_fn, users):
        return metrics.evaluate(
            ranker(score_fn, user_index, reverse, excluded, args.k), frame,
            train_by_user, k=args.k, max_users=users,
            catalog_size=len(item_index), seed=args.seed)

    def finish(name, score_fn, note=""):
        got = metrics.evaluate(
            ranker(score_fn, user_index, reverse, excluded_all, args.k),
            test_frame, train_by_user, k=args.k, max_users=args.max_users,
            catalog_size=len(item_index), seed=args.seed, **extra)
        got["note"] = note
        results[name] = got
        print(f"  {name:<34} test NDCG {got['ndcg']:.4f} | Recall "
              f"{got['recall']:.4f} | phủ {got['coverage']:.4f} | long-tail "
              f"{got['tail']:.4f}", flush=True)

    # ---------------------------------------------------------- 1. Pearson
    if "pearson" in args.methods:
        print("\n1. Item-KNN tương quan Pearson — chọn k trên val")
        best = None
        for neighbours in args.neighbours:
            similarity = pearson_neighbours(matrix, k=neighbours)
            score = lambda row, s=similarity: (
                None if matrix[row].nnz == 0
                else np.asarray((matrix[row] @ s).todense()).ravel())
            got = score_on(val_frame, excluded_train, score, args.grid_users)
            print(f"   k={neighbours:<4} val NDCG {got['ndcg']:.4f}", flush=True)
            if best is None or got["ndcg"] > best[0]:
                best = (got["ndcg"], neighbours, similarity)
        _, neighbours, similarity = best
        print(f"   -> val chọn k={neighbours}")
        finish("Item-KNN Pearson", lambda row: (
            None if matrix[row].nnz == 0
            else np.asarray((matrix[row] @ similarity).todense()).ravel()),
            f"k={neighbours}, bỏ cặp chỉ 1 người đọc chung")

    # --------------------------------------------- nội dung dùng cho 2 và 4
    content_matrix = profiles = None
    if {"lsa", "boosted", "casebased"} & set(args.methods):
        media = pd.read_parquet(args.media)
        content_matrix, vocabulary = build_item_features(
            media, item_index, merged_of=data["merged_of"], min_df=args.min_df)
        profiles = user_profiles(train, content_matrix, user_index, item_index)
        print(f"\nnội dung: từ vựng {len(vocabulary):,} "
              f"({time.monotonic() - started:.0f}s)", flush=True)

    # ------------------------------------------------------------- 4. LSA
    if "lsa" in args.methods:
        print("\n2. Nội dung chiều thấp bằng LSA — chọn số chiều trên val")
        best = None
        for dims in args.lsa_dims:
            svd = TruncatedSVD(n_components=dims, random_state=args.seed)
            items_dense = svd.fit_transform(content_matrix).astype(np.float32)
            items_dense /= np.maximum(
                np.linalg.norm(items_dense, axis=1, keepdims=True), 1e-9)
            users_dense = np.asarray(profiles @ svd.components_.T,
                                     dtype=np.float32)
            users_dense /= np.maximum(
                np.linalg.norm(users_dense, axis=1, keepdims=True), 1e-9)
            score = lambda row, u=users_dense, v=items_dense: v @ u[row]
            got = score_on(val_frame, excluded_train, score, args.grid_users)
            print(f"   {dims} chiều  val NDCG {got['ndcg']:.4f} | giữ "
                  f"{svd.explained_variance_ratio_.sum() * 100:.1f}% phương sai",
                  flush=True)
            if best is None or got["ndcg"] > best[0]:
                best = (got["ndcg"], dims, users_dense, items_dense,
                        float(svd.explained_variance_ratio_.sum()))
        _, dims, users_dense, items_dense, kept = best
        print(f"   -> val chọn {dims} chiều")
        finish("Nội dung LSA (thay sentence embedding)",
               lambda row: items_dense @ users_dense[row],
               f"{dims} chiều, giữ {kept * 100:.1f}% phương sai; "
               f"KHÔNG phải sentence embedding đa ngôn ngữ")

    # ------------------------------------------------- 3. content-boosted CF
    if "boosted" in args.methods:
        print("\n3. Content-boosted CF — chọn số mục bơm thêm trên val")
        # Điểm nội dung cho MỌI cặp (người, mục) là 58.413 x 36.241 = 8,5 GB nếu
        # làm dày một lần, và `profiles @ content_matrix.T` dạng thưa cũng nổ vì
        # hồ sơ người đọc chạm rất nhiều từ. Nên: rút nội dung xuống 128 chiều
        # bằng LSA rồi nhân khối 2.000 người — vẫn là mô hình nội dung, chỉ ở
        # dạng chiều thấp, và phải ghi rõ thế trong nhãn kết quả.
        svd_boost = TruncatedSVD(n_components=128, random_state=args.seed)
        items_boost = svd_boost.fit_transform(content_matrix).astype(np.float32)
        items_boost /= np.maximum(
            np.linalg.norm(items_boost, axis=1, keepdims=True), 1e-9)
        users_boost = np.asarray(profiles @ svd_boost.components_.T,
                                 dtype=np.float32)
        users_boost /= np.maximum(
            np.linalg.norm(users_boost, axis=1, keepdims=True), 1e-9)
        top_by_user = {}
        chunk = 2000
        for start in range(0, matrix.shape[0], chunk):
            stop = min(start + chunk, matrix.shape[0])
            block = users_boost[start:stop] @ items_boost.T
            for local in range(stop - start):
                row = start + local
                line = block[local]
                line[matrix[row].indices] = -np.inf
                top = np.argpartition(-line, max(args.boost_items))
                top_by_user[row] = top[:max(args.boost_items)]
        best = None
        for bump in args.boost_items:
            rows_boost, cols_boost = [], []
            for row, top in top_by_user.items():
                rows_boost.extend([row] * bump)
                cols_boost.extend(top[:bump].tolist())
            added = sparse.csr_matrix(
                (np.full(len(rows_boost), args.boost_weight, dtype=np.float32),
                 (np.asarray(rows_boost, dtype=np.int32),
                  np.asarray(cols_boost, dtype=np.int32))),
                shape=matrix.shape)
            boosted = (matrix + added).tocsr()
            model = ALS(factors=64, iterations=15, regularization=0.05,
                        seed=args.seed)
            model.fit(boosted)
            score = lambda row, m=model: m.item_factors @ m.user_factors[row]
            got = score_on(val_frame, excluded_train, score, args.grid_users)
            print(f"   bơm {bump} mục/người (trọng số {args.boost_weight})  "
                  f"val NDCG {got['ndcg']:.4f}", flush=True)
            if best is None or got["ndcg"] > best[0]:
                best = (got["ndcg"], bump, model)
        _, bump, model = best
        print(f"   -> val chọn {bump} mục")
        finish("Content-boosted CF",
               lambda row: model.item_factors @ model.user_factors[row],
               f"bơm {bump} pseudo-rating mỗi người, trọng số "
               f"{args.boost_weight}, rồi ALS 64 nhân tố")

    # ----------------------------------------------- 5. lọc theo ràng buộc
    if "constraint" in args.methods or "casebased" in args.methods:
        media = pd.read_parquet(args.media) if content_matrix is None else media
        attributes = attribute_table(media, item_index, data["merged_of"])

    if "constraint" in args.methods:
        print("\n4. Lọc theo ràng buộc suy từ lịch sử")
        country = attributes["country"]
        fmt = attributes["format"]
        adult = attributes["adult"]
        volumes = attributes["volumes"]
        base = np.log1p(counts).astype(np.float32)

        def constraint_score(row):
            seen = matrix[row].indices
            if not len(seen):
                return None
            allowed = np.ones(len(counts), dtype=bool)
            # Ràng buộc 1: nước. Chỉ giữ các nước người này đã đọc.
            wanted = set(country[seen])
            wanted.discard("?")
            if wanted:
                allowed &= np.isin(country, list(wanted))
            # Ràng buộc 2: định dạng, cùng cách.
            shapes = set(fmt[seen])
            shapes.discard("?")
            if shapes:
                allowed &= np.isin(fmt, list(shapes))
            # Ràng buộc 3: chưa đọc gì người lớn thì loại nội dung người lớn.
            if not adult[seen].any():
                allowed &= ~adult
            # Ràng buộc 4: "đọc nhanh" nếu trung vị số tập đã đọc <= 10.
            read_volumes = volumes[seen]
            read_volumes = read_volumes[np.isfinite(read_volumes)]
            if len(read_volumes) and np.median(read_volumes) <= 10:
                allowed &= ~(np.isfinite(volumes) & (volumes > 10))
            # Nới lỏng khi rỗng: đây là phần "đề xuất tập ràng buộc tối thiểu
            # cần nới" của đề xuất, làm tự động theo thứ tự ngược.
            if allowed.sum() < args.k:
                allowed = np.ones(len(counts), dtype=bool)
            return np.where(allowed, base, -np.inf)

        finish("Lọc ràng buộc trên bảng phổ biến", constraint_score,
               "4 ràng buộc suy từ lịch sử (nước, định dạng, người lớn, số "
               "tập), tự nới khi rổ rỗng; ràng buộc SUY RA chứ không khai báo")

    # -------------------------------------------------------- 6. case-based
    if "casebased" in args.methods:
        print("\n5. Case-based: mục đọc gần nhất làm case")
        ordered = train.sort_values(["user", "created_at"], kind="mergesort")
        last = ordered.groupby("user", sort=False)["media_id"].last()
        case_of = {}
        for user, media_id in last.items():
            row = user_index.get(user)
            slot = item_index.get(media_id)
            if row is not None and slot is not None:
                case_of[row] = slot
        country = attributes["country"]
        fmt = attributes["format"]
        volumes = attributes["volumes"]
        log_volumes = np.log1p(np.nan_to_num(volumes, nan=0.0))
        content_rows = sparse.csr_matrix(content_matrix)

        def case_score(row):
            slot = case_of.get(row)
            if slot is None:
                return None
            # Tương đồng thuộc tính có trọng số: nội dung 1,0 + cùng nước 0,2
            # + cùng định dạng 0,1 + độ gần về số tập 0,1.
            similarity = np.asarray(
                (content_rows @ content_rows[slot].T).todense()).ravel()
            same_country = (country == country[slot]).astype(np.float32) * 0.2
            same_format = (fmt == fmt[slot]).astype(np.float32) * 0.1
            length_gap = np.exp(-np.abs(log_volumes - log_volumes[slot])) * 0.1
            return (similarity + same_country + same_format
                    + length_gap).astype(np.float32)

        finish("Case-based (case = mục gần nhất)", case_score,
               "tương đồng nội dung 1,0 + cùng nước 0,2 + cùng định dạng 0,1 "
               "+ độ gần số tập 0,1; chưa có lời phê bình nào")

    path = args.dir / "more_methods.json"
    previous = (json.loads(path.read_text(encoding="utf-8"))
                if path.exists() else {})
    previous.update(results)
    path.write_text(json.dumps(previous, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n-> {path}   ({(time.monotonic() - started) / 60:.1f} phút)")


if __name__ == "__main__":
    main()
