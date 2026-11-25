"""Bù ba chỗ trống của thang L2-L10 mà hai nhánh hiện có chưa chạm tới.

    python scripts/run_ladder_gaps.py --dir data/processed_hgn_none

Ba phần, mỗi phần lấp một ô trong bảng "dữ liệu x phương pháp":

**A. L2 - phân rã ma trận cho rating hiện (MAE/RMSE).** Mục 6 của đề xuất đòi
MAE/RMSE, và cả hai nhánh hiện có **không có một con số nào**. Lý do không phải
quên: nhánh giao thức lọc `rating >= 4` nên cột `score` trong `train.parquet`
chỉ còn hai giá trị (80 và 100) — bài toán dự đoán điểm trên đó là hình thức.
Hai đòi hỏi của đề xuất (lọc rating >= 4 **và** báo MAE/RMSE) xung đột nhau,
nên MAE/RMSE phải đo trên một tiền xử lý **thứ ba**: toàn bộ dòng *có* chấm
điểm, 1-5 sao, không lọc ngưỡng.

**B. L6 - luật kết hợp.** `{a} -> {b}` với support và confidence, dùng cho
"người đọc truyện này cũng đọc". Khác Item-KNN ở chỗ confidence **không đối
xứng**: `co(a,b)/count(a)`, nên nó trả lời "đọc a rồi thì bao nhiêu phần trăm
đọc b", một câu giải thích được cho người dùng.

**C. L7 - lai ghép song song có trọng số + đa dạng hoá MMR.** Ba nguồn ứng viên
(Session-KNN, Item-KNN, nội dung) chuẩn hoá z-score theo từng người rồi cộng có
trọng số; trọng số tìm bằng grid search **trên validation**, test đọc đúng một
lần. MMR xếp lại top-50 với lambda cũng chọn trên validation.

Chỗ phải cẩn thận ở phần C: ba nguồn có thang điểm khác nhau hoàn toàn (cosine
nội dung trong [0,1], tổng tương đồng Item-KNN có thể hàng chục), nên cộng thô
là để một nguồn lấn hết. Chuẩn hoá z-score **theo từng người** trên đúng vectơ
điểm của người đó, không phải theo toàn cục.
"""

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import sparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data.protocol_peruser import (  # noqa: E402
    excluded_candidates, reachable_test, split_per_user)
from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models.content import (  # noqa: E402
    ContentModel, build_item_features, user_profiles)
from truyen_reco.models.knn import (  # noqa: E402
    top_k_neighbours, user_item_matrix)

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_protocol_baselines import load, ranker  # noqa: E402


# ----------------------------------------------------------------- A. MAE/RMSE

def k_core_ratings(frame, min_user=5, min_item=5, rounds=6):
    """Lọc lõi lặp lại cho tới khi ổn định (hoặc hết `rounds`)."""
    for _ in range(rounds):
        before = len(frame)
        counts = frame["media_id"].value_counts()
        frame = frame[frame["media_id"].isin(counts.index[counts >= min_item])]
        counts = frame["user"].value_counts()
        frame = frame[frame["user"].isin(counts.index[counts >= min_user])]
        if len(frame) == before:
            break
    return frame


def fit_biased_mf(train, n_users, n_items, factors=32, epochs=15, lr=0.02,
                  reg=0.05, batch=50_000, seed=0, verbose=False):
    """MF có chệch (FunkSVD + bias), SGD theo lô, vectơ hoá bằng `np.add.at`.

    Vòng lặp từng mẫu bằng Python cho 2,5 triệu rating x 15 epoch là 37 triệu
    lần lặp — không chạy nổi. SGD theo lô với `np.add.at` cho cùng phép cập nhật
    mà chỉ còn vài trăm phép trên mảng mỗi epoch.
    """
    rng = np.random.default_rng(seed)
    users = train["row"].to_numpy(np.int32)
    items = train["col"].to_numpy(np.int32)
    ratings = train["rating"].to_numpy(np.float32)
    mu = float(ratings.mean())
    bu = np.zeros(n_users, np.float32)
    bi = np.zeros(n_items, np.float32)
    P = (rng.normal(0, 0.05, (n_users, factors))).astype(np.float32)
    Q = (rng.normal(0, 0.05, (n_items, factors))).astype(np.float32)
    order = np.arange(len(ratings))
    for epoch in range(epochs):
        rng.shuffle(order)
        for start in range(0, len(order), batch):
            idx = order[start:start + batch]
            u, i, r = users[idx], items[idx], ratings[idx]
            pu, qi = P[u], Q[i]
            err = r - (mu + bu[u] + bi[i] + np.einsum("ij,ij->i", pu, qi))
            # Chia theo số lần xuất hiện trong lô. Bản đầu không có dòng này và
            # nó **phân kỳ ra nan ngay epoch đầu**: `np.add.at` cộng *tổng* các
            # cập nhật, nên một người có 40 rating trong cùng một lô nhận một
            # bước lớn gấp 40 lần, vectơ nổ, rồi nan lan ra toàn mô hình. Chia
            # ra thì mỗi người và mỗi truyện nhận một bước *trung bình* — đúng
            # tinh thần SGD, và nó chặn đúng chỗ hay nổ: người đọc nhiều và
            # truyện phổ biến.
            scale_u = 1.0 / np.maximum(np.bincount(u, minlength=n_users)[u], 1)
            scale_i = 1.0 / np.maximum(np.bincount(i, minlength=n_items)[i], 1)
            np.add.at(bu, u, lr * scale_u * (err - reg * bu[u]))
            np.add.at(bi, i, lr * scale_i * (err - reg * bi[i]))
            np.add.at(P, u, lr * scale_u[:, None] * (err[:, None] * qi - reg * pu))
            np.add.at(Q, i, lr * scale_i[:, None] * (err[:, None] * pu - reg * qi))
        if not np.isfinite(P).all() or not np.isfinite(Q).all():
            # Báo thẳng thay vì trả về nan: lần đầu viết hàm này nó phân kỳ và
            # grid search vẫn chạy hết, in "val MAE nan" cho cả sáu cấu hình rồi
            # "chọn" cấu hình đầu. Một mô hình phân kỳ phải là lỗi, không phải
            # một dòng kết quả.
            raise FloatingPointError(
                f"MF phân kỳ ở epoch {epoch + 1} (f={factors}, reg={reg}, "
                f"lr={lr}): giảm lr hoặc tăng reg")
        if verbose:
            print(f"    epoch {epoch + 1}/{epochs}", flush=True)
    return mu, bu, bi, P, Q


def rating_errors(frame, mu, bu, bi, P=None, Q=None):
    u = frame["row"].to_numpy(np.int32)
    i = frame["col"].to_numpy(np.int32)
    pred = mu + bu[u] + bi[i]
    if P is not None:
        pred = pred + np.einsum("ij,ij->i", P[u], Q[i])
    pred = np.clip(pred, 1.0, 5.0)
    err = frame["rating"].to_numpy(np.float32) - pred
    return float(np.abs(err).mean()), float(np.sqrt((err ** 2).mean()))


def part_a(args):
    print("=" * 78)
    print("A. L2 — dự đoán điểm (MAE/RMSE) trên TOÀN BỘ dòng có chấm điểm")
    print("=" * 78)
    raw = pd.read_parquet(args.interactions,
                          columns=["user", "media_id", "score", "created_at"])
    rated = raw[raw["score"] > 0].copy()
    rated["rating"] = (rated["score"] / 20.0).astype(np.float32)
    print(f"  {len(raw):,} dòng thô -> {len(rated):,} dòng có điểm "
          f"({100 * len(rated) / len(raw):.1f}%)")
    rated = k_core_ratings(rated, args.min_user_rating, args.min_item_rating)
    print(f"  sau lọc lõi {args.min_user_rating}/{args.min_item_rating}: "
          f"{len(rated):,} dòng | {rated['user'].nunique():,} người | "
          f"{rated['media_id'].nunique():,} mục")
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
    print(f"  train/val/test {len(frames['train']):,} / {len(frames['val']):,}"
          f" / {len(frames['test']):,}")

    out = {}
    mu = float(frames["train"]["rating"].mean())
    zeros_u = np.zeros(len(users), np.float32)
    zeros_i = np.zeros(len(items), np.float32)
    out["trung bình toàn cục"] = rating_errors(frames["test"], mu, zeros_u,
                                               zeros_i)
    # Trung bình theo người / theo mục, viết dưới dạng chệch để dùng lại hàm trên.
    by_user = frames["train"].groupby("row")["rating"].mean() - mu
    bu = zeros_u.copy()
    bu[by_user.index.to_numpy()] = by_user.to_numpy(np.float32)
    out["trung bình theo người"] = rating_errors(frames["test"], mu, bu, zeros_i)
    by_item = frames["train"].groupby("col")["rating"].mean() - mu
    bi = zeros_i.copy()
    bi[by_item.index.to_numpy()] = by_item.to_numpy(np.float32)
    out["trung bình theo truyện"] = rating_errors(frames["test"], mu, zeros_u, bi)

    best = None
    for factors, reg in itertools.product(args.factors, args.regs):
        started = time.monotonic()
        mu_f, bu_f, bi_f, P, Q = fit_biased_mf(
            frames["train"], len(users), len(items), factors=factors,
            epochs=args.mf_epochs, reg=reg, seed=args.seed)
        mae, rmse = rating_errors(frames["val"], mu_f, bu_f, bi_f, P, Q)
        print(f"  MF f={factors:<3} reg={reg:<5} val MAE {mae:.4f} RMSE {rmse:.4f}"
              f"  ({time.monotonic() - started:.0f}s)", flush=True)
        if best is None or rmse < best[0]:
            best = (rmse, factors, reg, (mu_f, bu_f, bi_f, P, Q))
    _, factors, reg, fitted = best
    out["chỉ chệch (mu + bu + bi)"] = rating_errors(
        frames["test"], fitted[0], fitted[1], fitted[2])
    out[f"MF có chệch (f={factors}, reg={reg})"] = rating_errors(
        frames["test"], *fitted)
    print(f"  -> val chọn f={factors}, reg={reg}")
    print(f"\n  {'mô hình':<34}{'test MAE':>10}{'test RMSE':>11}")
    for name, (mae, rmse) in out.items():
        print(f"  {name:<34}{mae:>10.4f}{rmse:>11.4f}")
    return {"rows_rated": int(len(rated)), "users": len(users),
            "items": len(items), "chosen": {"factors": factors, "reg": reg},
            "results": {k: {"mae": v[0], "rmse": v[1]} for k, v in out.items()}}


# ------------------------------------------------------------ B. luật kết hợp

def confidence_rules(matrix, min_co=5, block=512):
    """Ma trận confidence `co(a,b) / count(a)`, chỉ giữ cặp có `co >= min_co`.

    Ngưỡng `min_co` không phải để cho nhanh mà vì **ý nghĩa**: một luật dựng từ
    2 người đọc chung có confidence có thể bằng 1,0 và chẳng nói gì. Nó cũng là
    thứ giữ ma trận đủ nhỏ để nằm trong bộ nhớ — phép đo ở mục 14
    `docs/protocol-arm.md` cho thấy 90% cặp đồng xuất hiện có `co < 5`.
    """
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    n_items = matrix.shape[1]
    transposed = matrix.T.tocsr()
    rows, cols, values = [], [], []
    for start in range(0, n_items, block):
        stop = min(start + block, n_items)
        co = (transposed[start:stop] @ matrix).toarray()
        for local in range(stop - start):
            antecedent = start + local
            co[local, antecedent] = 0.0
            keep = np.flatnonzero(co[local] >= min_co)
            if not len(keep) or counts[antecedent] == 0:
                continue
            rows.append(np.full(len(keep), antecedent, dtype=np.int32))
            cols.append(keep.astype(np.int32))
            values.append((co[local, keep] / counts[antecedent]).astype(np.float32))
    if not rows:
        return sparse.csr_matrix((n_items, n_items), dtype=np.float32), 0
    rules = sparse.csr_matrix(
        (np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
        shape=(n_items, n_items))
    return rules, int(rules.nnz)


def part_b(args, data, matrix, train_by_user, excluded, frames, reverse, sizes):
    print("\n" + "=" * 78)
    print("B. L6 — luật kết hợp {a} -> {b} với support và confidence")
    print("=" * 78)
    started = time.monotonic()
    rules, n_rules = confidence_rules(matrix, min_co=args.min_co)
    n_users = matrix.shape[0]
    print(f"  {n_rules:,} luật có co >= {args.min_co} "
          f"({time.monotonic() - started:.0f}s) | support tối thiểu "
          f"{args.min_co / n_users:.2e}")

    user_index, item_index = data["user_index"], data["item_index"]
    out = {}
    for mode in ("max", "sum"):
        def score(row, mode=mode):
            positions = matrix[row].indices
            if not len(positions):
                return None
            block = rules[positions]
            if block.nnz == 0:
                return np.zeros(matrix.shape[1], np.float32)
            reduced = block.max(axis=0) if mode == "max" else block.sum(axis=0)
            # `max` trả về sparse, `sum` trả về np.matrix — hai kiểu khác nhau
            # cho cùng một phép rút gọn, nên phải quy về một kiểu ở đây.
            reduced = (reduced.toarray() if sparse.issparse(reduced)
                       else np.asarray(reduced))
            return reduced.ravel().astype(np.float32)

        got = metrics.evaluate(
            ranker(score, user_index, reverse, excluded, args.k),
            frames["test"], train_by_user, k=args.k, max_users=args.max_users,
            catalog_size=len(item_index), seed=args.seed,
            popularity=sizes["popularity"], n_users=n_users,
            pop_percentile=sizes["pop_percentile"],
            head_items=sizes["head_items"])
        out[f"luật kết hợp ({mode} confidence)"] = got
        print(f"  {mode:<4} NDCG {got['ndcg']:.4f} | Recall {got['recall']:.4f}"
              f" | HR {got['hit_rate']:.4f} | phủ {got['coverage']:.4f}"
              f" | long-tail {got['tail']:.4f}", flush=True)
    return {"n_rules": n_rules, "min_co": args.min_co,
            "results": {k: {m: v[m] for m in ("ndcg", "recall", "hit_rate",
                                              "coverage", "tail", "arp")}
                        for k, v in out.items()}}


# ------------------------------------------------------------------- C. lai ghép

def zscore(values):
    mean = float(values.mean())
    spread = float(values.std())
    return (values - mean) / spread if spread > 0 else values - mean


def part_c(args, data, matrix, train_by_user, excluded_train, excluded_all,
           frames, reverse, sizes, item_similarity, media_path):
    print("\n" + "=" * 78)
    print("C. L7 — lai ghép song song có trọng số, trọng số chọn trên VAL")
    print("=" * 78)
    user_index, item_index = data["user_index"], data["item_index"]
    train = data["train"]
    media = pd.read_parquet(media_path)
    content_matrix, vocabulary = build_item_features(
        media, item_index, merged_of=data["merged_of"], min_df=args.min_df)
    profiles = user_profiles(train, content_matrix, user_index, item_index)
    content = ContentModel(content_matrix, profiles)
    print(f"  nội dung: từ vựng {len(vocabulary):,}")

    ordered = train.sort_values(["user", "created_at"], kind="mergesort")
    tail = ordered.groupby("user", sort=False).tail(args.session_length)
    session = {}
    for user, group in tail.groupby("user", sort=False)["media_id"]:
        row = user_index.get(user)
        if row is None:
            continue
        positions = [item_index[m] for m in group if m in item_index]
        if positions:
            session[row] = np.asarray(positions, dtype=np.int32)

    def components(row):
        profile = matrix[row]
        if profile.nnz == 0:
            return None
        item_scores = np.asarray((profile @ item_similarity).todense()).ravel()
        positions = session.get(row)
        if positions is None:
            session_scores = item_scores
        else:
            weights = np.power(args.decay, np.arange(len(positions) - 1, -1, -1))
            vector = sparse.csr_matrix(
                (weights.astype(np.float32),
                 (np.zeros(len(positions), np.int32), positions)),
                shape=(1, matrix.shape[1]))
            session_scores = np.asarray(
                (vector @ item_similarity).todense()).ravel()
        content_scores = content.scores_for(row)
        if content_scores is None:
            content_scores = np.zeros(matrix.shape[1], np.float32)
        return (zscore(session_scores.astype(np.float32)),
                zscore(item_scores.astype(np.float32)),
                zscore(np.asarray(content_scores, np.float32)))

    def blended(weights):
        def score(row):
            parts = components(row)
            if parts is None:
                return None
            return sum(w * p for w, p in zip(weights, parts))
        return score

    grid = [(a / 4, b / 4, c / 4) for a in range(5) for b in range(5)
            for c in range(5) if a + b + c == 4]
    best = None
    for weights in grid:
        got = metrics.evaluate(
            ranker(blended(weights), user_index, reverse, excluded_train,
                   args.k),
            frames["val"], train_by_user, k=args.k,
            max_users=args.grid_users, catalog_size=len(item_index),
            seed=args.seed)
        print(f"  w(session,item,content)={weights}  val NDCG {got['ndcg']:.4f}",
              flush=True)
        if best is None or got["ndcg"] > best[0]:
            best = (got["ndcg"], weights)
    weights = best[1]
    print(f"  -> val chọn {weights}")

    results = {}
    got = metrics.evaluate(
        ranker(blended(weights), user_index, reverse, excluded_all, args.k),
        frames["test"], train_by_user, k=args.k, max_users=args.max_users,
        catalog_size=len(item_index), seed=args.seed,
        popularity=sizes["popularity"], n_users=matrix.shape[0],
        pop_percentile=sizes["pop_percentile"], head_items=sizes["head_items"])
    results[f"lai ghép có trọng số {weights}"] = got
    print(f"  test NDCG {got['ndcg']:.4f} | phủ {got['coverage']:.4f}"
          f" | long-tail {got['tail']:.4f}")

    # MMR trên top-50 của chính hệ lai: lambda = 1 là không đa dạng hoá.
    normalized = sparse.csr_matrix(content_matrix)

    def mmr_recommender(weights, lam, excluded, pool=50):
        # `excluded` phải là tham số, không được viết cứng `excluded_all`. Bản đầu
        # viết cứng và hậu quả đúng như cảnh báo trong `tune_protocol.py`: lúc
        # chấm trên val, các mục val chính là đáp án, nên loại chúng làm đáp án
        # rỗng và NDCG ra **0,0000 ở cả bốn lambda** — một lỗi im lặng trông
        # giống "MMR không hoạt động".
        base = blended(weights)

        def recommend(user, _history):
            row = user_index.get(user)
            if row is None:
                return []
            scores = base(row)
            if scores is None:
                return []
            scores = scores.astype(np.float32, copy=True)
            drop = excluded.get(user)
            if drop is not None and len(drop):
                scores[drop] = -np.inf
            top = np.argpartition(-scores, pool)[:pool]
            top = top[np.argsort(-scores[top])]
            top = [int(i) for i in top if np.isfinite(scores[int(i)])]
            if not top:
                return []
            vectors = normalized[top]
            similarity = (vectors @ vectors.T).toarray()
            relevance = scores[top]
            relevance = ((relevance - relevance.min()) /
                         max(relevance.max() - relevance.min(), 1e-9))
            chosen, remaining = [0], list(range(1, len(top)))
            while len(chosen) < args.k and remaining:
                penalty = similarity[np.ix_(remaining, chosen)].max(axis=1)
                value = lam * relevance[remaining] - (1 - lam) * penalty
                pick = remaining[int(np.argmax(value))]
                chosen.append(pick)
                remaining.remove(pick)
            return [reverse[top[i]] for i in chosen[:args.k]]
        return recommend

    best_lam = None
    for lam in args.lambdas:
        got = metrics.evaluate(mmr_recommender(weights, lam, excluded_train),
                               frames["val"],
                               train_by_user, k=args.k,
                               max_users=args.grid_users,
                               catalog_size=len(item_index), seed=args.seed)
        print(f"  MMR lambda={lam:<4} val NDCG {got['ndcg']:.4f}", flush=True)
        if best_lam is None or got["ndcg"] > best_lam[0]:
            best_lam = (got["ndcg"], lam)
    lam = best_lam[1]
    # Không truyền `facet_vectors`: `intra_list_diversity` cần mảng DÀY
    # (`np.linalg.norm(block, axis=1)`), còn ma trận nội dung ở đây là thưa
    # 36.241 x 38.256 — làm dày là không thể, và truyền thưa vào thì scipy báo
    # "does not support dtype object" ở một chỗ cách đó ba tầng hàm. Câu chuyện
    # đa dạng của MMR đọc bằng độ phủ và tỉ lệ long-tail, hai chỉ số có sẵn.
    got = metrics.evaluate(
        mmr_recommender(weights, lam, excluded_all), frames["test"],
        train_by_user,
        k=args.k, max_users=args.max_users, catalog_size=len(item_index),
        seed=args.seed, popularity=sizes["popularity"],
        n_users=matrix.shape[0], pop_percentile=sizes["pop_percentile"],
        head_items=sizes["head_items"])
    results[f"lai ghép + MMR (lambda={lam})"] = got
    print(f"  -> val chọn lambda={lam} | test NDCG {got['ndcg']:.4f}"
          f" | phủ {got['coverage']:.4f} | long-tail {got['tail']:.4f}")
    return {"weights": list(weights), "lambda": lam,
            "results": {k: {m: v.get(m) for m in
                            ("ndcg", "recall", "hit_rate", "coverage", "tail",
                             "arp", "diversity")}
                        for k, v in results.items()}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--interactions",
                        default="data/processed_gr/interactions.parquet",
                        type=Path)
    parser.add_argument("--media", default="data/processed_hgn_none/../processed_gr/media.parquet",
                        type=Path)
    parser.add_argument("--parts", nargs="+", default=["a", "b", "c"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--grid-users", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--min-user-rating", type=int, default=5)
    parser.add_argument("--min-item-rating", type=int, default=5)
    parser.add_argument("--factors", nargs="+", type=int, default=[16, 32, 64])
    parser.add_argument("--regs", nargs="+", type=float, default=[0.02, 0.05])
    parser.add_argument("--mf-epochs", type=int, default=12)
    parser.add_argument("--min-co", type=int, default=5)
    parser.add_argument("--neighbours", type=int, default=500)
    parser.add_argument("--session-length", type=int, default=5)
    parser.add_argument("--decay", type=float, default=0.7)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--lambdas", nargs="+", type=float,
                        default=[1.0, 0.9, 0.7, 0.5])
    args = parser.parse_args()

    report = {}
    path = args.dir / "ladder_gaps.json"

    def save():
        """Gộp vào tệp cũ rồi ghi. Gọi sau mỗi phần."""
        merged = (json.loads(path.read_text(encoding="utf-8"))
                  if path.exists() else {})
        merged.update(report)
        path.write_text(json.dumps(merged, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"-> {path}", flush=True)

    if "a" in args.parts:
        report["A_rating_prediction"] = part_a(args)
        save()

    if "b" in args.parts or "c" in args.parts:
        data = load(args.dir)
        train, val, test = data["train"], data["val"], data["test"]
        user_index, item_index = data["user_index"], data["item_index"]
        reverse = {v: k for k, v in item_index.items()}
        matrix = user_item_matrix(train, user_index, item_index)
        counts = np.asarray(matrix.sum(axis=0)).ravel()
        order = np.argsort(-counts)
        percentile = np.empty(len(counts))
        percentile[order] = 1.0 - np.arange(len(counts)) / max(len(counts) - 1, 1)
        sizes = {
            "popularity": {reverse[i]: int(counts[i]) for i in range(len(counts))},
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
        frames = {"val": val_frame, "test": test_frame}

        if "b" in args.parts:
            report["B_association_rules"] = part_b(
                args, data, matrix, train_by_user, excluded_all, frames,
                reverse, sizes)
            save()
        if "c" in args.parts:
            similarity = top_k_neighbours(matrix, k=args.neighbours, shrink=0.0)
            report["C_hybrid"] = part_c(
                args, data, matrix, train_by_user, excluded_train, excluded_all,
                frames, reverse, sizes, similarity, args.media)

    save()


if __name__ == "__main__":
    main()
