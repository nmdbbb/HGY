"""Huấn luyện và đánh giá năm mô hình nơ-ron còn thiếu của thang.

    python scripts/run_neural_ladder.py --dir data/processed_hgn_none \
        --models neumf lightgcn gru4rec sasrec cl4srec nrms

Cùng giao thức, cùng rổ ứng viên, cùng `metrics.evaluate` với mọi dòng khác
trong bảng. Ba điều quyết định cách viết script này:

1. **Chấm trên toàn danh mục.** Mọi mô hình chuỗi đều dễ cài thành "chấm trên
   100 ứng viên lấy mẫu" — nhanh hơn hàng chục lần và cho con số cao hơn hẳn,
   nhưng không so được với bất kỳ dòng nào đã có. Ở đây `scores()` trả về vectơ
   dài bằng cả danh mục.
2. **Chọn trên validation bằng dừng sớm.** Máy không có GPU nên dò lưới là
   không khả thi. Dừng sớm vẫn là chọn trên val: sau mỗi `--check-every` epoch
   thì chấm val, giữ bản tốt nhất, và chỉ bản đó được chấm test — đọc test đúng
   một lần.
3. **Lô chuỗi dựng một lần.** Chuỗi của 58.413 người, cắt còn 50 mục gần nhất,
   là một mảng int32 58.413 x 50 = 11 MB. Dựng lại mỗi epoch thì phần lớn thời
   gian chạy là pandas, không phải mạng.
"""

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data.protocol_peruser import (  # noqa: E402
    excluded_candidates, reachable_test)
from truyen_reco.evaluation import metrics  # noqa: E402
from truyen_reco.models import neural  # noqa: E402
from truyen_reco.models.knn import user_item_matrix  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_protocol_baselines import load  # noqa: E402


def build_sequences(train, user_index, item_index, length, n_items):
    """Mảng (người x length) các mục gần nhất, đệm bằng `n_items` ở bên trái."""
    frame = train[["user", "media_id", "created_at"]].copy()
    frame["row"] = frame["user"].map(user_index)
    frame["col"] = frame["media_id"].map(item_index)
    frame = frame.dropna(subset=["row", "col"])
    frame = frame.sort_values(["user", "created_at"], kind="mergesort")
    frame["rank"] = frame.groupby("row", sort=False).cumcount(ascending=False)
    frame = frame[frame["rank"] < length]
    out = np.full((len(user_index), length), n_items, dtype=np.int64)
    out[frame["row"].astype(int).to_numpy(),
        length - 1 - frame["rank"].to_numpy()] = frame["col"].astype(int)
    return out


def ranker_for(model, user_index, reverse, excluded, k, sequences=None,
               args_length=50):
    """recommend_fn chấm TOÀN danh mục rồi mới loại ứng viên."""
    def recommend(user, _history):
        row = user_index.get(user)
        if row is None:
            return []
        if sequences is None:
            scores = model.scores([row])[0]
        else:
            # Cắt đúng độ dài mô hình được huấn luyện. Huấn luyện đưa vào
            # `block[:, :-1]` (49 cột) nên bảng vị trí của SASRec chỉ có 49
            # hàng; đưa cả 50 cột vào lúc chấm thì tra vị trí 49 và vỡ
            # (IndexError), mà nếu không vỡ thì mọi vị trí lệch một bước so
            # với lúc học. Lấy 49 mục **gần nhất** giữ đúng chỗ "cột phải
            # cùng là mục mới nhất" ở cả hai phía.
            window = sequences[row:row + 1, -(args_length - 1):]
            scores = model.scores(torch.as_tensor(window))[0]
        scores = scores.astype(np.float32, copy=True)
        drop = excluded.get(user)
        if drop is not None and len(drop):
            scores[drop] = -np.inf
        top = np.argpartition(-scores, min(k, len(scores) - 1))[:k]
        top = top[np.argsort(-scores[top])]
        return [reverse[int(i)] for i in top if np.isfinite(scores[int(i)])]
    return recommend


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dir", default="data/processed_hgn_none", type=Path)
    parser.add_argument("--models", nargs="+",
                        default=["neumf", "lightgcn", "gru4rec", "sasrec",
                                 "cl4srec", "nrms"])
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-users", type=int, default=2000)
    parser.add_argument("--grid-users", type=int, default=600)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--check-every", type=int, default=2)
    parser.add_argument("--patience", type=int, default=0,
                        help="dừng sau bao nhiêu LẦN CHẤM val liên "
                             "tiếp không khá hơn; 0 = chạy hết")
    parser.add_argument("--length", type=int, default=50)
    parser.add_argument("--factors", type=int, default=64)
    parser.add_argument("--batch", type=int, default=4096)
    parser.add_argument("--seq-batch", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--content-dims", type=int, default=128)
    parser.add_argument("--media", default="data/processed_gr/media.parquet",
                        type=Path)
    parser.add_argument("--min-df", type=int, default=20)
    parser.add_argument("--out-name", default="neural_results.json",
                        help="đổi tên khi chạy thử để số nháp không "
                             "lẫn vào tệp kết quả thật")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    neural.set_threads(args.threads)
    torch.manual_seed(args.seed)

    started = time.monotonic()
    data = load(args.dir)
    train, val, test = data["train"], data["val"], data["test"]
    user_index, item_index = data["user_index"], data["item_index"]
    reverse = {v: k for k, v in item_index.items()}
    n_users, n_items = len(user_index), len(item_index)
    matrix = user_item_matrix(train, user_index, item_index)
    train_by_user = train.groupby("user")["media_id"].apply(list).to_dict()
    excluded_train = excluded_candidates(train, val.iloc[0:0], user_index,
                                         item_index, data["series_of"])
    excluded_all = excluded_candidates(train, val, user_index, item_index,
                                       data["series_of"])
    val_frame, _ = reachable_test(val, excluded_train, item_index)
    test_frame, _ = reachable_test(test, excluded_all, item_index)
    sequences = build_sequences(train, user_index, item_index, args.length,
                               n_items)
    # Bốn bảng để `metrics.evaluate` tính được novelty / ARP / long-tail.
    # Không đưa thì bốn cột đó ra nan, và các dòng nơ-ron sẽ trống đúng ở chỗ
    # bảng so sánh cần nhất: độ chính xác đổi lấy cái gì.
    counts = np.asarray(matrix.sum(axis=0)).ravel()
    order = np.argsort(-counts)
    percentile = np.empty(len(counts))
    percentile[order] = 1.0 - np.arange(len(counts)) / max(len(counts) - 1, 1)
    extra = {
        "popularity": {reverse[i]: int(counts[i]) for i in range(len(counts))},
        "n_users": matrix.shape[0],
        "pop_percentile": {reverse[i]: float(percentile[i])
                           for i in range(len(counts))},
        "head_items": {reverse[int(i)]
                       for i in order[:max(1, len(counts) // 100)]},
    }
    pairs_user = matrix.tocoo().row.astype(np.int64)
    pairs_item = matrix.tocoo().col.astype(np.int64)
    print(f"{args.dir}: {n_users:,} người | {n_items:,} mục | "
          f"{len(pairs_user):,} cặp | chuỗi {sequences.shape} "
          f"({time.monotonic() - started:.0f}s)", flush=True)

    content = None
    if "nrms" in args.models:
        from sklearn.decomposition import TruncatedSVD

        from truyen_reco.models.content import build_item_features
        media = pd.read_parquet(args.media)
        item_matrix, vocabulary = build_item_features(
            media, item_index, merged_of=data["merged_of"],
            min_df=args.min_df)
        svd = TruncatedSVD(n_components=args.content_dims,
                           random_state=args.seed)
        content = svd.fit_transform(item_matrix).astype(np.float32)
        content /= np.maximum(np.linalg.norm(content, axis=1, keepdims=True),
                              1e-9)
        print(f"  nội dung cho NRMS: {content.shape}, giữ "
              f"{svd.explained_variance_ratio_.sum() * 100:.1f}% phương sai",
              flush=True)

    rng = np.random.default_rng(args.seed)
    results = {}

    def evaluate(model, frame, excluded, users, sequence_based, full=False):
        model.eval()
        recommend = ranker_for(model, user_index, reverse, excluded, args.k,
                               sequences if sequence_based else None,
                               args.length)
        got = metrics.evaluate(recommend, frame, train_by_user, k=args.k,
                               max_users=users, catalog_size=n_items,
                               seed=args.seed, **(extra if full else {}))
        model.train()
        return got

    def train_model(name, model, label, note):
        sequence_based = model.kind == "sequence"
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        best, stale = None, 0
        rows = np.arange(n_users) if sequence_based else np.arange(len(pairs_user))
        batch_size = args.seq_batch if sequence_based else args.batch
        for epoch in range(1, args.epochs + 1):
            order = rng.permutation(rows)
            total, steps = 0.0, 0
            for start in range(0, len(order), batch_size):
                picked = order[start:start + batch_size]
                if sequence_based:
                    block = sequences[picked]
                    # Mục tiêu ở mỗi vị trí là mục kế tiếp trong chính chuỗi.
                    inputs = torch.as_tensor(block[:, :-1])
                    positives = torch.as_tensor(block[:, 1:])
                    negatives = torch.as_tensor(
                        rng.integers(0, n_items, positives.shape))
                    if name == "nrms":
                        positives = positives[:, -1]
                        negatives = negatives[:, -1]
                    batch = (inputs, positives, negatives)
                    if (positives == n_items).all():
                        continue
                else:
                    batch = (torch.as_tensor(pairs_user[picked]),
                             torch.as_tensor(pairs_item[picked]),
                             torch.as_tensor(
                                 rng.integers(0, n_items, len(picked))))
                loss = model.loss(batch)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total += float(loss.detach())
                steps += 1
            line = (f"  {label} epoch {epoch}/{args.epochs} "
                    f"loss {total / max(steps, 1):.4f}")
            if epoch % args.check_every == 0 or epoch == args.epochs:
                got = evaluate(model, val_frame, excluded_train,
                               args.grid_users, sequence_based)
                line += f" | val NDCG {got['ndcg']:.4f}"
                if best is None or got["ndcg"] > best[0]:
                    best = (got["ndcg"], epoch, copy.deepcopy(model.state_dict()))
                    stale = 0
                    line += " *"
                else:
                    stale += 1
            print(line + f"  ({time.monotonic() - started:.0f}s)", flush=True)
            if args.patience and stale >= args.patience:
                # Mốc đầu tiên lộ ra sau lượt 20 epoch: **cả năm** mô hình đều
                # chọn đúng epoch cuối, val vẫn đang lên. Năm lần như nhau thì
                # không phải tình cờ — ngân sách epoch đang là thứ chặn, không
                # phải kiến trúc. Nên đổi cách dừng: cho ngân sách lớn và để
                # val tự nói chỗ bẹt, thay vì đọc một con số ở biên rồi gọi nó
                # là giá trị của phương pháp.
                print(f"  dừng sớm: {stale} lần chấm không khá hơn "
                      f"(tốt nhất epoch {best[1]})", flush=True)
                break
        if best is None:
            return
        model.load_state_dict(best[2])
        got = evaluate(model, test_frame, excluded_all, args.max_users,
                       sequence_based, full=True)
        got["note"] = note + f"; val chọn epoch {best[1]}/{args.epochs}"
        got["val_ndcg"] = best[0]
        results[label] = got
        print(f"  -> {label}: val chọn epoch {best[1]}, test NDCG "
              f"{got['ndcg']:.4f} | Recall {got['recall']:.4f}", flush=True)
        path = args.dir / args.out_name
        previous = (json.loads(path.read_text(encoding="utf-8"))
                    if path.exists() else {})
        previous.update(results)
        path.write_text(json.dumps(previous, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        print(f"  -> {path}", flush=True)

    if "neumf" in args.models:
        train_model("neumf", neural.NeuMF(n_users, n_items, factors=32),
                    "NeuMF", "GMF 32 chiều + MLP 64-32-16, 1 mẫu âm, BCE")
    if "lightgcn" in args.models:
        adjacency = neural.normalized_adjacency(matrix)
        train_model("lightgcn",
                    neural.LightGCN(n_users, n_items, factors=args.factors,
                                    layers=3, adjacency=adjacency),
                    "LightGCN",
                    f"{args.factors} chiều, 3 tầng lan truyền, BPR, L2 1e-4")
    if "gru4rec" in args.models:
        train_model("gru4rec",
                    neural.GRU4Rec(n_items, factors=args.factors,
                                   hidden=args.factors),
                    "GRU4Rec",
                    f"GRU 1 tầng {args.factors} chiều, chuỗi {args.length}, "
                    "BPR ở mọi vị trí")
    if "sasrec" in args.models:
        train_model("sasrec",
                    neural.SASRec(n_items, factors=args.factors, blocks=2,
                                  heads=2, length=args.length - 1),
                    "SASRec",
                    f"2 khối, 2 đầu, {args.factors} chiều, chuỗi "
                    f"{args.length}, mất mát ở mọi vị trí")
    if "cl4srec" in args.models:
        train_model("cl4srec",
                    neural.CL4SRec(n_items, factors=args.factors, blocks=2,
                                   heads=2, length=args.length - 1,
                                   contrast=0.1),
                    "CL4SRec",
                    "SASRec + NT-Xent trên hai biến thể (cắt / che / đảo), "
                    "lambda 0,1")
    if "nrms" in args.models and content is not None:
        train_model("nrms",
                    neural.NRMSStyle(content, factors=args.factors, heads=2),
                    "NRMS-style",
                    f"bộ mã mục từ nội dung {args.content_dims} chiều, "
                    f"attention trên lịch sử; KHÔNG có attention mức từ")

    print(f"\nxong ({(time.monotonic() - started) / 60:.1f} phút)")


if __name__ == "__main__":
    main()
