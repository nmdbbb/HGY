"""So bản chất hai bộ dữ liệu TRƯỚC khi chạy mô hình trên bộ thứ hai.

Chạy mô hình trước rồi mới giải thích con số là cách chắc chắn nhất để tự lừa
mình: thấy kết quả thấp thì sẽ đi tìm lý do nào nghe hợp lý nhất. Nên đo trước
các tính chất quyết định xem từng phát hiện của nhánh AniList có *thể* lặp lại
được hay không, rồi mới huấn luyện.

Bốn phát hiện cần kiểm tính lặp lại, và tính chất dữ liệu quyết định mỗi cái:

  F1  bốn mặt tách riêng hơn chỉ dùng chủ đề; riêng mặt `art` thật (t=4,10).
      -> cần: phủ ảnh bìa đủ lớn, và lát `new_item` không rỗng.
  F2  chấm toàn danh mục hơn xếp lại top-400 của ALS (trần 35,6%).
      -> cần: đo lại chính trần đó trên bộ mới. Đây là số so sánh trực tiếp
         nhất, vì nó không phụ thuộc mô hình nào mà chỉ phụ thuộc dữ liệu.
  F3  DropoutNet cho cổng độ phổ biến làm new_user tăng 99%.
      -> cần: đầu độ phổ biến phải lệch. Nếu Goodreads ÍT lệch hơn thì cổng
         có ít việc để làm, và hiệu ứng nhỏ hơn là điều phải dự đoán TRƯỚC,
         không phải chỗ để kết luận "phương pháp không lặp lại".
  F4  chỉ tiêu số nước ngược chiều với độ khớp gu.
      -> KHÔNG kiểm được: Goodreads không có trường xuất xứ. Ghi N/A.

    python scripts/diagnose_goodreads.py
    python scripts/diagnose_goodreads.py --candidates 400 --users 600
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.data import preprocess  # noqa: E402

PACE_COLUMNS = None  # nạp muộn để không buộc import nặng khi chỉ xem help


def gini(values):
    """Độ lệch của đầu độ phổ biến. 0 = đều tuyệt đối, 1 = một item ăn tất."""
    x = np.sort(np.asarray(values, dtype=np.float64))
    x = x[x >= 0]
    if x.size == 0 or x.sum() == 0:
        return float("nan")
    n = x.size
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def load_arm(out_dir):
    """Nạp những gì có, bỏ qua những gì chưa chạy tới."""
    out_dir = Path(out_dir)
    arm = {"dir": str(out_dir), "exists": out_dir.exists()}
    if not out_dir.exists():
        return arm
    for name, key in (("train.parquet", "train"), ("test.parquet", "test"),
                      ("cold_users.parquet", "cold"),
                      ("media_core.parquet", "media"),
                      ("interactions.parquet", "raw")):
        path = out_dir / name
        if path.exists():
            arm[key] = pd.read_parquet(path)
    index_path = out_dir / "index.json"
    if index_path.exists():
        arm["index"] = json.loads(index_path.read_text(encoding="utf-8"))
    facets_path = out_dir / "facets.npz"
    if facets_path.exists():
        arm["facets"] = np.load(facets_path, allow_pickle=False)
    pace_path = out_dir / "pace_table.parquet"
    if pace_path.exists():
        arm["pace_table"] = pd.read_parquet(pace_path)
    return arm


def shape_stats(arm):
    """Quy mô, độ thưa, và hình dạng phân phối — khung chung để so sánh."""
    out = {}
    train, test = arm.get("train"), arm.get("test")
    if train is None:
        return out
    users = train["user"].nunique()
    items = train["media_id"].nunique()
    out["train_rows"] = int(len(train))
    out["train_users"] = int(users)
    out["train_items"] = int(items)
    out["density"] = float(len(train) / max(users * items, 1))
    lengths = train.groupby("user")["media_id"].size()
    for q in (50, 75, 90, 99):
        out[f"hist_len_p{q}"] = float(np.percentile(lengths, q))
    out["hist_len_mean"] = float(lengths.mean())
    # Lát thưa và lát warm chia ở ngưỡng 10 — chính nơi hai kết quả ngược chiều
    # của cổng độ phổ biến sống.
    out["users_sparse_lt10_pct"] = round(100.0 * float((lengths < 10).mean()), 1)

    item_counts = train["media_id"].value_counts()
    out["pop_gini"] = round(gini(item_counts.values), 3)
    total = item_counts.sum()
    for share in (1, 5, 10):
        top = max(1, int(len(item_counts) * share / 100))
        out[f"top{share}pct_item_share"] = round(
            float(item_counts.head(top).sum() / total), 3)

    if test is not None:
        out["test_rows"] = int(len(test))
        slices = preprocess.evaluation_slices(train, test)
        if arm.get("cold") is not None:
            slices["new_user"] = arm["cold"]
        out["slice_rows"] = {k: int(len(v)) for k, v in slices.items()}
        out["slice_users"] = {k: int(v["user"].nunique()) for k, v in slices.items()}
    if "event_time" in train.columns:
        span = (pd.to_datetime(train["event_time"]).min(),
                pd.to_datetime(train["event_time"]).max())
        out["train_span"] = [str(span[0].date()), str(span[1].date())]
    return out


def catalog_stats(arm):
    """Thành phần danh mục và phủ của từng mặt đặc trưng."""
    out = {}
    media = arm.get("media")
    if media is None:
        return out
    out["media_rows"] = int(len(media))
    if "country" in media:
        counts = media["country"].value_counts().to_dict()
        out["country_counts"] = {str(k): int(v) for k, v in counts.items()}
        out["country_missing_pct"] = round(
            100.0 * float(media["country"].isna().mean()), 1)
    out["cover_pct"] = round(100.0 * float(media["cover_url"].notna().mean()), 1)
    out["art_role_pct"] = round(
        100.0 * float((media["staff_art_ids"].str.len() > 0).mean()), 1)
    out["story_role_pct"] = round(
        100.0 * float((media["staff_story_ids"].str.len() > 0).mean()), 1)
    out["tagged_pct"] = round(
        100.0 * float((media["tag_names"].str.len() > 0).mean()), 1)
    out["tags_per_item"] = round(float(media["tag_names"].str.len().mean()), 1)
    length = pd.to_numeric(media["chapters"], errors="coerce")
    out["length_known_pct"] = round(100.0 * float(length.notna().mean()), 1)
    out["length_median"] = (None if length.notna().sum() == 0
                            else float(length.median()))

    facets = arm.get("facets")
    if facets is not None:
        out["facet_coverage"] = {}
        for key in facets.files:
            if not key.startswith("facet_"):
                continue
            vectors = facets[key]
            out["facet_coverage"][key[len("facet_"):]] = {
                "dim": int(vectors.shape[1]),
                "nonzero_pct": round(100.0 * float(
                    (np.linalg.norm(vectors, axis=1) > 1e-8).mean()), 1),
            }

    # Cột nhịp truyện nào bị thoái hoá thành hằng số: đó là hệ quả đo được của
    # việc một bộ dữ liệu thiếu hẳn một loại tín hiệu (bỏ dở, tiến độ).
    table = arm.get("pace_table")
    if table is not None:
        from truyen_reco.features.build_features import PACE_COLUMNS as cols
        flat = [c for c in cols if c in table.columns
                and float(table[c].std()) < 1e-9]
        out["pace_columns_total"] = len(cols)
        out["pace_columns_flat"] = flat

    # Cùng câu hỏi cho đầu THỜI ĐIỂM của hai tháp. Bảng thời điểm không được
    # lưu ra đĩa nên tính lại ở đây đúng công thức của
    # `train_embedding.time_features`, rồi chuẩn hoá: cột nào std = 0 thì sau
    # phép chuẩn hoá thành 0 toàn bộ, tức đầu thời điểm mất một trục mà vẫn
    # chạy bình thường, không báo gì. Trên Goodreads mọi quyển đã in đều được
    # gán `FINISHED` (mục 1 docs/goodreads-arm.md), nên hai cột trạng thái
    # phát hành bẹt từ trước khi mô hình nhìn thấy dữ liệu.
    recomputed = {
        "is_releasing": (media["pub_status"] == "RELEASING").astype(float),
        "is_finished": (media["pub_status"] == "FINISHED").astype(float),
        "age_years": pd.to_numeric(media["start_year"], errors="coerce"),
    }
    out["time_columns_flat"] = sorted(
        name for name, values in recomputed.items()
        if float(np.nanstd(values.to_numpy(dtype=float))) < 1e-9)
    out["time_columns_total"] = 5          # xem train_embedding.TIME_COLUMNS
    return out


def candidate_ceiling(arm, n_candidates=400, max_users=600, seed=0):
    """Trần thu hồi của hệ hai tầng: bao nhiêu % item trong test lọt top-N của ALS.

    Đây là con số so sánh trực tiếp nhất giữa hai bộ dữ liệu, vì nó không phụ
    thuộc vào mô hình xếp hạng nào — chỉ phụ thuộc vào việc ALS có kéo được
    item đúng vào rổ ứng viên hay không. Trên AniList số này là 35,6%, tức mọi
    cải tiến ở tầng xếp lại đều bị chặn trên bởi nó.
    """
    facets, index = arm.get("facets"), arm.get("index")
    train, test = arm.get("train"), arm.get("test")
    if facets is None or index is None or test is None or train is None:
        return None
    if "user_factors" not in facets.files:
        return None
    user_factors = facets["user_factors"]
    item_factors = facets["item_factors"]
    user_index = index["user_index"]
    item_index = {int(k): v for k, v in index["item_index"].items()}

    grouped = test.groupby("user")["media_id"].apply(set)
    users = [u for u in grouped.index if str(u) in user_index
             or u in user_index]
    if not users:
        return {"note": "không có người đọc nào của test nằm trong train"}
    rng = np.random.default_rng(seed)
    if len(users) > max_users:
        users = list(rng.choice(np.asarray(users, dtype=object), max_users,
                                replace=False))

    seen = train.groupby("user")["media_id"].apply(set).to_dict()
    hits = total = 0
    for user in users:
        row = user_index.get(user, user_index.get(str(user)))
        if row is None:
            continue
        scores = item_factors @ user_factors[row]
        # Bỏ item đã đọc khỏi rổ ứng viên, đúng như hệ hai tầng lúc phục vụ.
        for media_id in seen.get(user, ()):
            position = item_index.get(media_id)
            if position is not None:
                scores[position] = -np.inf
        top = np.argpartition(-scores, min(n_candidates, len(scores) - 1))
        top = set(top[:n_candidates].tolist())
        wanted = {item_index[m] for m in grouped[user] if m in item_index}
        if not wanted:
            continue
        hits += len(wanted & top)
        total += len(wanted)
    return {"n_candidates": n_candidates, "users": len(users),
            "test_items_considered": total,
            "recall_ceiling": round(hits / max(total, 1), 4)}


def slice_history(arm):
    """Số quan sát train trên mỗi người, TÁCH THEO LÁT đánh giá.

    Vì sao cần: cơ chế đã ghi cho P3 và P5 là *số quan sát trên mỗi người*. Nếu
    đúng thì chênh lệch giữa hai nhánh phải lớn nhất ở lát mà hai nhánh lệch
    nhau nhiều nhất về con số này, và nhỏ nhất ở lát chúng khớp nhau. Lát
    `sparse_user` (< 10 mục) khớp gần như hoàn hảo giữa hai bộ, nên nó là nhóm
    đối chứng có sẵn: chỗ nào vẫn lệch ở đó thì KHÔNG giải thích được bằng số
    quan sát, và phải tìm nguyên nhân khác (mục 10 docs/goodreads-arm.md).
    """
    train, test = arm.get("train"), arm.get("test")
    if train is None or test is None:
        return None
    hist = train.groupby("user")["media_id"].size()
    test_users = set(test["user"].unique())
    known = {u: int(hist[u]) for u in test_users if u in hist.index}
    out = {"new_users_in_test": int(len(test_users) - len(known))}
    groups = {"all": [v for v in known.values()],
              "warm": [v for v in known.values() if v >= 10],
              "sparse_user": [v for v in known.values() if v < 10]}
    for name, values in groups.items():
        if not values:
            continue
        out[f"{name}_users"] = len(values)
        out[f"{name}_hist_median"] = float(np.median(values))
        out[f"{name}_hist_mean"] = round(float(np.mean(values)), 1)
    return out


def nonpersonal_heads(arm, max_users=3000, seed=0, k=10):
    """Hai mốc KHÔNG cá nhân hoá: phổ biến toàn kỳ so với đang nổi 60 ngày.

    Vì sao phải đo: trên Goodreads mốc "đang nổi 60 ngày" đạt NDCG@10 0,0824 ở
    lát `all` — cao hơn cả ALS (0,0734) và cả hai tháp (0,0625). Trên AniList
    nó là mốc *yếu nhất*. Chênh lệch đó không phải tính chất của mô hình nào,
    nên phải truy về dữ liệu, và chỗ truy được là **đầu danh mục**: NDCG@10
    chỉ nhìn 10 vị trí, nên đo trực tiếp tỉ lệ item test nằm trong top-10 của
    mỗi bảng xếp hạng, sau khi bỏ những mục người đọc đã có trong train (đúng
    như lúc phục vụ).

    Đo thêm **tuổi của khối lượng tương tác ở đầu danh mục**, vì đó là lời giải
    thích kiểm được: nếu đầu danh mục toàn kỳ của một bộ là sách cũ mà lượt đọc
    đã xảy ra từ nhiều năm trước, thì phổ biến toàn kỳ là một ước lượng *lạc
    hậu* cho kỳ test, còn cửa sổ 60 ngày thì không.
    """
    train, test = arm.get("train"), arm.get("test")
    if train is None or test is None or "event_time" not in train.columns:
        return None
    event = pd.to_datetime(train["event_time"], utc=True, errors="coerce")
    cutoff = event.max()
    if pd.isna(cutoff):
        return None
    age_days = (cutoff - event).dt.days

    all_pop = train["media_id"].value_counts()
    recent = train[event >= cutoff - pd.Timedelta(days=60)]
    recent_pop = recent["media_id"].value_counts()

    seen = train.groupby("user")["media_id"].apply(set)
    rng = np.random.default_rng(seed)
    users = test["user"].unique()
    if len(users) > max_users:
        users = rng.choice(users, max_users, replace=False)
    truth = test[test["user"].isin(set(users))].groupby("user")["media_id"].apply(set)

    out = {"cutoff": str(cutoff.date()), "users_scored": int(len(users)),
           "recent_rows_pct": round(100.0 * len(recent) / max(len(train), 1), 1)}
    for tag, pop in (("all_time", all_pop), ("recent_60d", recent_pop)):
        ranked = list(pop.index)
        hits = total = 0
        for user in users:
            wanted = truth.get(user)
            if not wanted:
                continue
            have = seen.get(user, set())
            top, i = [], 0
            while len(top) < k and i < len(ranked):
                if ranked[i] not in have:
                    top.append(ranked[i])
                i += 1
            hits += len(set(top) & wanted)
            total += min(k, len(wanted))
        out[f"hit{k}_{tag}_pct"] = round(100.0 * hits / max(total, 1), 2)
    out[f"hit{k}_ratio_recent_over_all"] = round(
        out[f"hit{k}_recent_60d_pct"] / max(out[f"hit{k}_all_time_pct"], 1e-9), 2)

    head = set(all_pop.head(400).index)
    out["age_days_median_catalog"] = float(age_days.median())
    out["age_days_median_head400"] = float(age_days[train["media_id"].isin(head)].median())
    out["age_over_2y_pct_catalog"] = round(100.0 * float((age_days > 730).mean()), 1)
    out["age_over_2y_pct_head400"] = round(
        100.0 * float((age_days[train["media_id"].isin(head)] > 730).mean()), 1)
    return out


def pretty(label, left, right, fmt="{}"):
    def cell(value):
        if value is None:
            return "—"
        if isinstance(value, float):
            return fmt.format(value)
        return str(value)
    print(f"  {label:<34} {cell(left):>18} {cell(right):>18}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--anilist-dir", default="data/processed", type=Path)
    parser.add_argument("--goodreads-dir", default="data/processed_gr", type=Path)
    parser.add_argument("--candidates", type=int, default=400,
                        help="kích thước rổ ứng viên của hệ hai tầng cũ")
    # 1500 người đọc là đúng số mà mục 1 của docs/data-diagnosis.md đã dùng khi
    # đo trần 35,6%. Lấy số khác thì hai con số không so được với nhau: đo thử
    # với 600 người cho 37,4%, lệch 1,8 điểm chỉ vì cỡ mẫu.
    parser.add_argument("--users", type=int, default=1500)
    parser.add_argument("--skip-ceiling", action="store_true")
    args = parser.parse_args()

    arms = {"anilist": load_arm(args.anilist_dir),
            "goodreads": load_arm(args.goodreads_dir)}
    report = {}
    for name, arm in arms.items():
        if not arm.get("exists"):
            print(f"[THIẾU] {name}: {arm['dir']} chưa có")
            report[name] = {"missing": True}
            continue
        block = {"shape": shape_stats(arm), "catalog": catalog_stats(arm),
                 "heads": nonpersonal_heads(arm),
                 "slice_history": slice_history(arm)}
        if not args.skip_ceiling:
            block["candidate_ceiling"] = candidate_ceiling(
                arm, args.candidates, args.users)
        report[name] = block

    a = report.get("anilist", {})
    g = report.get("goodreads", {})
    sa, sg = a.get("shape", {}), g.get("shape", {})
    ca, cg = a.get("catalog", {}), g.get("catalog", {})

    print("\n" + "=" * 74)
    print("  QUY MÔ VÀ ĐỘ THƯA".ljust(36) + "AniList".rjust(18) + "Goodreads".rjust(18))
    print("=" * 74)
    for key, label in (("train_rows", "tương tác train"),
                       ("train_users", "người đọc"),
                       ("train_items", "truyện"),
                       ("test_rows", "tương tác test")):
        pretty(label, f"{sa.get(key):,}" if sa.get(key) else None,
               f"{sg.get(key):,}" if sg.get(key) else None)
    pretty("độ thưa", sa.get("density"), sg.get("density"), "{:.7f}")
    pretty("khoảng thời gian",
           " → ".join(sa.get("train_span", [])) or None,
           " → ".join(sg.get("train_span", [])) or None)

    print("\n" + "=" * 74)
    print("  ĐẦU ĐỘ PHỔ BIẾN  (F3: cổng có việc để làm hay không)")
    print("=" * 74)
    pretty("Gini của số lượt theo truyện", sa.get("pop_gini"), sg.get("pop_gini"),
           "{:.3f}")
    for share in (1, 5, 10):
        pretty(f"tỉ lệ lượt vào {share}% truyện đầu",
               sa.get(f"top{share}pct_item_share"),
               sg.get(f"top{share}pct_item_share"), "{:.3f}")

    print("\n" + "=" * 74)
    print("  ĐỘ DÀI LỊCH SỬ  (quyết định lát thưa / warm chia ở đâu)")
    print("=" * 74)
    for q in (50, 75, 90, 99):
        pretty(f"phân vị {q}", sa.get(f"hist_len_p{q}"), sg.get(f"hist_len_p{q}"),
               "{:.0f}")
    pretty("% người đọc có < 10 mục", sa.get("users_sparse_lt10_pct"),
           sg.get("users_sparse_lt10_pct"), "{:.1f}")

    print("\n" + "=" * 74)
    print("  LÁT ĐÁNH GIÁ  (F1: lát new_item có tồn tại không)")
    print("=" * 74)
    for name in ("all", "new_user", "new_item", "sparse_user", "warm"):
        pretty(f"lát {name}",
               f"{sa.get('slice_rows', {}).get(name, 0):,}",
               f"{sg.get('slice_rows', {}).get(name, 0):,}")

    print("\n" + "=" * 74)
    print("  DANH MỤC VÀ PHỦ ĐẶC TRƯNG")
    print("=" * 74)
    pretty("số truyện", f"{ca.get('media_rows'):,}" if ca.get("media_rows") else None,
           f"{cg.get('media_rows'):,}" if cg.get("media_rows") else None)
    pretty("có xuất xứ (JP/KR/CN)",
           f"{100 - ca.get('country_missing_pct', 100):.1f}%"
           if ca.get("country_missing_pct") is not None else None,
           "0,0%  (KHÔNG CÓ TRƯỜNG)" if cg.get("media_rows") else None)
    pretty("có ảnh bìa dùng được", ca.get("cover_pct"), cg.get("cover_pct"), "{:.1f}%")
    pretty("tách được vai hoạ sĩ", ca.get("art_role_pct"), cg.get("art_role_pct"),
           "{:.1f}%")
    pretty("tách được vai tác giả", ca.get("story_role_pct"),
           cg.get("story_role_pct"), "{:.1f}%")
    pretty("có tag/kệ", ca.get("tagged_pct"), cg.get("tagged_pct"), "{:.1f}%")
    pretty("số tag trung bình", ca.get("tags_per_item"), cg.get("tags_per_item"),
           "{:.1f}")
    pretty("biết độ dài tác phẩm", ca.get("length_known_pct"),
           cg.get("length_known_pct"), "{:.1f}%")
    for name in ("theme", "creator", "pace", "art"):
        la = ca.get("facet_coverage", {}).get(name)
        lg = cg.get("facet_coverage", {}).get(name)
        pretty(f"mặt {name}: phủ",
               f"{la['nonzero_pct']:.1f}% ({la['dim']}d)" if la else None,
               f"{lg['nonzero_pct']:.1f}% ({lg['dim']}d)" if lg else None)
    pretty("cột nhịp bị thoái hoá thành hằng",
           f"{len(ca.get('pace_columns_flat', []))}/{ca.get('pace_columns_total', '?')}"
           if ca.get("pace_columns_total") else None,
           f"{len(cg.get('pace_columns_flat', []))}/{cg.get('pace_columns_total', '?')}"
           if cg.get("pace_columns_total") else None)
    for name, block in (("AniList", ca), ("Goodreads", cg)):
        if block.get("pace_columns_flat"):
            print(f"       {name} mất: {', '.join(block['pace_columns_flat'])}")
    # `pretty` tự in rồi trả về None, nên bọc thêm print() sẽ in ra một dòng
    # "None" giữa bảng.
    pretty("cột thời điểm bẹt",
           f"{len(ca.get('time_columns_flat', []))}/"
           f"{ca.get('time_columns_total', '?')}",
           f"{len(cg.get('time_columns_flat', []))}/"
           f"{cg.get('time_columns_total', '?')}")
    for name, block in (("AniList", ca), ("Goodreads", cg)):
        if block.get("time_columns_flat"):
            print(f"       {name} mất: {', '.join(block['time_columns_flat'])}")

    print("\n" + "=" * 74)
    print(f"  F2: TRẦN THU HỒI CỦA HỆ HAI TẦNG (rổ {args.candidates} ứng viên)")
    print("=" * 74)
    for name, block in (("AniList", a), ("Goodreads", g)):
        ceiling = block.get("candidate_ceiling")
        if not ceiling or "recall_ceiling" not in ceiling:
            print(f"  {name:<12} —  {ceiling.get('note', 'chưa đo được') if ceiling else 'chưa chạy'}")
            continue
        print(f"  {name:<12} {ceiling['recall_ceiling'] * 100:5.1f}% "
              f"({ceiling['test_items_considered']:,} item test, "
              f"{ceiling['users']} người đọc)")
    print("  -> mọi cải tiến ở tầng XẾP LẠI đều bị chặn trên bởi con số này.")

    print()
    print("=" * 74)
    print("  F3: HAI MỐC KHÔNG CÁ NHÂN HOÁ — ĐẦU DANH MỤC CÓ LẠC HẬU KHÔNG")
    print("=" * 74)
    ha, hg = a.get("heads") or {}, g.get("heads") or {}
    for key, label, fmt in (
            ("hit10_all_time_pct", "hit@10 phổ biến toàn kỳ", "{:.2f}%"),
            ("hit10_recent_60d_pct", "hit@10 đang nổi 60 ngày", "{:.2f}%"),
            ("hit10_ratio_recent_over_all", "tỉ số 60 ngày / toàn kỳ", "{:.2f}x"),
            ("recent_rows_pct", "% dòng train trong 60 ngày cuối", "{:.1f}%"),
            ("age_days_median_catalog", "tuổi lượt đọc, toàn danh mục", "{:.0f} ngày"),
            ("age_days_median_head400", "tuổi lượt đọc, top-400 toàn kỳ", "{:.0f} ngày"),
            ("age_over_2y_pct_head400", "% lượt ở top-400 cũ hơn 2 năm", "{:.1f}%")):
        pretty(label, ha.get(key), hg.get(key), fmt)
    print("  -> đầu danh mục GIÀ hơn cả danh mục thì phổ biến toàn kỳ là ước")
    print("     lượng lạc hậu, và cửa sổ 60 ngày thắng vì dữ liệu, không vì mô hình.")

    print()
    print("=" * 74)
    print("  F4: SỐ QUAN SÁT TRÊN MỖI NGƯỜI, TÁCH THEO LÁT")
    print("=" * 74)
    la, lg = a.get("slice_history") or {}, g.get("slice_history") or {}
    for key, label, fmt in (
            ("all_hist_median", "lịch sử trung vị, lát all", "{:.0f} mục"),
            ("warm_hist_median", "lịch sử trung vị, lát warm", "{:.0f} mục"),
            ("sparse_user_hist_median", "lịch sử trung vị, lát sparse", "{:.0f} mục"),
            ("warm_users", "số người lát warm", "{}"),
            ("sparse_user_users", "số người lát sparse", "{}"),
            ("new_users_in_test", "người đọc mới trong test", "{}")):
        pretty(label, la.get(key), lg.get(key), fmt)
    print("  -> lát sparse KHỚP giữa hai nhánh (5 so với 4 mục), lát warm thì")
    print("     lệch 4,4 lần. Cơ chế quan sát-trên-mỗi-người dự đoán chênh lệch")
    print("     kết quả phải theo đúng thứ tự đó — kiểm ở mục 11 của tài liệu.")

    out = args.goodreads_dir / "goodreads_diagnosis.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2, ensure_ascii=False,
                              default=str), encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
