"""Giao thức của đề xuất đồ án (HGN-style), tách hẳn khỏi giao thức đang dùng.

Đây là **nhánh đối chứng thứ hai**, và nó đối chứng một thứ khác với nhánh
Goodreads ở `docs/goodreads-arm.md`: nhánh kia hỏi "phương pháp có lặp lại trên
dữ liệu khác không" (đổi dữ liệu, giữ nguyên giao thức); nhánh này hỏi "kết quả
đổi bao nhiêu khi đổi **giao thức**" (giữ nguyên dữ liệu, đổi cách chia tập và
cách lọc). Hai câu hỏi đó không thể trả lời bằng một thí nghiệm.

Bốn chỗ khác với `preprocess.run`, mỗi chỗ đều là một lựa chọn của đề xuất:

1. **Dương = rating >= 4** (score >= 80 sau khi quy về thang 100), không phải
   nhãn suy từ trạng thái đọc. Tuỳ chọn `rule="read"` cho định nghĩa kia, vì đề
   xuất ghi "sẽ thử cả hai".
2. **Lọc k-core 10/5** thay vì 5/5 — người đọc phải có >= 10 tương tác.
3. **Chia 70/10/20 theo thời gian CỦA TỪNG NGƯỜI**, không phải một mốc thời gian
   chung cho cả tập. Đây là chỗ khác biệt nặng nhất: với mốc chung, kỳ test là
   một *thời kỳ khác* và có cả người đọc chưa từng xuất hiện trong train; với
   70/10/20 theo từng người, mọi người test đều có lịch sử và mục test là phần
   sau của chính chuỗi đọc của họ. Hai giao thức này cho hai con số NDCG khác
   nhau cả về *bậc*, nên so chéo chúng mà không ghi rõ là một lỗi đọc số.
4. **Gộp tập thành bộ truyện** (tuỳ chọn), đúng bước 4 mục 3.1 của đề xuất.

Về gộp bộ: trường `series` của dữ liệu thô **không** dùng được trực tiếp. Đo
trên dữ liệu thật (`scripts/build_series_map.py` và mục "gộp bộ" của
`scripts/prepare_protocol.py`):

    series_id là mỗi TẬP một id: series 155838 "The Sandman" chỉ ứng với đúng
    một work trong tập comics, còn các tập khác của cùng bộ nằm ở series_id
    khác. Gộp theo series_id cho 39.910 nhóm / 39.910 tác phẩm — tức không gộp
    được gì.

Nên có hai chiến lược, và chúng khác nhau về *bản chất*, không phải về tham số:

    "volume"   gộp theo ngoặc "(Tên bộ #n)" trong tiêu đề — bắt đúng chuỗi tập
               được đánh số: One Piece 86 tác phẩm, Naruto 71, Fairy Tail 60.
               9.147 tác phẩm, nhưng **36,8% tương tác** (42,0% tương tác dương).
    "franchise" gộp theo TÊN bộ lấy từ series_id — phủ 78,9% tương tác nhưng
               gộp quá tay: Batman thành MỘT mục gồm 316 tác phẩm, "Marvel
               Masterworks" (một dòng xuất bản, không phải một bộ) gồm 168.

Mặc định là "volume", vì nó đúng thứ đề xuất nêu ("Naruto Vol.1…72 thành một
item") và vì "franchise" trộn những truyện không liên quan. "franchise" giữ lại
làm phép thử độ bền, không phải để chọn con số đẹp hơn.
"""

import gzip
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from . import preprocess

# "Tiêu đề (Tên bộ, #12)" hoặc "(Tên bộ #12)". Chặn ngoặc lồng bằng [^()] để
# không ăn sang "(Vol. 3) (Naruto #3)".
VOLUME_PATTERN = re.compile(r"\(([^()]{2,80}?),?\s*#(\d+(?:\.\d+)?)\s*\)")

POSITIVE_RULES = ("rating", "read", "either", "signal")


def series_titles(raw_dir):
    """series_id -> tên bộ, đọc từ file series thô."""
    path = Path(raw_dir) / "goodreads_book_series.json.gz"
    out = {}
    if not path.exists():
        return out
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            name = (row.get("title") or "").strip()
            if name:
                out[str(row.get("series_id"))] = name
    return out


def series_groups(raw_dir, strategy="volume"):
    """`media_id` (work_id) -> khoá nhóm bộ truyện, theo chiến lược đã chọn.

    Trả về cả nhóm một-tác-phẩm; người gọi tự quyết định có gộp hay không, vì
    phép loại "cùng bộ" lúc đánh giá cần biết bộ của *mọi* mục, kể cả mục không
    được gộp.
    """
    if strategy not in ("volume", "franchise"):
        raise ValueError(f"chiến lược gộp không biết: {strategy}")
    path = Path(raw_dir) / "goodreads_books_comics_graphic.json.gz"
    if not path.exists():
        raise FileNotFoundError(f"thiếu {path}")
    titles = series_titles(raw_dir) if strategy == "franchise" else {}
    volume_of = {}
    votes = defaultdict(lambda: defaultdict(int))
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                book = json.loads(line)
            except json.JSONDecodeError:
                continue
            work_id = str(book.get("work_id") or "")
            if not work_id.isdigit():
                continue
            work_id = int(work_id)
            if strategy == "volume":
                match = VOLUME_PATTERN.search(book.get("title") or "")
                if match and work_id not in volume_of:
                    volume_of[work_id] = match.group(1).strip()
            else:
                for series_id in (book.get("series") or []):
                    name = titles.get(str(series_id))
                    if name:
                        votes[work_id][name] += 1
    if strategy == "volume":
        return volume_of
    # Hoà phiếu thì lấy tên theo thứ tự chữ cái để tái lập được.
    return {work: max(counter.items(), key=lambda kv: (kv[1], kv[0]))[0]
            for work, counter in votes.items()}


def mark_positive(frame, rule="rating", rating_threshold=80.0):
    """Cột `is_positive` theo đúng một trong ba định nghĩa, không trộn lẫn."""
    if rule not in POSITIVE_RULES:
        raise ValueError(f"định nghĩa dương không biết: {rule}")
    if rule == "signal":
        # AniList: 78% dòng không có điểm, nên "rating >= 4" sẽ vứt gần hết.
        # Dùng đúng nhãn mà nhánh A đã dùng (`preprocess.label`), để hai giao
        # thức cùng một tập dòng dương và chỉ khác nhau ở phép chia.
        return frame["signal"].isin({"positive", "weak"})
    read = frame["status"].eq("COMPLETED")
    rated = frame["score"] >= rating_threshold
    if rule == "rating":
        return rated.fillna(False)
    if rule == "read":
        return read
    return (rated.fillna(False)) | read


def merge_series(frame, groups):
    """Gộp các tập của cùng một bộ thành một mục, theo đúng mục 3.1 của đề xuất.

    - có tương tác nếu người đọc đọc **ít nhất một tập**;
    - `score` = trung bình các tập *đã chấm* (tập không chấm không kéo xuống);
    - `created_at` = lần tương tác **đầu tiên**;
    - `n_volumes_read` = số tập đã đọc, giữ lại làm tín hiệu mức gắn bó.

    Mục mới nhận id âm (`-hash` của khoá nhóm) để không đụng không gian
    `work_id` dương: nếu dùng id dương thì một mục bộ có thể trùng id với một
    tác phẩm lẻ, và lỗi đó sẽ không báo gì cả, chỉ làm sai kết quả.
    """
    key = frame["media_id"].map(groups)
    has_group = key.notna()
    sizes = key[has_group].value_counts()
    multi = set(sizes[sizes > 1].index)
    in_multi = has_group & key.isin(multi)
    if not in_multi.any():
        out = frame.copy()
        out["n_volumes_read"] = 1
        return out, {"items_merged": 0, "rows_collapsed": 0, "merged_of": {}}

    singles = frame[~in_multi].copy()
    singles["n_volumes_read"] = 1
    group_rows = frame[in_multi].copy()
    group_rows["series_key"] = key[in_multi]
    # id ổn định giữa các lần chạy: băm tên bộ, không dùng thứ tự xuất hiện.
    codes, uniques = pd.factorize(group_rows["series_key"], sort=True)
    group_rows["media_id"] = -(codes.astype(np.int64) + 1)

    aggregated = group_rows.groupby(["user", "media_id"], sort=False).agg(
        created_at=("created_at", "min"),
        updated_at=("updated_at", "max"),
        score=("score", "mean"),
        n_volumes_read=("media_id", "size"),
        status=("status", lambda s: "COMPLETED" if (s == "COMPLETED").any()
                else s.iloc[0]),
        is_positive=("is_positive", "any"),
    ).reset_index()

    keep = [c for c in singles.columns if c in aggregated.columns]
    out = pd.concat([singles[keep], aggregated[keep]], ignore_index=True)
    # Bản đồ `work_id -> id mục đã gộp`: cần cho đặc trưng nội dung (vector của
    # một bộ là trung bình vector các tập) và cho mọi phép tra ngược sau này.
    # Không trả về thì bên ngoài chỉ còn cách đoán, mà id âm thì không đoán được.
    key_to_id = {name: -(i + 1) for i, name in enumerate(uniques)}
    merged_of = {int(work): key_to_id[key] for work, key in groups.items()
                 if key in key_to_id}
    return out, {"items_merged": int(len(uniques)),
                 "rows_collapsed": int(len(group_rows) - len(aggregated)),
                 "merged_of": merged_of,
                 "series_item_ids": {str(name): -(i + 1)
                                     for i, name in enumerate(uniques[:5])}}


def split_per_user(frame, fractions=(0.7, 0.1, 0.2), time_column="created_at",
                   min_items=10):
    """Chia 70/10/20 theo thời gian của **từng người đọc**.

    Cắt theo chỉ số sau khi sắp theo thời gian, dùng `floor`, và bắt buộc mỗi
    tập test có ít nhất 1 mục: với người có đúng 10 mục thì 70/10/20 cho
    7/1/2 — vừa đủ. Người ít hơn `min_items` đã bị lọc k-core loại trước đó;
    kiểm lại ở đây để không âm thầm tạo ra người test không có lịch sử.
    """
    if not np.isclose(sum(fractions), 1.0):
        raise ValueError(f"ba tỉ lệ phải cộng thành 1: {fractions}")
    frame = frame.sort_values(["user", time_column], kind="mergesort")
    position = frame.groupby("user", sort=False).cumcount()
    total = frame.groupby("user", sort=False)["media_id"].transform("size")
    # Cộng epsilon TRƯỚC khi floor, và lý do không phải là thẩm mỹ: trong số
    # thực nhị phân `0,7 + 0,1 = 0,7999999999999999`, nên người có đúng 10 mục
    # cho `floor(10 * 0,7999...) = 7` thay vì 8 — tức mốc val trùng mốc train
    # và **tập val rỗng**. Không có gì báo lỗi; nó chỉ âm thầm biến 70/10/20
    # thành 70/0/30 ở đúng nhóm người đọc đông nhất (vừa qua ngưỡng k-core).
    # Một test trong tests/test_protocol.py giữ chỗ này.
    epsilon = 1e-9
    train_end = np.floor(total * fractions[0] + epsilon).astype(int)
    val_end = np.floor(total * (fractions[0] + fractions[1]) + epsilon).astype(int)
    # Người vừa đủ mục vẫn phải có cả val lẫn test, nếu không thì họ lặng lẽ
    # rơi khỏi tập đánh giá mà số "người test" vẫn trông bình thường.
    if min_items >= 3:
        val_end = np.minimum(np.maximum(val_end, train_end + 1), total - 1)
        train_end = np.minimum(train_end, val_end)
    part = np.where(position < train_end, "train",
                    np.where(position < val_end, "val", "test"))
    frame = frame.assign(part=part)
    short = int((total < min_items).sum())
    return (frame[frame["part"] == "train"].drop(columns="part"),
            frame[frame["part"] == "val"].drop(columns="part"),
            frame[frame["part"] == "test"].drop(columns="part"),
            {"rows_from_users_under_min_items": short})


def run(interactions, raw_dir=None, rule="rating", merge="volume",
        min_user=10, min_item=5, fractions=(0.7, 0.1, 0.2),
        series_strategy=None, groups=None, time_column="created_at"):
    """Dựng đủ một bộ dữ liệu theo giao thức đề xuất và trả về báo cáo quy mô."""
    report = {"rule": rule, "merge": merge or "none",
              "min_user": min_user, "min_item": min_item,
              "rows_raw": int(len(interactions)),
              "users_raw": int(interactions["user"].nunique()),
              "items_raw": int(interactions["media_id"].nunique())}

    frame = interactions.copy()
    frame["is_positive"] = mark_positive(frame, rule)
    frame = frame[frame["is_positive"]]
    report["rows_positive"] = int(len(frame))
    report["users_positive"] = int(frame["user"].nunique())
    report["items_positive"] = int(frame["media_id"].nunique())

    # Bản đồ bộ truyện phải dựng DÙ CÓ GỘP HAY KHÔNG. Lần đầu viết hàm này tôi
    # chỉ dựng nó khi bật gộp, nên biến thể `none` chạy mà **không có phép loại
    # cùng bộ** — đúng phép loại mà mục 7.1 của đề xuất đòi, và đúng chỗ nó
    # quan trọng nhất (bản đã gộp thì các tập đã là một mục nên chẳng còn gì để
    # loại). Dấu hiệu lộ ra trong log: "bỏ vì cùng bộ/đã đọc 0". Hậu quả không
    # phải một lỗi nhỏ: Item-KNN được 0,1537 nhờ phần điểm "gợi tập kế tiếp",
    # tức đúng cái mà cả nhánh này tồn tại để không tính.
    strategy = series_strategy or (merge if merge in ("volume", "franchise")
                                   else "volume")
    if groups is None:
        if raw_dir is None:
            raise ValueError("cần raw_dir để đọc tiêu đề sách, hoặc truyền groups")
        groups = series_groups(raw_dir, strategy)
    report["series_strategy"] = strategy
    report["works_with_series"] = int(len(groups))
    merged_of = {}
    if merge and merge != "none":
        frame, merge_report = merge_series(frame, groups)
        merged_of = merge_report.pop("merged_of", {})
        report.update({f"merge_{k}": v for k, v in merge_report.items()})
        report["rows_after_merge"] = int(len(frame))
        report["items_after_merge"] = int(frame["media_id"].nunique())

    core = preprocess.filter_core(frame, min_user=min_user, min_item=min_item)
    report["rows_core"] = int(len(core))
    report["users_core"] = int(core["user"].nunique())
    report["items_core"] = int(core["media_id"].nunique())
    report["density_pct"] = round(
        100.0 * len(core) / max(core["user"].nunique()
                                * core["media_id"].nunique(), 1), 4)

    train, val, test, split_report = split_per_user(
        core, fractions=fractions, time_column=time_column)
    report.update(split_report)
    report.update({"train_rows": int(len(train)), "val_rows": int(len(val)),
                   "test_rows": int(len(test)),
                   "train_users": int(train["user"].nunique()),
                   "test_users": int(test["user"].nunique()),
                   "train_items": int(train["media_id"].nunique())})
    lengths = train.groupby("user")["media_id"].size()
    report["hist_median"] = float(lengths.median())
    report["hist_mean"] = round(float(lengths.mean()), 1)

    # Bộ của từng mục, để loại ứng viên "cùng bộ" lúc đánh giá. Với bản đã gộp,
    # mục bộ là chính nó; với bản chưa gộp, phải tra sang nhóm.
    series_of = {}
    for media_id in core["media_id"].unique():
        if media_id < 0:
            series_of[int(media_id)] = f"series:{media_id}"
        else:
            name = groups.get(media_id)
            series_of[int(media_id)] = f"series:{name}" if name else None

    return {"train": train, "val": val, "test": test, "core": core,
            "series_of": series_of, "merged_of": merged_of, "report": report}

def excluded_candidates(train, val, user_index, item_index, series_of):
    """Chỉ số item bị loại khỏi rổ ứng viên của từng người đọc.

    Hai phép loại, đúng mục 7.1 của đề xuất: (1) mục đã có trong **train và
    val** — val phải loại, nếu không thì một mục người ta đã đọc ở kỳ val lại
    được tính là gợi ý đúng ở kỳ test; (2) mục **cùng bộ** với bất kỳ mục đã
    đọc. Phép thứ hai là chỗ đáng tiền: không có nó, một mô hình chỉ cần gợi
    "tập kế tiếp" là có điểm cao, và đó là định nghĩa của gợi ý vô dụng.

    Hàm nằm ở đây (không nằm trong script) để có test: lọc sai ở bước này thì
    mọi con số của nhánh đều sai theo, mà không có gì báo.
    """
    import numpy as np

    siblings = {}
    for media_id, series in series_of.items():
        if series:
            siblings.setdefault(series, []).append(media_id)
    seen = pd.concat([train[["user", "media_id"]], val[["user", "media_id"]]])
    out = {}
    for user, group in seen.groupby("user", sort=False)["media_id"]:
        row = user_index.get(user)
        if row is None:
            continue
        drop = set()
        for media_id in group:
            position = item_index.get(media_id)
            if position is not None:
                drop.add(position)
            series = series_of.get(media_id)
            if series:
                for sibling in siblings.get(series, ()):
                    position = item_index.get(sibling)
                    if position is not None:
                        drop.add(position)
        out[user] = np.fromiter(drop, dtype=np.int32, count=len(drop))
    return out

def reachable_test(test, excluded, item_index):
    """Giữ lại những dòng test mà mô hình **có thể** với tới được.

    Vì sao bắt buộc: mục 7.1 của đề xuất loại "truyện đã đọc và cùng series"
    khỏi *rổ ứng viên*. Nếu chỉ làm đúng câu đó thì một mục test là tập kế tiếp
    của bộ người ta đang đọc sẽ bị loại khỏi ứng viên mà **vẫn nằm trong đáp
    án** — không mô hình nào với tới được, nhưng nó vẫn vào mẫu số của
    Recall/NDCG. Kết quả là mọi mô hình bị trừ điểm như nhau vì một thứ không
    phải lỗi của mô hình, và tỉ lệ bị trừ lại khác nhau giữa hai biến thể gộp
    và không gộp — tức đúng phép so mà nhánh này tồn tại để làm thì bị méo.

    Quy tắc: đáp án phải là giao của đáp án và rổ ứng viên. Trả về cả số dòng
    bị bỏ để con số đó được in ra chứ không nằm im.
    """
    import numpy as np

    positions = test["media_id"].map(item_index)
    out = test.assign(_pos=positions).dropna(subset=["_pos"])
    out["_pos"] = out["_pos"].astype(np.int32)
    dropped_unknown = int(len(test) - len(out))
    if excluded:
        users = np.repeat(np.asarray(list(excluded.keys()), dtype=object),
                          [len(v) for v in excluded.values()])
        pairs = pd.DataFrame({"user": users,
                              "_pos": np.concatenate(list(excluded.values()))})
        pairs["_bad"] = True
        out = out.merge(pairs, on=["user", "_pos"], how="left")
        dropped_excluded = int(out["_bad"].notna().sum())
        out = out[out["_bad"].isna()].drop(columns=["_bad"])
    else:
        dropped_excluded = 0
    return out.drop(columns=["_pos"]), {
        "test_rows_in": int(len(test)),
        "test_rows_kept": int(len(out)),
        "dropped_item_not_in_train": dropped_unknown,
        "dropped_same_series_or_seen": dropped_excluded}
