"""Bản đồ tác phẩm -> bộ truyện (series), cho nhánh "hướng ban đầu".

Vì sao cần, và vì sao nó KHÔNG trùng với phép gộp đã có: tầng đọc dữ liệu hiện
tại gộp **ấn bản** theo `work_id` (89.411 book_id -> 62.944 work_id, rơi 29,6%
dòng trùng). Phép gộp đó xử lý chuyện "cùng một quyển in lại nhiều lần". Nó
*không* xử lý chuyện khác hẳn mà đề xuất đồ án nêu ở mục 3.1 bước 4: **mỗi tập
của một bộ dài là một bản ghi riêng**. Naruto Vol.1…72 vẫn là 72 tác phẩm khác
nhau sau khi gộp ấn bản, nên một hệ gợi ý học từ dữ liệu đó sẽ gợi "Naruto
Vol.2" cho người vừa đọc "Naruto Vol.1" — chỉ số đánh giá cao mà gợi ý vô dụng.

Script này dựng `media_id (work_id) -> series_id` từ trường `series` của file
sách thô, rồi nối thêm tên bộ từ `goodreads_book_series.json.gz`.

    python scripts/build_series_map.py

Hai chỗ phải quyết, ghi rõ thay vì để mặc định ngầm:

1. **Một tác phẩm có thể thuộc nhiều bộ** (bản lẻ và bản omnibus "3-in-1" là hai
   series_id khác nhau). Chọn theo số phiếu: series nào được nhiều ấn bản của
   tác phẩm đó khai nhất; hoà thì lấy series_id nhỏ nhất cho tái lập được.
2. **Tác phẩm không thuộc bộ nào thì tự nó là một mục** — không nhóm chúng vào
   một rổ "không series", vì làm thế là trộn những truyện chẳng liên quan.
"""

import argparse
import gzip
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

sys.stdout.reconfigure(encoding="utf-8")


def read_series_titles(path):
    """series_id -> (tên bộ, số tác phẩm trong bộ theo Goodreads toàn bộ)."""
    titles = {}
    if not path.exists():
        return titles
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            series_id = str(row.get("series_id") or "")
            if series_id.isdigit():
                titles[int(series_id)] = (
                    row.get("title") or "",
                    int(row.get("series_works_count") or 0))
    return titles


def build(books_path, series_path):
    votes = defaultdict(Counter)        # work_id -> {series_id: số ấn bản khai}
    work_title = {}
    rows_read = multi = 0
    with gzip.open(books_path, "rt", encoding="utf-8") as handle:
        for line in handle:
            try:
                book = json.loads(line)
            except json.JSONDecodeError:
                continue
            rows_read += 1
            work_id = str(book.get("work_id") or "")
            if not work_id.isdigit():
                continue
            work_id = int(work_id)
            work_title.setdefault(work_id, book.get("title_without_series")
                                  or book.get("title") or "")
            series = book.get("series") or []
            if isinstance(series, str):
                series = [series]
            ids = {int(s) for s in series if str(s).isdigit()}
            if len(ids) > 1:
                multi += 1
            for series_id in ids:
                votes[work_id][series_id] += 1

    titles = read_series_titles(series_path)
    records = []
    for work_id, counter in votes.items():
        if not counter:
            continue
        best = max(counter.items(), key=lambda kv: (kv[1], -kv[0]))[0]
        name, works_count = titles.get(best, ("", 0))
        records.append({"media_id": work_id, "series_id": best,
                        "series_title": name,
                        "series_works_count_global": works_count,
                        "series_candidates": len(counter)})
    frame = pd.DataFrame.from_records(records)
    return frame, {"books_read": rows_read, "works_seen": len(work_title),
                   "books_multi_series": multi}, work_title


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-dir", default="data/external/goodreads", type=Path)
    parser.add_argument("--out-dir", default="data/processed_gr", type=Path)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    started = time.monotonic()
    frame, report, work_title = build(
        args.raw_dir / "goodreads_books_comics_graphic.json.gz",
        args.raw_dir / "goodreads_book_series.json.gz")
    print(f"đọc {report['books_read']:,} bản ghi sách "
          f"({time.monotonic() - started:.0f}s)")
    print(f"  tác phẩm (work_id) thấy được : {report['works_seen']:,}")
    print(f"  tác phẩm có khai bộ          : {len(frame):,} "
          f"({100 * len(frame) / max(report['works_seen'], 1):.1f}%)")
    print(f"  bản ghi khai NHIỀU bộ        : {report['books_multi_series']:,}")

    if frame.empty:
        sys.exit("không dựng được bản đồ series — kiểm lại trường `series`")

    sizes = frame.groupby("series_id")["media_id"].size()
    print(f"\n  số bộ xuất hiện trong tập comics: {len(sizes):,}")
    print(f"  số tác phẩm mỗi bộ (trong tập này): trung vị {sizes.median():.0f}"
          f" | p90 {sizes.quantile(0.9):.0f} | tối đa {sizes.max():,}")
    top = sizes.sort_values(ascending=False).head(8)
    print("\n  tám bộ gom nhiều tác phẩm nhất — đây chính là chỗ 'gợi tập 2 cho")
    print("  người vừa đọc tập 1' xảy ra nếu không gộp:")
    for series_id, count in top.items():
        name = frame.loc[frame["series_id"] == series_id, "series_title"].iloc[0]
        print(f"    {count:>5} tác phẩm  {str(name)[:52]}")

    # Số mục sau khi gộp: mỗi bộ thành 1 mục, tác phẩm lẻ giữ nguyên.
    in_series = len(frame)
    alone = report["works_seen"] - in_series
    print(f"\n  nếu gộp theo bộ: {report['works_seen']:,} tác phẩm -> "
          f"{len(sizes) + alone:,} mục ({len(sizes):,} bộ + {alone:,} lẻ)"
          f" — giảm {100 * (1 - (len(sizes) + alone) / report['works_seen']):.1f}%")

    path = args.out_dir / "series_map.parquet"
    frame.to_parquet(path, index=False)
    report_path = args.out_dir / "series_report.json"
    report.update({"series_in_subset": int(len(sizes)),
                   "works_in_series": int(in_series),
                   "works_alone": int(alone),
                   "items_after_merge": int(len(sizes) + alone)})
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False),
                           encoding="utf-8")
    print(f"\n-> {path}")
    print(f"-> {report_path}")


if __name__ == "__main__":
    main()
