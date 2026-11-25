"""Kiểm chứng bộ Goodreads Comics & Graphic trước khi chốt nguồn dữ liệu.

Trả lời các câu hỏi trong bảng mục 9 của docs/data-sources.md.
Không giả định schema: in ra key thật tìm thấy trong dữ liệu.

    python scripts/check_data.py --data-dir data/external/goodreads
    python scripts/check_data.py --data-dir data/external/goodreads --check-images 300

Bảng mục 9 đã được điền bằng dữ liệu thật, xem docs/goodreads-arm.md mục 2.
Script này đo ở mức `book_id` (từng ấn bản); bộ chuyển đổi
`src/truyen_reco/data/ingest_goodreads.py` đo lại ở mức `work_id` (tác phẩm đã
gộp ấn bản) và ghi ra `ingest_report.json`.
"""

import argparse
import gzip
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

BOOKS = "goodreads_books_comics_graphic.json.gz"
INTERACTIONS = "goodreads_interactions_comics_graphic.json.gz"
REVIEWS = "goodreads_reviews_comics_graphic.json.gz"

SHELF_STOPLIST = {
    "to-read", "currently-reading", "owned", "default", "books-i-own",
    "favorites", "to-buy", "owned-books", "library", "wish-list", "all",
}


def read_jsonl(path, limit=None):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def section(title):
    print(f"\n{'=' * 70}\n{title}\n{'=' * 70}")


def check_schema(data_dir):
    section("1. SCHEMA THẬT (key của bản ghi đầu tiên)")
    for name in (BOOKS, INTERACTIONS, REVIEWS):
        path = data_dir / name
        if not path.exists():
            print(f"[THIẾU] {name}")
            continue
        record = next(read_jsonl(path, limit=1), None)
        if record is None:
            print(f"[RỖNG] {name}")
            continue
        print(f"\n{name}\n  {sorted(record.keys())}")
        for key in ("authors", "popular_shelves", "series"):
            if key in record:
                print(f"  mẫu {key}: {str(record[key])[:200]}")


def check_interactions(data_dir, limit):
    section("2. TƯƠNG TÁC: rating, chuỗi hành vi, ngày tháng")
    path = data_dir / INTERACTIONS
    if not path.exists():
        print(f"[THIẾU] {INTERACTIONS}")
        return

    ratings = Counter()
    behavior = Counter()
    users, books = set(), set()
    has_read_at = bad_read_at = total = 0
    has_started_at = 0

    for rec in read_jsonl(path, limit=limit):
        total += 1
        rating = rec.get("rating", 0)
        is_read = bool(rec.get("is_read", False))
        ratings[rating] += 1
        behavior[(is_read, rating == 0)] += 1
        users.add(rec.get("user_id"))
        books.add(rec.get("book_id"))

        read_at = rec.get("read_at")
        if read_at:
            has_read_at += 1
            year = str(read_at)[-4:] if str(read_at)[-4:].isdigit() else str(read_at)[:4]
            if not year.isdigit() or not (2007 <= int(year) <= 2017):
                bad_read_at += 1
        if rec.get("started_at"):
            has_started_at += 1

    print(f"Số bản ghi đã đọc: {total:,}")
    print(f"User riêng biệt: {len(users):,} | Item riêng biệt: {len(books):,}")
    if users and books:
        print(f"Độ thưa: {total / (len(users) * len(books)):.8f}")

    print("\nPhân bố rating (0 = CHƯA CHẤM ĐIỂM, không phải điểm 0):")
    for rating in sorted(ratings):
        pct = 100 * ratings[rating] / total
        flag = "  <-- ngưỡng báo động nếu > 40%" if rating == 0 and pct > 40 else ""
        print(f"  {rating}: {ratings[rating]:>10,} ({pct:5.1f}%){flag}")

    print("\nBa mức chuỗi hành vi (bẫy 2):")
    labels = {
        (False, True): "chưa đọc, chưa chấm  -> chỉ 'muốn đọc', tín hiệu YẾU",
        (True, True): "đã đọc, chưa chấm    -> implicit dương, MẠNH",
        (True, False): "đã đọc, có chấm      -> explicit",
        (False, False): "chưa đọc mà có chấm  -> bất thường, cần xem",
    }
    for key, label in labels.items():
        count = behavior.get(key, 0)
        print(f"  {label}: {count:>10,} ({100 * count / total:5.1f}%)")

    print(f"\nCó read_at: {100 * has_read_at / total:.1f}%"
          f" | trong đó ngày vô lý: {100 * bad_read_at / max(has_read_at, 1):.1f}%")
    print(f"Có started_at: {100 * has_started_at / total:.1f}%")
    print("  -> read_at dùng được < 20% thì chỉ dùng date_added làm trục thời gian")


def check_books(data_dir, limit):
    section("3. ITEM: work_id, series, tác giả, shelves")
    path = data_dir / BOOKS
    if not path.exists():
        print(f"[THIẾU] {BOOKS}")
        return

    book_ids, work_ids = set(), set()
    in_series = roles = with_illustrator = total = 0
    no_image = 0
    shelves = Counter()
    role_names = Counter()

    for rec in read_jsonl(path, limit=limit):
        total += 1
        book_ids.add(rec.get("book_id"))
        if rec.get("work_id"):
            work_ids.add(rec.get("work_id"))
        if rec.get("series"):
            in_series += 1

        authors = rec.get("authors") or []
        has_illustrator = False
        for author in authors:
            role = (author.get("role") or "").strip()
            if role:
                roles += 1
                role_names[role] += 1
                if "illustrat" in role.lower() or "art" in role.lower():
                    has_illustrator = True
        if has_illustrator:
            with_illustrator += 1

        image_url = rec.get("image_url") or ""
        if "nophoto" in image_url or not image_url:
            no_image += 1

        for shelf in (rec.get("popular_shelves") or []):
            name = (shelf.get("name") or "").lower()
            if name and name not in SHELF_STOPLIST:
                try:
                    shelves[name] += int(shelf.get("count", 0))
                except (TypeError, ValueError):
                    pass

    print(f"Số bản ghi đã đọc: {total:,}")
    print(f"book_id riêng biệt: {len(book_ids):,}")
    print(f"work_id riêng biệt: {len(work_ids):,}")
    if book_ids and work_ids:
        reduction = 100 * (1 - len(work_ids) / len(book_ids))
        flag = "  <-- BẮT BUỘC gộp work_id" if reduction > 30 else ""
        print(f"Gộp theo work_id giảm: {reduction:.1f}%{flag}")

    print(f"\nThuộc series: {100 * in_series / total:.1f}%"
          f"  -> càng cao thì quyết định ở bẫy 4 càng cấp thiết")
    print(f"Có ghi vai trò tác giả: {100 * with_illustrator / total:.1f}% item có Illustrator/Artist")
    print("  -> dưới 20% thì không dựa được vào Goodreads cho ý 'họa sĩ <-> nét vẽ'")
    if role_names:
        print(f"  Các vai trò gặp: {dict(role_names.most_common(10))}")

    print(f"\nKhông có ảnh bìa (nophoto/rỗng): {100 * no_image / total:.1f}%")

    print("\n30 shelf phổ biến nhất sau stoplist (để dựng stoplist đầy đủ ở bẫy 7):")
    for name, count in shelves.most_common(30):
        marker = "  <-- tín hiệu NHỊP" if name in {"dnf", "abandoned", "couldnt-finish"} else ""
        print(f"  {name:<30} {count:>9,}{marker}")


def check_images(data_dir, sample_size, limit):
    section(f"4. ẢNH BÌA: thử {sample_size} URL (quyết định nhánh nét vẽ)")
    try:
        import io
        import urllib.request

        from PIL import Image
    except ImportError:
        print("[BỎ QUA] cần Pillow: pip install pillow")
        return

    path = data_dir / BOOKS
    if not path.exists():
        print(f"[THIẾU] {BOOKS}")
        return

    urls = [rec.get("image_url") for rec in read_jsonl(path, limit=limit)
            if rec.get("image_url")]
    if not urls:
        print("Không tìm thấy image_url nào.")
        return

    random.seed(42)
    sample = random.sample(urls, min(sample_size, len(urls)))

    ok = failed = 0
    widths = []
    hashes = Counter()

    for url in sample:
        try:
            request = urllib.request.Request(url, headers={"User-Agent": "academic-research"})
            with urllib.request.urlopen(request, timeout=10) as response:
                raw = response.read()
            image = Image.open(io.BytesIO(raw))
            widths.append(image.width)
            hashes[hashlib.md5(raw).hexdigest()] += 1
            ok += 1
        except Exception:
            failed += 1

    total = ok + failed
    print(f"Tải được: {ok}/{total} ({100 * ok / total:.1f}%)")
    if ok < 0.7 * total:
        print("  <-- NGƯỠNG BÁO ĐỘNG: dưới 70%, nhánh nét vẽ lung lay")

    if widths:
        widths.sort()
        median = widths[len(widths) // 2]
        print(f"\nChiều rộng ảnh: nhỏ nhất {widths[0]}, trung vị {median}, lớn nhất {widths[-1]}")
        if median < 300:
            print("  <-- NGƯỠNG BÁO ĐỘNG: trung vị dưới 300px.")
            print("      Gram matrix KHÔNG đo được độ dày nét và screentone ở kích thước này.")
            print("      Thử sửa tham số kích thước trong URL để lấy bản lớn hơn,")
            print("      nếu không thì phải cắt nhánh nét vẽ khỏi phạm vi.")

    duplicates = sum(count for count in hashes.values() if count > 1)
    if ok:
        pct = 100 * duplicates / ok
        print(f"\nẢnh trùng hệt nhau (nghi placeholder): {pct:.1f}%")
        if pct > 15:
            print("  <-- cần bước lọc placeholder riêng")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/external/goodreads", type=Path)
    parser.add_argument("--limit", type=int, default=200_000,
                        help="Số bản ghi đọc mỗi file (None = toàn bộ, rất chậm)")
    parser.add_argument("--check-images", type=int, default=0,
                        help="Số URL ảnh bìa đem thử tải, 0 = bỏ qua")
    args = parser.parse_args()

    if not args.data_dir.exists():
        raise SystemExit(f"Không thấy thư mục {args.data_dir}")

    check_schema(args.data_dir)
    check_interactions(args.data_dir, args.limit)
    check_books(args.data_dir, args.limit)
    if args.check_images:
        check_images(args.data_dir, args.check_images, args.limit)

    print("\nĐiền kết quả vào bảng mục 9 của docs/data-sources.md trước khi chốt.")


if __name__ == "__main__":
    main()
