"""Đọc bộ Goodreads UCSD (Comics & Graphic), xuất đúng hợp đồng của ingest.py.

Đây là **bộ chuyển đổi, không phải hệ thứ hai**. Hai bảng trả về có đúng tên
cột như `ingest.run` của nhánh AniList, nên toàn bộ phần sau — preprocess,
build_features, ALS, hai tháp, eval_saved, measure_product — chạy không sửa
một dòng. Mục đích là để so sánh *phương pháp* trên hai bộ dữ liệu, chứ không
phải so hai cách lập trình.

GIẤY PHÉP: dữ liệu chỉ dùng cho mục đích học thuật, không phát hành lại, không
dùng thương mại. Trích dẫn bắt buộc:
  Mengting Wan, Julian McAuley. "Item Recommendation on Monotonic Behavior
  Chains". RecSys 2018.
  Mengting Wan, Rishabh Misra, Ndapa Nakashole, Julian McAuley. "Fine-Grained
  Spoiler Detection from Large-Scale Review Corpora". ACL 2019.

Những chỗ KHÔNG ánh xạ được, ghi rõ ở đây để không ai đọc lẫn thành "đã có":

  country        -> None. Goodreads KHÔNG có trường xuất xứ. `country_code`
                    trong dữ liệu là nước của *ấn bản* (gần như toàn bộ là US),
                    không phải nước ra đời tác phẩm. Vì vậy mọi chỉ số theo
                    nước (tỉ lệ manhua, khoảng cách hiệu chỉnh theo nước, số
                    nước trong top-10) là **N/A trên nhánh này** — không đoán,
                    không suy từ nhà xuất bản. Đây chính là điều mục 1 của
                    docs/data-sources.md nói, nay kiểm được bằng code.
  progress       -> 0 ở mọi dòng. Goodreads ghi nhận ở mức quyển, không có
                    tiến độ trong quyển. Kéo theo: `catchup` không tính được,
                    `mean_drop_position` rỗng.
  status DROPPED -> không tồn tại. `is_read` là nhị phân, không phân biệt
                    "bỏ dở" với "chưa đọc". Nên drop_rate và pause_rate bằng 0
                    trên toàn bộ danh mục.
  pub_status     -> "FINISHED" cho tất cả. media_id ở đây là một *tác phẩm đã
                    in* nên nó đã xong, khác với manhwa đang ra hàng tuần.

Những chỗ ánh xạ được nhưng đổi đơn vị, nói rõ để không bị hiểu là trùng khớp:

  score          -> rating 1-5 nhân 20 thành thang 100 của AniList. Mốc trung
                    tính 70 của AniList ứng với rating 3,5. rating = 0 nghĩa là
                    CHƯA CHẤM (không phải 0 điểm) nên quy về None, giống đúng
                    cách ingest.py xử lý score = 0 của AniList.
  chapters       -> num_pages. Hai bên đo "tác phẩm dài bao nhiêu" bằng hai
                    đơn vị khác nhau: AniList đếm chương, Goodreads đếm trang.
                    build_pace chuẩn hoá (log + StandardScaler) nên chỉ thứ tự
                    và độ phân tán có ý nghĩa, đơn vị thì không. Dùng num_pages
                    là để nhánh đối chứng được chạy với tín hiệu nó THẬT SỰ có,
                    không phải để nó trông giống AniList.
  tag_ranks      -> count của popular_shelves quy về phần trăm so với kệ nhiều
                    nhất của chính tác phẩm đó, bắt chước rank % của AniList.

Gộp ấn bản (bẫy 4): media_id = `work_id`, KHÔNG phải `book_id`. Một book_id là
một ấn bản (bìa cứng, bìa mềm, omnibus, từng bản dịch). Không gộp thì cùng một
tác phẩm nằm rải ở nhiều hàng của ma trận tương tác và ALS học ra "ai đọc bản
bìa cứng thì thích bản bìa mềm".
"""

import gzip
import json
import re
from array import array
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

BOOKS = "goodreads_books_comics_graphic.json.gz"
INTERACTIONS = "goodreads_interactions_comics_graphic.json.gz"
AUTHORS = "goodreads_book_authors.json.gz"
WORKS = "goodreads_book_works.json.gz"

# Kệ hành chính, không nói gì về nội dung. Mở rộng từ scripts/check_data.py:
# cùng một danh sách thì so sánh với lần kiểm dữ liệu trước mới có nghĩa.
SHELF_STOPLIST = {
    "to-read", "currently-reading", "owned", "default", "books-i-own",
    "favorites", "favourites", "to-buy", "owned-books", "library", "wish-list",
    "all", "books", "my-books", "read", "unread", "have", "wishlist",
    "to-read-fiction", "kindle", "ebook", "ebooks", "audiobook", "audiobooks",
    "borrowed", "library-book", "library-books", "own-it", "i-own",
    "my-library", "shelfari-favorites", "abandoned", "did-not-finish", "dnf",
    "re-read", "rereads", "series", "general", "default-shelf", "calibre",
    "to-get", "maybe", "wanted", "want-to-read", "on-hold", "paused",
}

# Vai trò trong `authors[].role` chỉ người vẽ. Goodreads ghi tự do nên phải so
# khớp theo chuỗi con, giống classify_role của nhánh AniList.
ART_ROLE = ("illustrat", "artist", "art by", "penciller", "penciler", "inker",
            "colorist", "colourist", "cover art", "drawings")
# Vai trò không liên quan tới sáng tác, loại hẳn khỏi v_creator.
NON_CREATIVE = ("translat", "letter", "editor", "edited", "introduction",
                "foreword", "afterword", "adapt", "contributor", "narrat",
                "photograph", "designer", "typeset", "compil", "preface")

# Ảnh bìa: đo thật trên 150 URL ngày 2026-11-10, không suy từ tài liệu.
#   URL ghi trong dataset (hậu tố 'm') : rộng trung vị  98px — DƯỚI ngưỡng báo
#       động 300px của mục 9, tức Gram matrix không đo được độ dày nét hay
#       screentone vì chính thông tin đó đã bị resize xoá mất.
#   Đổi hậu tố 'm' -> 'l'               : rộng trung vị 309px, cao 460px,
#       108/150 ảnh đạt >= 300px. So với AniList extraLarge 460x643.
#   Đo lại trên 2.500 ảnh ĐÃ TẢI THẬT (mẫu ngẫu nhiên seed 0 trong 27.999):
#       rộng trung vị 309px, cao 450px, 69,1% đạt >= 300px rộng, và 0 ảnh có
#       kích thước placeholder 111x148 lọt qua — tức ~45% số pixel của AniList.
#       Con số trên URL (108/150 = 72%) hơi lạc quan so với ảnh tải thật.
# Nên cách sửa mà mục 9 đề xuất ("thử sửa tham số kích thước trong URL") CÓ
# hiệu lực, và nhánh nét vẽ trên Goodreads không chết — đây là chỗ tài liệu cũ
# bi quan hơn dữ liệu.
#
# Riêng ảnh 'nophoto' thì bỏ hẳn, không tải: 37,4% tác phẩm dùng cùng MỘT ảnh
# xám 111x148. Đưa chúng vào mặt nét vẽ sẽ tạo một cụm giả 23.527 tác phẩm có
# vector giống hệt nhau — đúng bẫy 5 của docs/data-sources.md.
COVER_SIZE_PATTERN = re.compile(r"/books/(\d+)m/")


def cover_url(url):
    if not url or "nophoto" in url:
        return None
    return COVER_SIZE_PATTERN.sub(lambda m: f"/books/{m.group(1)}l/", url)


MONTHS = {"Jan": 1, "Feb": 2, "Mar": 3, "Apr": 4, "May": 5, "Jun": 6,
          "Jul": 7, "Aug": 8, "Sep": 9, "Oct": 10, "Nov": 11, "Dec": 12}

# Bẫy 6: ngày do người dùng tự nhập, rác rất nhiều. Goodreads ra đời 2007 và
# bộ dữ liệu đóng băng cuối 2017; ngoài khoảng này là lỗi nhập, bỏ.
YEAR_MIN, YEAR_MAX = 2007, 2018


def _days_from_civil(y, m, d):
    """Số ngày từ 1970-01-01 (thuật toán Howard Hinnant, toàn số nguyên).

    Dùng thay strptime vì phải xử lý ~14,7 triệu mốc thời gian; strptime mất
    khoảng 10 µs mỗi lần, tức hơn hai phút chỉ để đọc ngày.
    """
    y -= m <= 2
    era = (y if y >= 0 else y - 399) // 400
    yoe = y - era * 400
    doy = (153 * (m + (-3 if m > 2 else 9)) + 2) // 5 + d - 1
    doe = yoe * 365 + yoe // 4 - yoe // 100 + doy
    return era * 146097 + doe - 719468


def parse_gr_time(text):
    """'Thu Dec 14 09:06:26 -0800 2017' -> giây unix, hoặc 0 nếu không đọc được.

    Định dạng cố định chiều rộng nên cắt theo vị trí. Có bỏ qua lệch múi giờ
    (trường -0800): sai số tối đa một ngày, không ảnh hưởng tới mốc cắt
    train/test tính theo phân vị trên hơn bảy triệu mốc.
    """
    if not text or len(text) < 30:
        return 0
    month = MONTHS.get(text[4:7])
    if month is None:
        return 0
    try:
        day = int(text[8:10])
        hour, minute, second = int(text[11:13]), int(text[14:16]), int(text[17:19])
        year = int(text[-4:])
    except ValueError:
        return 0
    if not (1990 <= year <= 2030):
        return 0
    return (_days_from_civil(year, month, day) * 86400
            + hour * 3600 + minute * 60 + second)


def _plausible(seconds):
    """Mốc thời gian nằm trong vòng đời thật của Goodreads."""
    if not seconds:
        return None
    year = pd.Timestamp(seconds, unit="s").year
    return seconds if YEAR_MIN <= year <= YEAR_MAX else None


def read_jsonl_gz(path, limit=None):
    with gzip.open(path, "rt", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                return
            line = line.strip()
            if line:
                yield json.loads(line)


def _number(value, cast=float):
    """Goodreads để chuỗi rỗng thay vì null ở gần như mọi trường số."""
    if value is None or value == "":
        return None
    try:
        out = cast(value)
    except (TypeError, ValueError):
        return None
    return out


def classify_role(role):
    """'Illustrator' -> ('art',); vai trống -> ('story',); biên tập -> ()."""
    text = (role or "").strip().lower()
    if not text:
        return ("story",)           # vai trống trên Goodreads = tác giả chính
    if any(word in text for word in NON_CREATIVE):
        return ()
    is_art = any(word in text for word in ART_ROLE)
    is_story = any(word in text for word in ("author", "writer", "text", "story",
                                             "script", "words"))
    if is_art and is_story:
        return ("story", "art")
    if is_art:
        return ("art",)
    return ("story",) if is_story else ()


def load_authors(raw_dir):
    names = {}
    path = Path(raw_dir) / AUTHORS
    if not path.exists():
        return names
    for record in read_jsonl_gz(path):
        names[record["author_id"]] = record.get("name")
    return names


def load_work_years(raw_dir):
    """`original_publication_year` theo work_id — năm ra đời của *tác phẩm*.

    Năm trên từng ấn bản là năm in lại, nên một bộ 1986 tái bản 2015 sẽ trông
    như truyện mới. Trường này của file works mới là năm thật.
    """
    years = {}
    path = Path(raw_dir) / WORKS
    if not path.exists():
        return years
    for record in read_jsonl_gz(path):
        year = _number(record.get("original_publication_year"), int)
        if year and 1800 <= year <= 2018:
            years[record["work_id"]] = year
    return years


def load_media(raw_dir, author_names=None, work_years=None, report=None):
    """Gộp ấn bản theo work_id, trả về một dòng mỗi tác phẩm.

    Ấn bản đại diện là ấn bản được chấm điểm nhiều nhất (bản mà người ta thật
    sự đọc), hoà thì lấy bản in sớm nhất. Các trường đếm được (ratings_count,
    text_reviews_count, popular_shelves) thì CỘNG qua mọi ấn bản, vì tương tác
    cũng đã được gộp theo cùng cách.
    """
    raw_dir = Path(raw_dir)
    author_names = author_names or {}
    work_years = work_years or {}

    primary = {}                     # work_id -> bản ghi ấn bản đại diện
    ratings = Counter()              # work_id -> tổng ratings_count
    reviews = Counter()              # work_id -> tổng text_reviews_count
    shelves = defaultdict(Counter)   # work_id -> {kệ: tổng count}
    authors = defaultdict(dict)      # work_id -> {author_id: vai}
    book_to_work = {}
    n_books = 0
    role_tag = Counter()

    n_bad_id = 0
    for book in read_jsonl_gz(raw_dir / BOOKS):
        n_books += 1
        book_id, work_id = book.get("book_id"), book.get("work_id")
        # media_id phải là số nguyên để trùng kiểu với nhánh AniList (index.json
        # và parquet đều giả định vậy). Goodreads ghi id dạng chuỗi số; dòng
        # nào không phải số thì bỏ và ĐẾM, không im lặng cho qua.
        if not book_id or not work_id or not str(work_id).isdigit():
            n_bad_id += 1
            continue
        work_id = str(work_id)
        book_to_work[book_id] = work_id

        count = _number(book.get("ratings_count"), int) or 0
        ratings[work_id] += count
        reviews[work_id] += _number(book.get("text_reviews_count"), int) or 0

        bucket = shelves[work_id]
        for shelf in book.get("popular_shelves") or []:
            name = (shelf.get("name") or "").strip().lower()
            if name and name not in SHELF_STOPLIST:
                bucket[name] += _number(shelf.get("count"), int) or 0
        # popular_shelves có đuôi dài toàn kệ cá nhân dùng một lần ('calibre-list',
        # tên riêng của một người). Giữ hết thì bảng kệ phình lên hàng triệu ô
        # cho vài chục nghìn tác phẩm, mà máy chỉ còn ~3,4 GB trống. Cắt về 60
        # kệ nhiều nhất khi vượt 240: ta chỉ dùng 40 kệ đầu.
        if len(bucket) > 240:
            shelves[work_id] = Counter(dict(bucket.most_common(60)))

        for entry in book.get("authors") or []:
            author_id = entry.get("author_id")
            if not author_id:
                continue
            kinds = classify_role(entry.get("role"))
            role_tag.update(kinds or ("bo-qua",))
            if kinds:
                merged = set(authors[work_id].get(author_id, ())) | set(kinds)
                authors[work_id][author_id] = tuple(sorted(merged))

        year = _number(book.get("publication_year"), int) or 0
        best = primary.get(work_id)
        if best is None or (count, -year) > (best[0], -best[1]):
            primary[work_id] = (count, year or 9999, book)

    rows = []
    for work_id, (_, _, book) in primary.items():
        shelf_counts = shelves[work_id].most_common(40)
        top = shelf_counts[0][1] if shelf_counts else 0
        tag_names = [name for name, _ in shelf_counts]
        tag_ranks = [round(100.0 * c / top, 1) if top else 0.0
                     for _, c in shelf_counts]

        story_ids, art_ids = [], []
        for author_id, kinds in authors[work_id].items():
            if "story" in kinds:
                story_ids.append(author_id)
            if "art" in kinds:
                art_ids.append(author_id)

        rows.append({
            "media_id": int(work_id),
            "title": (book.get("title_without_series")
                      or book.get("title") or f"#{work_id}"),
            "title_romaji": None,
            # KHÔNG CÓ xuất xứ trên Goodreads — xem docstring đầu file.
            "country": None,
            "format": book.get("format") or None,
            # Một quyển đã in thì đã xong, khác manhwa đang ra hàng tuần.
            "pub_status": "FINISHED",
            # Đổi đơn vị: chương -> trang. Xem docstring đầu file.
            "chapters": _number(book.get("num_pages"), int),
            "volumes": None,
            "start_year": work_years.get(work_id)
                          or _number(book.get("publication_year"), int),
            "end_year": work_years.get(work_id)
                        or _number(book.get("publication_year"), int),
            "genres": [],
            "tag_names": tag_names,
            "tag_ranks": tag_ranks,
            "cover_url": cover_url(book.get("image_url")),
            "staff_story_ids": story_ids,
            "staff_art_ids": art_ids,
            "staff_story_names": [author_names.get(a) for a in story_ids],
            "staff_art_names": [author_names.get(a) for a in art_ids],
            "description": book.get("description") or "",
            "average_score": (lambda r: r * 20.0 if r else None)(
                _number(book.get("average_rating"))),
            "popularity": ratings[work_id],
            "favourites": reviews[work_id],
            "is_adult": False,
        })

    media = pd.DataFrame(rows).drop_duplicates(subset="media_id")
    if report is not None:
        report["books_read"] = n_books
        report["books_bad_id"] = n_bad_id
        report["works"] = len(media)
        report["edition_merge_drop_pct"] = round(
            100.0 * (1 - len(media) / max(n_books, 1)), 1)
        report["role_counts"] = dict(role_tag)
        report["works_with_art_role_pct"] = round(
            100.0 * float((media["staff_art_ids"].str.len() > 0).mean()), 1)
        report["works_with_cover_pct"] = round(
            100.0 * float(media["cover_url"].notna().mean()), 1)
        report["works_placeholder_cover_pct"] = round(
            100.0 * float(media["cover_url"].isna().mean()), 1)
        report["works_with_pages_pct"] = round(
            100.0 * float(media["chapters"].notna().mean()), 1)
        report["median_pages"] = float(
            pd.to_numeric(media["chapters"], errors="coerce").median())
    return media, book_to_work


def load_interactions(raw_dir, book_to_work, min_items=1, max_items_per_user=3000,
                      report=None):
    """Đọc 7,3 triệu tương tác theo luồng, gộp book_id -> work_id.

    Dùng `array` thay vì list Python: 7,3 triệu số nguyên trong list mất khoảng
    280 MB mỗi cột vì mỗi phần tử là một đối tượng int riêng; `array('i')` giữ
    dạng C, còn 29 MB. Máy chạy đồ án này chỉ còn khoảng 3,4 GB trống nên khác
    biệt đó là khác biệt giữa chạy được và bị kill.
    """
    path = Path(raw_dir) / INTERACTIONS
    user_ids = {}
    users = array("i")
    items = array("i")
    read_flag = array("b")
    ratings = array("b")
    added = array("q")
    updated = array("q")
    started = array("q")
    finished = array("q")

    n_lines = n_unknown_book = 0
    for record in read_jsonl_gz(path):
        n_lines += 1
        work_id = book_to_work.get(record.get("book_id"))
        if work_id is None:
            n_unknown_book += 1
            continue
        user = record.get("user_id")
        position = user_ids.get(user)
        if position is None:
            position = user_ids[user] = len(user_ids)
        users.append(position)
        items.append(int(work_id))
        read_flag.append(1 if record.get("is_read") else 0)
        ratings.append(_number(record.get("rating"), int) or 0)
        added.append(parse_gr_time(record.get("date_added")))
        updated.append(parse_gr_time(record.get("date_updated")))
        started.append(parse_gr_time(record.get("started_at")) or 0)
        finished.append(parse_gr_time(record.get("read_at")) or 0)

    # Mã số nội bộ chỉ dùng để dồn vào `array`; cột `user` xuất ra phải là
    # CHUỖI id gốc của Goodreads, đúng kiểu dữ liệu mà nhánh AniList dùng
    # (tên người dùng đã băm, 16 ký tự hex). Hợp đồng cột không chỉ là tên cột
    # mà còn là kiểu của hai khoá: `user` chuỗi, `media_id` số nguyên. Giữ mã
    # số int32 ở đây thì mọi thứ vẫn chạy cho tới bước ghi index.json, nơi
    # json từ chối khoá numpy.int32 — tức là lỗi hiện ra sau một tiếng rưỡi
    # huấn luyện, ở chỗ xa nhất với nguyên nhân.
    # Ánh xạ lại bằng cách lập chỉ mục mảng object: 7,3 triệu con trỏ (58 MB)
    # trỏ vào đúng 342 nghìn chuỗi đã nằm trong `user_ids`, không nhân bản
    # chuỗi. Cách này rẻ gần bằng Categorical mà không đổi kiểu cột.
    user_names = np.asarray(list(user_ids), dtype=object)
    frame = pd.DataFrame({
        "user": user_names[np.frombuffer(users, dtype=np.int32)],
        "media_id": np.frombuffer(items, dtype=np.int32),
        "is_read": np.frombuffer(read_flag, dtype=np.int8).astype(bool),
        "rating": np.frombuffer(ratings, dtype=np.int8),
        "created_at": np.frombuffer(added, dtype=np.int64),
        "updated_at": np.frombuffer(updated, dtype=np.int64),
        "_started": np.frombuffer(started, dtype=np.int64),
        "_finished": np.frombuffer(finished, dtype=np.int64),
    })
    del users, items, read_flag, ratings, added, updated, started, finished

    # is_read nhị phân: đã đọc -> COMPLETED, chưa đọc -> PLANNING. Gộp nhóm
    # chưa đọc vào tương tác dương là đúng sai lầm mà bẫy 2 cảnh báo, và
    # preprocess.POSITIVE đã loại PLANNING sẵn.
    # Kiểu dữ liệu ở đây là chuyện sống còn, không phải chuyện gọn gàng: 7,3
    # triệu dòng mà để `status` dạng chuỗi cố định '<U9' thì riêng cột đó đã
    # 265 MB, còn ba cột tiến độ để int64 thì thêm 176 MB. Máy chạy đồ án này
    # còn khoảng 3,4 GB trống và còn phải sắp xếp cả bảng.
    frame["status"] = pd.Categorical.from_codes(
        frame["is_read"].to_numpy().astype(np.int8),
        categories=["PLANNING", "COMPLETED"])
    # rating = 0 là CHƯA CHẤM, không phải 0 điểm.
    frame["score"] = np.where(frame["rating"] > 0,
                              frame["rating"].astype(np.float32) * 20.0,
                              np.float32("nan")).astype(np.float32)
    zero = np.zeros(len(frame), dtype=np.int8)
    frame["progress"] = zero
    frame["progress_volumes"] = zero
    frame["repeat"] = zero
    frame["recruit_country"] = pd.Series(
        pd.Categorical([None] * len(frame), categories=["JP", "KR", "CN"]),
        index=frame.index)

    # Bẫy 6: lọc ngày vô lý trước khi dùng; trục thời gian chính là date_added
    # (hệ thống ghi) chứ không phải read_at (người dùng khai).
    bad_added = ~frame["created_at"].between(
        int(pd.Timestamp(f"{YEAR_MIN}-01-01").timestamp()),
        int(pd.Timestamp(f"{YEAR_MAX}-12-31").timestamp()))
    if report is not None:
        report["rows_read"] = n_lines
        report["rows_unknown_book"] = n_unknown_book
        report["rows_kept"] = int(len(frame))
        report["users_raw"] = int(len(user_ids))
        report["is_read_pct"] = round(100.0 * float(frame["is_read"].mean()), 1)
        report["rating_zero_pct"] = round(
            100.0 * float((frame["rating"] == 0).mean()), 1)
        report["bad_date_added_pct"] = round(100.0 * float(bad_added.mean()), 2)
        has_read_at = frame["_finished"] > 0
        report["has_read_at_pct"] = round(100.0 * float(has_read_at.mean()), 1)
        report["has_started_at_pct"] = round(
            100.0 * float((frame["_started"] > 0).mean()), 1)
        both = has_read_at & (frame["_started"] > 0)
        report["negative_duration_pct"] = round(100.0 * float(
            (frame.loc[both, "_finished"] < frame.loc[both, "_started"]).mean()
        ), 2) if both.any() else None

    frame = frame[~bad_added].copy()
    frame["started_at"] = pd.to_datetime(
        frame["_started"].where(frame["_started"] > 0), unit="s", errors="coerce")
    frame["completed_at"] = pd.to_datetime(
        frame["_finished"].where(frame["_finished"] > 0), unit="s", errors="coerce")
    frame = frame.drop(columns=["_started", "_finished", "is_read", "rating"])

    # Cùng cách chống đánh dấu hàng loạt như nhánh AniList, cùng ngưỡng.
    # Chỉ sắp xếp khi THẬT CÓ trùng: sort_values trên 7,3 triệu dòng tạo thêm
    # một bản copy cả bảng, mà sau khi gộp ấn bản thì trùng là chuyện ít gặp
    # (một người phải đọc hai ấn bản của cùng một tác phẩm).
    if frame.duplicated(subset=["user", "media_id"]).any():
        frame = (frame.sort_values("updated_at")
                 .drop_duplicates(subset=["user", "media_id"], keep="last"))
    sizes = frame.groupby("user")["media_id"].transform("size")
    frame = frame[(sizes <= max_items_per_user) & (sizes >= min_items)]

    if report is not None:
        report["rows_after_dedupe"] = int(len(frame))
        report["users_after_dedupe"] = int(frame["user"].nunique())
        report["items_after_dedupe"] = int(frame["media_id"].nunique())
    return frame.reset_index(drop=True)


def run(raw_dir, out_dir):
    raw_dir, out_dir = Path(raw_dir), Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    report = {}

    author_names = load_authors(raw_dir)
    work_years = load_work_years(raw_dir)
    media, book_to_work = load_media(raw_dir, author_names, work_years, report)
    interactions = load_interactions(raw_dir, book_to_work, report=report)

    # Chỉ giữ metadata của tác phẩm thật sự có tương tác.
    media = media[media["media_id"].isin(set(interactions["media_id"]))]
    media = media.reset_index(drop=True)

    interactions.to_parquet(out_dir / "interactions.parquet", index=False)
    media.to_parquet(out_dir / "media.parquet", index=False)
    (out_dir / "ingest_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    return interactions, media
