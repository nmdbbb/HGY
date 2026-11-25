"""Bộ chuyển đổi Goodreads phải trả về ĐÚNG hợp đồng của ingest AniList.

Test quan trọng nhất trong file này là `test_hai_nguon_cho_cung_bo_cot`: nếu nó
đỏ thì nhánh đối chứng không còn chạy qua cùng pipeline, và mọi so sánh giữa
hai bộ dữ liệu mất nghĩa — chênh lệch kết quả sẽ không quy được về dữ liệu nữa
vì code đã khác nhau.
"""

import gzip
import json

import numpy as np
import pandas as pd
import pytest

from truyen_reco.data import ingest, ingest_goodreads, preprocess
from truyen_reco.data.ingest_goodreads import (
    _days_from_civil, classify_role, load_interactions, load_media,
    parse_gr_time)


def write_gz(path, records):
    with gzip.open(path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")


def book(book_id, work_id, **kwargs):
    base = {
        "book_id": book_id, "work_id": work_id,
        "title": f"Bo {work_id} (Series, #1)",
        "title_without_series": f"Bo {work_id}",
        "ratings_count": "10", "text_reviews_count": "2",
        "average_rating": "4.00", "num_pages": "200",
        "publication_year": "2015", "image_url": "http://x/y.jpg",
        "popular_shelves": [{"count": "50", "name": "manga"},
                            {"count": "40", "name": "to-read"}],
        "authors": [{"author_id": "a1", "role": ""}],
        "description": "mo ta", "format": "Paperback",
    }
    base.update(kwargs)
    return base


def interaction(user, book_id, is_read=True, rating=4,
                added="Thu Dec 14 09:06:26 -0800 2015", **kwargs):
    base = {"user_id": user, "book_id": book_id, "is_read": is_read,
            "rating": rating, "date_added": added, "date_updated": added,
            "read_at": "", "started_at": ""}
    base.update(kwargs)
    return base


# ---------------------------------------------------------------- ngày tháng

def test_doi_ngay_goodreads_ra_giay_unix():
    # Bỏ qua lệch múi giờ là có chủ ý (xem docstring parse_gr_time), nên mốc
    # kỳ vọng tính theo UTC của chính các số trong chuỗi.
    got = parse_gr_time("Thu Dec 14 09:06:26 -0800 2017")
    assert got == int(pd.Timestamp("2017-12-14 09:06:26", tz="UTC").timestamp())


def test_days_from_civil_khop_moc_epoch():
    assert _days_from_civil(1970, 1, 1) == 0
    assert _days_from_civil(2000, 3, 1) == 11017
    assert _days_from_civil(1969, 12, 31) == -1


@pytest.mark.parametrize("bad", ["", None, "rac", "Thu Xyz 14 09:06:26 -0800 2017",
                                 "Thu Dec 14 09:06:26 -0800 1000"])
def test_ngay_rac_tra_ve_0_chu_khong_nem_loi(bad):
    # Hơn bảy triệu dòng do người dùng tự nhập: một dòng rác không được làm
    # sập cả lần đọc.
    assert parse_gr_time(bad) == 0


# ------------------------------------------------------------------ vai staff

@pytest.mark.parametrize("role,expected", [
    ("", ("story",)),               # vai trống trên Goodreads = tác giả chính
    ("Illustrator", ("art",)),
    ("Author, Illustrator", ("story", "art")),
    ("Translator", ()),             # không sáng tác, loại khỏi v_creator
    ("Editor", ()),
    ("Colorist", ("art",)),
    ("Writer", ("story",)),
])
def test_phan_vai_tac_gia_hoa_si(role, expected):
    assert tuple(sorted(classify_role(role))) == tuple(sorted(expected))


# ------------------------------------------------------------- gộp ấn bản

def test_gop_an_ban_theo_work_id(tmp_path):
    """Ba book_id của cùng một work phải thành MỘT dòng, số đếm thì cộng lại."""
    write_gz(tmp_path / ingest_goodreads.BOOKS, [
        book("b1", "101", ratings_count="100", publication_year="2010"),
        book("b2", "101", ratings_count="500", publication_year="2016",
             title_without_series="Ban bia mem"),
        book("b3", "101", ratings_count="20", publication_year="2018"),
        book("b9", "102", ratings_count="7"),
    ])
    report = {}
    media, book_to_work = load_media(tmp_path, report=report)

    assert len(media) == 2
    assert book_to_work == {"b1": "101", "b2": "101", "b3": "101", "b9": "102"}
    w1 = media[media["media_id"] == 101].iloc[0]
    # Ấn bản đại diện là bản được chấm nhiều nhất, không phải bản đầu tiên.
    assert w1["title"] == "Ban bia mem"
    # ratings_count cộng qua mọi ấn bản vì tương tác cũng được gộp như vậy.
    assert w1["popularity"] == 620
    assert report["edition_merge_drop_pct"] == 50.0


def test_ke_sach_cong_qua_cac_an_ban_va_bo_ke_hanh_chinh(tmp_path):
    write_gz(tmp_path / ingest_goodreads.BOOKS, [
        book("b1", "101", popular_shelves=[{"count": "30", "name": "manga"},
                                          {"count": "90", "name": "to-read"}]),
        book("b2", "101", popular_shelves=[{"count": "10", "name": "manga"},
                                          {"count": "20", "name": "shounen"}]),
    ])
    media, _ = load_media(tmp_path)
    row = media.iloc[0]
    tags = dict(zip(row["tag_names"], row["tag_ranks"]))
    assert "to-read" not in tags                 # kệ hành chính, đã loại
    assert set(tags) == {"manga", "shounen"}
    # rank là phần trăm so với kệ nhiều nhất của chính tác phẩm đó: 40 và 20.
    assert tags["manga"] == 100.0
    assert tags["shounen"] == 50.0


def test_khong_doan_xuat_xu(tmp_path):
    """Goodreads không có trường xuất xứ, nên cột country phải rỗng hẳn.

    Suy xuất xứ từ nhà xuất bản hay từ kệ 'manga' là heuristic có lỗi và không
    tách được manhwa khỏi manhua; để None là cách duy nhất không nói sai.
    """
    write_gz(tmp_path / ingest_goodreads.BOOKS, [
        book("b1", "101", country_code="US", publisher="VIZ Media"),
    ])
    media, _ = load_media(tmp_path)
    assert media["country"].isna().all()


# -------------------------------------------------------------- tương tác

def test_is_read_nhi_phan_thanh_trang_thai_va_diem(tmp_path):
    write_gz(tmp_path / ingest_goodreads.BOOKS, [book("b1", "101"),
                                                 book("b2", "102")])
    write_gz(tmp_path / ingest_goodreads.INTERACTIONS, [
        interaction("u1", "b1", is_read=True, rating=5),
        interaction("u1", "b2", is_read=False, rating=0),
    ])
    media, book_to_work = load_media(tmp_path)
    frame = load_interactions(tmp_path, book_to_work)

    assert set(frame["status"]) == {"COMPLETED", "PLANNING"}
    read = frame[frame["status"] == "COMPLETED"].iloc[0]
    want = frame[frame["status"] == "PLANNING"].iloc[0]
    # rating 5 -> 100 trên thang AniList; rating 0 là CHƯA CHẤM -> None.
    assert read["score"] == 100.0
    assert pd.isna(want["score"])
    # Chưa đọc là "muốn đọc", preprocess phải loại khỏi tương tác dương.
    labelled = preprocess.label(frame, media=media)
    assert labelled.loc[labelled["status"] == "PLANNING", "signal"].eq(
        "wishlist").all()


def test_tien_do_trong_quyen_khong_ton_tai(tmp_path):
    """progress = 0 ở mọi dòng, nên catchup không được tính ra số nào."""
    write_gz(tmp_path / ingest_goodreads.BOOKS, [book("b1", "101")])
    write_gz(tmp_path / ingest_goodreads.INTERACTIONS,
             [interaction(f"u{i}", "b1") for i in range(3)])
    media, book_to_work = load_media(tmp_path)
    frame = load_interactions(tmp_path, book_to_work)

    assert (frame["progress"] == 0).all()
    labelled = preprocess.label(frame, media=media)
    assert labelled["catchup"].isna().all()


def test_bo_ngay_ngoai_vong_doi_goodreads(tmp_path):
    write_gz(tmp_path / ingest_goodreads.BOOKS, [book("b1", "101"),
                                                 book("b2", "102")])
    write_gz(tmp_path / ingest_goodreads.INTERACTIONS, [
        interaction("u1", "b1", added="Thu Dec 14 09:06:26 -0800 2015"),
        interaction("u1", "b2", added="Thu Dec 14 09:06:26 -0800 1999"),
    ])
    media, book_to_work = load_media(tmp_path)
    report = {}
    frame = load_interactions(tmp_path, book_to_work, report=report)

    assert len(frame) == 1
    assert report["bad_date_added_pct"] == 50.0


def test_bo_dong_tro_tới_book_id_khong_co_metadata(tmp_path):
    write_gz(tmp_path / ingest_goodreads.BOOKS, [book("b1", "101")])
    write_gz(tmp_path / ingest_goodreads.INTERACTIONS, [
        interaction("u1", "b1"),
        interaction("u1", "b_khong_ton_tai"),
    ])
    media, book_to_work = load_media(tmp_path)
    report = {}
    frame = load_interactions(tmp_path, book_to_work, report=report)
    assert len(frame) == 1
    assert report["rows_unknown_book"] == 1


def test_gop_tuong_tac_trung_sau_khi_gop_an_ban(tmp_path):
    """Đọc hai ấn bản của cùng một tác phẩm chỉ còn một dòng.

    Không gộp thì ALS học ra 'ai đọc bản bìa cứng cũng thích bản bìa mềm', và
    k-core đếm sai vì một người trông như đã đọc hai bộ.
    """
    write_gz(tmp_path / ingest_goodreads.BOOKS, [book("b1", "101"),
                                                 book("b2", "101")])
    write_gz(tmp_path / ingest_goodreads.INTERACTIONS, [
        interaction("u1", "b1", rating=3, added="Thu Dec 14 09:06:26 -0800 2014"),
        interaction("u1", "b2", rating=5, added="Thu Dec 14 09:06:26 -0800 2016"),
    ])
    media, book_to_work = load_media(tmp_path)
    frame = load_interactions(tmp_path, book_to_work)

    assert len(frame) == 1
    # Giữ bản cập nhật gần nhất, giống dedupe của nhánh AniList.
    assert frame.iloc[0]["score"] == 100.0


# ------------------------------------------------------ hợp đồng giữa hai nguồn

def test_hai_nguon_cho_cung_bo_cot(tmp_path):
    """Cột của bộ chuyển đổi Goodreads phải trùng cột của ingest AniList.

    Đây là điều kiện để nhánh đối chứng chạy qua cùng preprocess, cùng
    build_features, cùng ALS, cùng hai tháp. Thiếu một cột là pipeline sau sẽ
    đi nhánh khác, và lúc đó chênh lệch kết quả không còn quy được về dữ liệu.
    """
    anilist = tmp_path / "anilist"
    anilist.mkdir()
    (anilist / "lists.jsonl").write_text(json.dumps({
        "user": "u1", "source": "JP",
        "entries": [{"mediaId": 1, "status": "COMPLETED", "score": 80,
                     "progress": 10, "progressVolumes": 1, "repeat": 0,
                     "startedAt": {"year": 2020, "month": 1, "day": 1},
                     "completedAt": {"year": 2020, "month": 2, "day": 1},
                     "createdAt": 1600000000, "updatedAt": 1600000001}],
    }) + "\n", encoding="utf-8")
    (anilist / "media.jsonl").write_text(json.dumps({
        "id": 1, "title": {"english": "A", "romaji": "A"},
        "countryOfOrigin": "JP", "format": "MANGA", "status": "FINISHED",
        "chapters": 10, "volumes": 1,
        "startDate": {"year": 2019}, "endDate": {"year": 2020},
        "genres": ["Action"], "tags": [{"name": "t", "rank": 80}],
        "coverImage": {"extraLarge": "http://x"},
        "staff": {"edges": [{"role": "Story & Art",
                             "node": {"id": 7, "name": {"full": "N"}}}]},
        "description": "d", "averageScore": 80, "popularity": 5,
        "favourites": 1, "isAdult": False,
    }) + "\n", encoding="utf-8")

    gr = tmp_path / "gr"
    gr.mkdir()
    write_gz(gr / ingest_goodreads.BOOKS, [book("b1", "101")])
    write_gz(gr / ingest_goodreads.INTERACTIONS, [interaction("u1", "b1")])

    a_inter, a_media = ingest.run(anilist, tmp_path / "out_a")
    g_inter, g_media = ingest_goodreads.run(gr, tmp_path / "out_g")

    assert set(a_media.columns) == set(g_media.columns), (
        "media: lệch cột "
        f"{set(a_media.columns) ^ set(g_media.columns)}")
    assert set(a_inter.columns) == set(g_inter.columns), (
        "interactions: lệch cột "
        f"{set(a_inter.columns) ^ set(g_inter.columns)}")

    # Tên cột trùng nhau vẫn chưa đủ: hai khoá `user` và `media_id` còn phải
    # TRÙNG KIỂU. Lần đầu dựng nhánh này, `user` của Goodreads là mã int32 nên
    # cả pipeline chạy hết — ingest, đặc trưng, ALS — rồi mới chết ở dòng ghi
    # index.json vì json không nhận khoá numpy.int32. Chi phí của việc kiểm tra
    # kiểu ở đây muộn một bậc: một tiếng rưỡi huấn luyện bị bỏ.
    for column in ("user", "media_id"):
        assert a_inter[column].dtype.kind == g_inter[column].dtype.kind, (
            f"interactions['{column}']: AniList {a_inter[column].dtype} vs "
            f"Goodreads {g_inter[column].dtype} — khoá phải cùng kiểu, "
            "nếu không thì bước lưu artifact và bước tra cứu khi phục vụ "
            "sẽ xử sự khác nhau giữa hai nhánh")
    assert g_media["media_id"].dtype.kind == a_media["media_id"].dtype.kind
    assert isinstance(g_inter["user"].iloc[0], str), (
        "user phải là chuỗi id gốc, không phải mã số nội bộ: khoá của "
        "user_index đi vào JSON")


def test_pipeline_sau_chay_duoc_tren_du_lieu_goodreads(tmp_path):
    """preprocess và build_features phải chạy khi country rỗng và progress = 0.

    Cột xuất xứ rỗng và tiến độ bằng 0 là hai thứ nhánh AniList luôn có; nếu
    phần sau ngầm dựa vào chúng thì phải đỏ ở đây, không phải đỏ sau ba tiếng
    huấn luyện.
    """
    from truyen_reco.features import build_features

    # Mỗi tác phẩm có kệ khác nhau: dữ liệu thật có hàng nghìn kệ, nếu fixture
    # chỉ có một kệ thì SVD của mặt theme không chạy được vì chỉ còn 1 chiều.
    write_gz(tmp_path / ingest_goodreads.BOOKS, [
        book(f"b{i}", f"{100 + i}",
             popular_shelves=[{"count": "50", "name": f"ke-{i % 4}"},
                              {"count": "30", "name": f"ke-{(i + 1) % 4}"},
                              {"count": "90", "name": "to-read"}],
             authors=[{"author_id": f"a{i % 3}", "role": ""},
                      {"author_id": f"v{i % 2}", "role": "Illustrator"}])
        for i in range(12)])
    write_gz(tmp_path / ingest_goodreads.INTERACTIONS, [
        interaction(f"u{u}", f"b{i}",
                    added=f"Thu Dec 1{u % 10} 09:06:26 -0800 201{(i % 3) + 4}")
        for u in range(8) for i in range(12)
    ])
    media, book_to_work = load_media(tmp_path)
    frame = load_interactions(tmp_path, book_to_work)

    prepared = preprocess.run(frame, media=media, min_user=2, min_item=2)
    assert len(prepared["train"]) > 0
    assert len(prepared["core"]) > 0

    built = build_features.build_all(media, prepared["train"])
    for name in ("theme", "creator", "pace"):
        vectors = built["facets"][name]
        assert vectors.shape[0] == len(media)
        assert np.isfinite(vectors).all(), f"{name} có NaN/inf"

    # Hệ quả đo được của việc Goodreads không có DROPPED: hai cột nhịp truyện
    # dựa vào nhãn bỏ dở phải bằng 0 trên toàn danh mục, không phải giá trị bịa.
    table = built["pace_table"]
    assert (table["drop_rate"] == 0).all()
    assert (table["pause_rate"] == 0).all()


# ------------------------------------------------------------------- ảnh bìa

def test_nang_cap_url_anh_bia_va_bo_anh_placeholder():
    """URL trong dataset cho ảnh 98px; đổi hậu tố cho 309px.

    Đo thật trên 150 URL: bản 'm' rộng trung vị 98px (dưới ngưỡng báo động
    300px, tức không đo được độ dày nét), bản 'l' rộng trung vị 309px và cao
    460px. Không đổi hậu tố thì mặt nét vẽ trên nhánh Goodreads chết oan.
    """
    from truyen_reco.data.ingest_goodreads import cover_url

    small = "https://images.gr-assets.com/books/1392195267m/18131.jpg"
    assert cover_url(small) == (
        "https://images.gr-assets.com/books/1392195267l/18131.jpg")
    # 37,4% tác phẩm dùng cùng một ảnh xám 111x148; để lại sẽ thành một cụm
    # giả 23.527 tác phẩm có vector nét vẽ giống hệt nhau.
    assert cover_url(
        "https://s.gr-assets.com/assets/nophoto/book/111x148-bcc042.png") is None
    assert cover_url("") is None
    assert cover_url(None) is None
    # URL không theo khuôn thì giữ nguyên, không đoán.
    other = "https://example.org/anh.jpg"
    assert cover_url(other) == other


# ------------------------------------------- thiếu dữ liệu ≠ mô hình trượt

def test_chi_tieu_theo_nuoc_la_na_chu_khong_phai_truot():
    """Danh mục không có trường xuất xứ thì chỉ tiêu theo nước phải ra N/A.

    Để nguyên thì "số nước trong top-10" bằng 0 và bảng bậc 4 báo KHÔNG ĐẠT,
    tức là kết luận sai về mô hình khi nguyên nhân nằm hẳn ở dữ liệu. Đây đúng
    là kiểu lỗi đã xảy ra một lần với chính chỉ tiêu này trên nhánh AniList, chỉ
    khác chiều: ở đó nó *đạt* vì sai lý do.
    """
    import importlib.util
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "scripts" / "measure_product.py"
    spec = importlib.util.spec_from_file_location("measure_product", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert "countries" in module.NEEDS_COUNTRY
    assert module.passed("countries", "N/A") == "N/A"
    # Vẫn phải chấm đúng khi có dữ liệu thật.
    assert module.passed("countries", 2.5) is True
    assert module.passed("countries", 1.5) is False
    # Không đo được thì vẫn là trượt, không phải N/A — hai chuyện khác nhau.
    assert module.passed("countries", None) is False


def test_ti_le_theo_nuoc_la_nan_khi_khong_co_truong_xuat_xu():
    """country_map toàn None phải cho mix_* = nan, không phải 0,00.

    `{1: None, 2: None}` vẫn là dict "truthy", nên một phép kiểm tra
    `if country_map` sẽ chạy tiếp và báo tỉ lệ manhua 0,00 — đọc lên hệt như
    "mô hình không gợi ý manhua nào", trong khi sự thật là bộ dữ liệu không có
    trường xuất xứ. Đây đúng là kiểu lỗi mà bảng chỉ tiêu sản phẩm đã mắc một
    lần ở chỗ khác: thiếu dữ liệu hiện ra thành một con số.
    """
    import math

    from truyen_reco.evaluation.metrics import evaluate

    test_frame = pd.DataFrame({"user": ["u1", "u2"], "media_id": [1, 2]})
    train_by_user = {"u1": [3], "u2": [4]}

    def recommend(user, history):
        return [1, 2, 3]

    empty = evaluate(recommend, test_frame, train_by_user, k=3,
                     country_map={1: None, 2: None, 3: None})
    for country in ("JP", "KR", "CN"):
        assert math.isnan(empty[f"mix_{country}"]), (
            f"mix_{country} = {empty[f'mix_{country}']} — thiếu dữ liệu xuất xứ "
            "bị báo thành một tỉ lệ có thật")

    filled = evaluate(recommend, test_frame, train_by_user, k=3,
                      country_map={1: "JP", 2: "KR", 3: "JP"})
    assert filled["mix_JP"] == pytest.approx(2 / 3)
    assert filled["mix_CN"] == 0.0, "có trường xuất xứ thì 0 là kết quả thật"
