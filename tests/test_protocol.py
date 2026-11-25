"""Hợp đồng của giao thức "hướng ban đầu" (đề xuất đồ án).

Nhánh này có bốn chỗ dễ sai mà không báo gì, nên mỗi chỗ một test:

1. chia 70/10/20 **theo từng người**: lệch một chỉ số là rò dữ liệu tương lai;
2. gộp tập thành bộ: điểm phải là trung bình các tập *đã chấm*, mốc thời gian
   phải là lần **đầu**, và id mục bộ phải không đụng không gian `work_id`;
3. ba định nghĩa "dương" phải tách bạch, không trộn;
4. loại ứng viên phải cắt cả **cùng bộ**, không chỉ "đã đọc" — đây là chỗ quyết
   định giữa một con số thật và một con số do gợi "tập kế tiếp".
"""

import numpy as np
import pandas as pd
import pytest

from truyen_reco.data import protocol_peruser as proto


def frame(rows):
    return pd.DataFrame(rows, columns=["user", "media_id", "created_at",
                                       "updated_at", "status", "score"])


def test_chia_70_10_20_theo_tung_nguoi_va_khong_ro_tuong_lai():
    rows = [("u", i, 100 + i, 100 + i, "COMPLETED", 100.0) for i in range(10)]
    rows += [("v", 50 + i, 200 + i, 200 + i, "COMPLETED", 100.0) for i in range(20)]
    data = frame(rows)
    train, val, test, _ = proto.split_per_user(data)

    for user, expect in (("u", (7, 1, 2)), ("v", (14, 2, 4))):
        counts = (len(train[train["user"] == user]),
                  len(val[val["user"] == user]),
                  len(test[test["user"] == user]))
        assert counts == expect, f"{user}: {counts} != {expect}"

    # Mọi mốc thời gian của train phải TRƯỚC val, và val trước test.
    for user in ("u", "v"):
        t = train[train["user"] == user]["created_at"].max()
        v = val[val["user"] == user]["created_at"]
        e = test[test["user"] == user]["created_at"]
        assert t < v.min() and v.max() < e.min()


def test_moi_nguoi_con_it_nhat_mot_muc_test():
    # 10 mục là ngưỡng k-core; 70/10/20 trên 10 mục không được cho test rỗng.
    data = frame([("u", i, i, i, "COMPLETED", 100.0) for i in range(10)])
    _, val, test, _ = proto.split_per_user(data)
    assert len(test) >= 1 and len(val) >= 1


def test_ba_dinh_nghia_duong_khong_tron_lan():
    data = frame([
        ("u", 1, 1, 1, "COMPLETED", 60.0),    # đã đọc, chấm 3 sao
        ("u", 2, 2, 2, "PLANNING", 100.0),    # chưa đọc, chấm 5 sao
        ("u", 3, 3, 3, "COMPLETED", np.nan),  # đã đọc, không chấm
    ])
    assert proto.mark_positive(data, "rating").tolist() == [False, True, False]
    assert proto.mark_positive(data, "read").tolist() == [True, False, True]
    assert proto.mark_positive(data, "either").tolist() == [True, True, True]
    with pytest.raises(ValueError):
        proto.mark_positive(data, "khong-ton-tai")


def test_gop_bo_lay_trung_binh_diem_da_cham_va_moc_dau_tien():
    data = frame([
        ("u", 10, 500, 500, "COMPLETED", 100.0),   # Naruto #1, 5 sao
        ("u", 11, 300, 400, "COMPLETED", 80.0),    # Naruto #2, 4 sao, sớm hơn
        ("u", 12, 600, 600, "PLANNING", np.nan),   # Naruto #3, chưa chấm
        ("u", 99, 700, 700, "COMPLETED", 80.0),    # truyện lẻ
    ])
    data["is_positive"] = True
    groups = {10: "Naruto", 11: "Naruto", 12: "Naruto", 99: "One Shot"}
    merged, report = proto.merge_series(data, groups)

    assert report["items_merged"] == 1, "chỉ Naruto có >1 tập trong danh mục"
    assert report["rows_collapsed"] == 2
    series_row = merged[merged["media_id"] < 0].iloc[0]
    assert series_row["created_at"] == 300, "mốc phải là lần tương tác đầu tiên"
    # Trung bình CỦA CÁC TẬP ĐÃ CHẤM: (100 + 80) / 2 = 90, tập không chấm không
    # được tính thành 0 để kéo điểm xuống.
    assert series_row["score"] == pytest.approx(90.0)
    assert series_row["n_volumes_read"] == 3
    # Truyện lẻ giữ id dương; mục bộ nhận id âm nên hai không gian không đụng.
    assert set(merged["media_id"]) == {-1, 99}


def test_loai_ung_vien_cat_ca_cung_bo_khong_chi_da_doc():
    train = pd.DataFrame({"user": ["u"], "media_id": [10]})
    val = pd.DataFrame({"user": ["u"], "media_id": [11]})
    item_index = {10: 0, 11: 1, 12: 2, 99: 3}
    series_of = {10: "series:Naruto", 11: "series:Naruto",
                 12: "series:Naruto", 99: None}
    excluded = proto.excluded_candidates(
        train, val, {"u": 0}, item_index, series_of)
    dropped = set(excluded["u"].tolist())
    # 0 và 1 vì đã đọc (train và val), 2 vì CÙNG BỘ dù chưa đọc.
    assert dropped == {0, 1, 2}
    assert 3 not in dropped, "truyện lẻ không liên quan thì không được loại"


def test_val_cung_bi_loai_khoi_ung_vien():
    train = pd.DataFrame({"user": ["u"], "media_id": [5]})
    val = pd.DataFrame({"user": ["u"], "media_id": [6]})
    excluded = proto.excluded_candidates(
        train, val, {"u": 0}, {5: 0, 6: 1, 7: 2}, {5: None, 6: None, 7: None})
    assert set(excluded["u"].tolist()) == {0, 1}


def test_mau_tieu_de_bat_dung_tap_duoc_danh_so():
    hits = {
        "Crossed + One Hundred, Volume 2 (Crossed +100 #2)": "Crossed +100",
        "Naruto, Vol. 7 (Naruto, #7)": "Naruto",
        "20th Century Boys, Volume 20 (20th Century Boys #20)":
            "20th Century Boys",
    }
    for title, expect in hits.items():
        match = proto.VOLUME_PATTERN.search(title)
        assert match and match.group(1).strip() == expect, title
    # Không có số tập thì không gộp: "(Vol. 3)" một mình không phải tên bộ.
    assert proto.VOLUME_PATTERN.search("Watchmen") is None
    assert proto.VOLUME_PATTERN.search("Akira (Vol. 3)") is None


def test_dap_an_phai_nam_trong_ro_ung_vien():
    # u đã đọc mục 10 (Naruto #1). Mục test 11 là Naruto #2 -> bị loại khỏi rổ
    # ứng viên, nên cũng phải bị loại khỏi đáp án; mục 99 thì giữ.
    train = pd.DataFrame({"user": ["u"], "media_id": [10]})
    val = pd.DataFrame({"user": pd.Series(dtype=object),
                        "media_id": pd.Series(dtype=int)})
    item_index = {10: 0, 11: 1, 99: 2}
    series_of = {10: "series:Naruto", 11: "series:Naruto", 99: None}
    excluded = proto.excluded_candidates(train, val, {"u": 0}, item_index, series_of)
    test = pd.DataFrame({"user": ["u", "u", "u"], "media_id": [11, 99, 12345]})
    kept, report = proto.reachable_test(test, excluded, item_index)

    assert kept["media_id"].tolist() == [99]
    assert report["dropped_same_series_or_seen"] == 1
    assert report["dropped_item_not_in_train"] == 1


def test_ban_do_bo_duoc_dung_ca_khi_khong_gop():
    """Biến thể `none` vẫn phải biết bộ của từng mục, nếu không thì phép loại
    "cùng bộ" của mục 7.1 đề xuất lặng lẽ không chạy — và đó là chỗ nó quan
    trọng nhất, vì bản đã gộp thì các tập đã thành một mục."""
    rows = []
    for user in ("u", "v", "w", "x", "y", "z"):
        for item in (10, 11, 12, 13, 14, 20, 21, 22, 23, 24, 30, 31):
            rows.append((user, item, item, item, "COMPLETED", 100.0))
    data = frame(rows)
    groups = {10: "Naruto", 11: "Naruto", 12: "Naruto", 13: "Naruto",
              14: "Naruto", 20: "One Piece", 21: "One Piece"}
    bundle = proto.run(data, rule="rating", merge="none", groups=groups,
                       min_user=10, min_item=5)

    series = bundle["series_of"]
    assert any(v is not None for v in series.values()), (
        "không gộp thì vẫn phải có bản đồ bộ để loại ứng viên cùng bộ")
    assert series[11] == "series:Naruto" and series[30] is None
    assert bundle["report"]["series_strategy"] == "volume"
    # Và phép loại phải thật sự cắt: người đọc Naruto #1 không được nhận #2.
    excluded = proto.excluded_candidates(
        bundle["train"], bundle["val"],
        {u: i for i, u in enumerate(sorted(bundle["train"]["user"].unique()))},
        {m: i for i, m in enumerate(sorted(bundle["train"]["media_id"].unique()))},
        series)
    assert all(len(v) > 0 for v in excluded.values())
