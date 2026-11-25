"""Mô hình lưu ra rồi nạp lại phải chấm y hệt lúc còn trong bộ nhớ.

Đây là phép kiểm cho đúng chỗ đã hỏng: trước đây script đánh giá chấm trong bộ
nhớ rồi ném mô hình đi, còn API phục vụ hệ hai tầng cũ, nên bảng kết quả không
nói gì về sản phẩm. Nếu `ServedTwoTower` lệch khỏi mô hình đã huấn luyện thì
test này phải đỏ.
"""

import numpy as np
import pytest
import torch

from truyen_reco.models.two_tower import (
    RoutedTwoTower, ServedTwoTower, TwoTower, build_history_tensors,
    pack_for_serving, pool_history, unit)


def tiny_model(n_users=4, n_items=9, content_dim=6, dim=5, time_dim=0,
               trend_gate=False):
    torch.manual_seed(0)
    return TwoTower(n_users, n_items, content_dim, time_dim=time_dim, dim=dim,
                    trend_gate=trend_gate)


def tiny_blob(use_time=False, trend_gate=False):
    model = tiny_model(time_dim=2 if use_time else 0, trend_gate=trend_gate)
    if use_time:
        # Đầu thời điểm khởi tạo bằng 0 (có chủ ý: lúc chưa học thì nó không
        # được nói gì), nên phải gán khác 0 mới kiểm được nó có tác dụng.
        with torch.no_grad():
            model.time.weight.copy_(torch.tensor([[0.5, -0.25]]))
    # Gieo lại trước khi rút nội dung: bật cổng làm mô hình tạo thêm hai tham
    # số, tức là tiêu thêm số ngẫu nhiên, nên không gieo lại thì hai cấu hình
    # nhận hai ma trận nội dung khác nhau và phép so "trùng khít" vô nghĩa.
    torch.manual_seed(12345)
    content = torch.randn(9, 6)
    ids = torch.arange(9)
    raw = model.item_vectors(content, ids, drop_id=False).detach()
    times = torch.randn(9, 2) if use_time else None
    bias = model.time_bias(times).detach() if use_time else None

    history = [[0, 1, 2], [3, 4], [], [5, 6, 7, 8]]
    stamps = [[10, 20, 30], [40, 50], [], [60, 70, 80, 90]]
    index, weight = build_history_tensors(history, stamps, 100, 4, max_len=4)
    blob = pack_for_serving(model, {"a": 0, "b": 1, "c": 2, "d": 3},
                            index, weight, raw, unit(raw), bias, use_time)
    return model, blob, raw


def test_served_scores_match_the_in_memory_model():
    model, blob, raw = tiny_blob()
    served = ServedTwoTower(blob)

    with torch.no_grad():
        pooled = pool_history(raw, blob["hist_index"][0:1], blob["hist_weight"][0:1])
        expected = (unit(raw) @ model.user_vectors(
            pooled, torch.tensor([0]), drop_id=False).squeeze(0) / 0.07).numpy()

    assert np.allclose(served.scores("a"), expected, atol=1e-5)


def test_round_trip_through_disk_changes_nothing(tmp_path):
    _, blob, _ = tiny_blob(use_time=True)
    path = tmp_path / "tower.pt"
    torch.save(blob, path)

    before = ServedTwoTower(blob).scores("d")
    after = ServedTwoTower.load(path).scores("d")
    assert np.allclose(before, after, atol=1e-6)


def test_unknown_user_is_profiled_from_picks_only():
    """Người đọc mới: không có vector ID, hồ sơ chỉ từ truyện vừa chọn."""
    _, blob, _ = tiny_blob()
    served = ServedTwoTower(blob)

    one = served.scores(None, history_positions=[0])
    other = served.scores(None, history_positions=[8])
    assert not np.allclose(one, other)

    # Tên lạ cũng phải đi đúng đường đó, không được tra vào lịch sử người khác
    assert np.allclose(served.scores("khong-ton-tai", [0]), one, atol=1e-6)


def test_top_k_never_returns_items_already_read():
    _, blob, _ = tiny_blob()
    served = ServedTwoTower(blob)
    read = [0, 1, 2, 3]
    top = served.top_k(None, read, k=4)
    assert not set(int(i) for i in top) & set(read)


def test_time_head_shifts_scores_when_present():
    _, blob, _ = tiny_blob(use_time=True)
    served = ServedTwoTower(blob)
    assert served.config["time_dim"] == 2
    assert served.item_bias is not None

    plain = dict(blob)
    plain["item_bias"] = None
    assert not np.allclose(served.scores("a"),
                           ServedTwoTower(plain).scores("a"), atol=1e-6)


# ---- cổng độ phổ biến --------------------------------------------------
def gated_blob(base_slope=-0.2, base_intercept=1.6, dropout=0.3):
    """Cổng đặt tay theo đúng dấu đo được trên dữ liệu thật.

    `g_u = base(log(1+n)) + delta_u`. Trên dữ liệu thật phần chung giảm theo độ
    dài lịch sử (tương quan hạng −0,51), nên ở đây `base_slope` âm. Bốn người
    trong blob nhỏ có độ dài lịch sử 3, 2, 0, 4.
    """
    model, blob, raw = tiny_blob(use_time=True, trend_gate=True)
    with torch.no_grad():
        model.trend.weight.copy_(torch.tensor([[0.3], [-0.2], [0.0], [0.5]]))
        model.trend_base.weight.copy_(torch.tensor([[base_slope]]))
        model.trend_base.bias.copy_(torch.tensor([base_intercept]))
    model.trend_dropout = dropout
    return model, pack_for_serving(
        model, blob["user_index"], blob["hist_index"], blob["hist_weight"],
        blob["item_raw"], blob["item_unit"], blob["item_bias"], True), raw


def base_at(n, slope=-0.2, intercept=1.6):
    return intercept + slope * float(np.log1p(n))


def test_known_user_gets_the_shared_part_plus_their_own_deviation():
    _, blob, _ = gated_blob()
    served = ServedTwoTower(blob)
    # Trọng số lưu ở float32 nên so bằng sai số, không so bằng dấu bằng
    assert served.gate(torch.tensor([0]), 3) == pytest.approx(
        base_at(3) + 0.3, abs=1e-5)
    assert served.gate(torch.tensor([3]), 4) == pytest.approx(
        base_at(4) + 0.5, abs=1e-5)


def test_cold_user_gets_exactly_the_shared_part():
    """Người đọc mới nhận đúng công thức mà huấn luyện đã luyện.

    Không hồi quy hậu kiểm, không trung bình toàn dân: phần chung là một hàm
    của độ dài lịch sử nằm trong chính mô hình, nên lúc phục vụ nó khớp đúng
    với những bước đã bỏ `delta_u`. Dấu phải đúng: ít lịch sử → cổng cao hơn.
    """
    _, blob, _ = gated_blob()
    served = ServedTwoTower(blob)
    assert not served.legacy_gate
    assert blob["trend_fit"] is None

    assert served.gate(None, 1) == pytest.approx(base_at(1), abs=1e-5)
    assert served.gate(None, 1) > served.gate(None, 4)


def test_dropping_the_deviation_leaves_the_shared_part():
    """`drop_delta` phải bỏ đúng phần riêng, không chạm phần chung."""
    model, _, _ = gated_blob(dropout=1.0)
    users = torch.tensor([0, 1, 3])
    lengths = torch.tensor([3, 2, 4])
    with torch.no_grad():
        dropped = model.trend_weight(users, lengths, drop_delta=True)
        kept = model.trend_weight(users, lengths, drop_delta=False)
        shared = model.trend_weight(None, lengths)
    assert torch.allclose(dropped, shared, atol=1e-6)
    assert not torch.allclose(kept, shared, atol=1e-3)


def test_untrained_gate_reproduces_the_ungated_model():
    """Khởi tạo phải trùng khít bản không cổng, nên mọi khác biệt là do học.

    Hai mô hình không thể so bằng cách gieo cùng seed: bật cổng tạo thêm hai
    tham số, tiêu thêm số ngẫu nhiên, nên *mọi* trọng số sau đó lệch đi. Nên
    sao thẳng trọng số dùng chung sang bản không cổng rồi mới so — như vậy phép
    so cô lập đúng một thứ là cái cổng.
    """
    gated, blob, _ = tiny_blob(use_time=True, trend_gate=True)
    plain_model, plain_blob, _ = tiny_blob(use_time=True)

    assert float(gated.trend.weight.detach().abs().sum()) == 0.0
    assert ServedTwoTower(blob).gate(None, 7) == pytest.approx(1.0, abs=1e-6)

    shared = {k: v for k, v in gated.state_dict().items()
              if not k.startswith("trend")}
    plain_model.load_state_dict(shared)
    plain = dict(plain_blob)
    plain["state_dict"] = shared
    plain["item_raw"] = blob["item_raw"]
    plain["item_unit"] = blob["item_unit"]
    plain["item_bias"] = blob["item_bias"]

    assert np.allclose(ServedTwoTower(blob).scores("a"),
                       ServedTwoTower(plain).scores("a"), atol=1e-5)


def test_legacy_gate_blob_still_loads_and_uses_the_old_path():
    """Blob lưu trước khi cổng tách hai phần vẫn phải chấm lại được."""
    _, blob, _ = gated_blob()
    legacy = dict(blob)
    legacy["state_dict"] = {k: v for k, v in blob["state_dict"].items()
                            if not k.startswith("trend_base")}
    legacy["trend_fit"] = {"intercept": 1.5, "slope": -0.1,
                           "lo": 0.5, "hi": 2.0, "mean": 1.3}
    served = ServedTwoTower(legacy)
    assert served.legacy_gate
    assert served.gate(None, 1) == pytest.approx(1.5 - 0.1 * np.log1p(1), abs=1e-6)
    assert served.gate(torch.tensor([0]), 3) == pytest.approx(0.3, abs=1e-6)


def test_blob_without_a_gate_is_unaffected():
    """Blob cũ (không cổng) phải chấm y như trước khi có đoạn này."""
    _, blob, _ = tiny_blob(use_time=True)
    served = ServedTwoTower(blob)
    assert served.trend_fit is None
    assert served.gate(None, 3) == 1.0
    assert served.gate(torch.tensor([0]), 3) == 1.0


# ---- định tuyến theo lịch sử (engine mặc định) -------------------------
def routed_pair():
    _, cold_blob, _ = tiny_blob(use_time=True)
    _, warm_blob, _ = gated_blob()
    return (ServedTwoTower(cold_blob), ServedTwoTower(warm_blob),
            RoutedTwoTower(ServedTwoTower(cold_blob),
                           ServedTwoTower(warm_blob)))


def test_routing_sends_each_reader_to_the_model_measured_better_for_them():
    """Ranh giới là CÓ hay KHÔNG CÓ lịch sử, không phải ít hay nhiều.

    Trong blob nhỏ, người 'c' có lịch sử rỗng nên phải đi đường người-đọc-mới
    dù tên họ có trong bảng — đó đúng là tình huống `feed` hơn `gated` 5,2%.
    """
    cold, warm, routed = routed_pair()
    assert routed.pick("a")[0] is routed.warm and routed.pick("a")[1] == 3
    assert routed.pick("b")[0] is routed.warm and routed.pick("b")[1] == 2
    assert routed.pick("c")[0] is routed.cold and routed.pick("c")[1] == 0
    assert routed.pick(None)[0] is routed.cold
    assert routed.pick("khong-ton-tai")[0] is routed.cold


def test_routing_reproduces_each_model_exactly_on_its_own_side():
    """Định tuyến chỉ chọn đường, không được làm lệch điểm một chút nào.

    Đây là lý do con số của nó không cần huấn luyện lại: nó kế thừa nguyên vẹn
    điểm của blob phục vụ, nên phép so là ghép cặp hoàn hảo.
    """
    cold, warm, routed = routed_pair()
    assert np.allclose(routed.scores("a"), warm.scores("a"), atol=0)
    assert np.allclose(routed.scores("c"), cold.scores("c"), atol=0)
    assert np.allclose(routed.scores(None, [0, 1]),
                       cold.scores(None, [0, 1]), atol=0)
    assert not np.allclose(routed.scores("a"), cold.scores("a"), atol=1e-4)


def test_routing_refuses_two_blobs_with_different_user_tables():
    """Ghép hai blob lệch bảng người đọc sẽ chấm sai âm thầm — phải nổ."""
    cold, warm, _ = routed_pair()
    warm.user_index = dict(warm.user_index)
    warm.user_index["z"] = 99
    with pytest.raises(ValueError):
        RoutedTwoTower(cold, warm)
