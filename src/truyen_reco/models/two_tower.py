"""Hai tháp: nhúng truyện trộn ID với nội dung, nhúng người đọc từ lịch sử.

Thiết kế suy ra từ `docs/data-diagnosis.md`, từng lựa chọn gắn với một số đo
được, không phải chọn kiến trúc cho sang:

1. *Chỉ 35,6% truyện trong test lọt vào top-400 ứng viên ALS.* Trần đó chặn
   mọi cải tiến ở tầng xếp lại. Bỏ hẳn hai tầng: 34.608 truyện × 64 chiều là
   một phép nhân ma trận, chấm hết danh mục cho mỗi người còn nhanh hơn dựng
   ứng viên. Trần 35,6% biến mất, không cần thêm nguồn ứng viên nào.

2. *Nội dung thêm +0,085 AUC lên ALS, nhưng chỉ +0,007 khi đã có thời điểm.*
   Nên nội dung và thời điểm phần lớn là một tín hiệu. Tách thành hai số hạng
   riêng: `<q_u, v_i>` lo gu, một đầu tuyến tính lo thời điểm. Tách ra mới
   tách được công, và mới trả lời được "hệ này đang gợi theo gu hay theo
   truyện đang nổi".

3. *Cách trộn cũ (z-score rồi trung bình trọng số cố định) đánh rơi tín hiệu
   mà hồi quy vẫn thấy.* Nên không đặt trọng số bằng tay nữa: `v_i` cộng
   ID với phép chiếu nội dung rồi học cả hai bằng chính mục tiêu xếp hạng.

4. *Nội dung mang tín hiệu thật (lift theme 15,2 / pace 8,7 / art 5,0).*
   Nhưng chỉ ID thôi thì truyện mới không có vector. Nên bỏ ngẫu nhiên vector
   ID trong lúc huấn luyện: phép chiếu nội dung buộc phải tự gánh, tức là
   đường đi cho truyện mới được huấn luyện trực tiếp chứ không phải chắp thêm
   lúc suy luận.

Không có bias tự do cho từng truyện, dù nó thường giúp: một bias tự do sẽ học
lại độ phổ biến và trộn lẫn vào "gu", làm mục 2 mất ý nghĩa. Độ phổ biến vào
hệ qua đầu thời điểm, nơi nó đo được và tắt được.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def unit(vectors, eps=1e-8):
    """Chuẩn hoá L2 để tích trong là cosine.

    Chỉ chuẩn hoá ở *vế chấm điểm*, không chuẩn hoá vector dùng để gộp lịch
    sử: phép gộp phải tuyến tính mới tách được thành
    `gộp(e_id) + P · gộp(c)`, mà chuẩn hoá từng mục thì không tuyến tính nữa.

    Có chuẩn hoá thì nhiệt độ 0,07 mới đúng nghĩa. Không chuẩn hoá mà vẫn chia
    0,07 là nhân logit lên 14 lần, khiến softmax bão hoà ngay từ đầu.
    """
    return vectors / vectors.norm(dim=-1, keepdim=True).clamp(min=eps)


class TwoTower(nn.Module):
    def __init__(self, n_users, n_items, content_dim, time_dim=0, dim=64,
                 id_dropout=0.3, user_dropout=0.2, trend_gate=False,
                 trend_dropout=0.3):
        super().__init__()
        self.dim = dim
        self.id_dropout = id_dropout
        self.user_dropout = user_dropout

        self.item_id = nn.Embedding(n_items, dim)
        self.content = nn.Linear(content_dim, dim, bias=False)
        self.user_id = nn.Embedding(n_users, dim)
        self.history = nn.Linear(dim, dim, bias=False)
        self.time = nn.Linear(time_dim, 1, bias=False) if time_dim else None
        # Cổng theo người đọc cho số hạng thời điểm. Có nó vì phép đo, không
        # phải cho đủ bộ: đầu thời điểm là một số hạng *cộng chung cho mọi
        # người*, nên nó đẩy tất cả về cùng một danh sách.
        #
        # Đã đo trọng số học được của đầu này, và nó buộc phải đổi tên: nó
        # KHÔNG phải đầu "đang nổi". Trọng số là log_pop_all +0,891 /
        # log_pop_60d +0,054 / is_releasing −0,034 / age_years −0,005 /
        # is_finished −0,012 — tức 94% là **độ phổ biến tích luỹ cả đời**, và
        # hai bản (có cổng / không cổng) cho cùng một thứ tự (tương quan hạng
        # +1,0000, trùng 997/1000 ở top-1000). Lý do manhua bị dìm cũng theo
        # đó mà rõ: top-100 theo đầu này có **0% manhua**, top-1000 có 5,6%.
        # Nên cái cổng này mở ra là "mức thích truyện phổ biến" của từng người,
        # và nó bám đúng nét đó: tương quan hạng giữa g_u và phân vị phổ biến
        # trung bình của những bộ chính người đó đã đọc là **+0,84**.
        # Cho phép từng người khác nhau được +5,8% NDCG ở lát `warm`.
        #
        # Tách cổng thành hai phần thay vì một bảng tra theo người, vì bản một
        # phần đã đo được là **lấy mất tín hiệu của đường đi cho người mới**.
        # Số đo: với người đọc mới 3 lượt chọn, top-10 của bản không cổng trùng
        # 0,532 với top-10 chấm bằng *riêng gu*, còn bản có cổng chỉ 0,082 —
        # và 0,720 top-10 của nó nằm trong top-200 phổ biến. Tắt cổng về 0 thì
        # NDCG người đọc mới rơi xuống 0,0264, nghĩa là đường gu của nó rỗng.
        # Cơ chế: khi mỗi người có một hệ số phổ biến riêng, đường gộp-lịch-sử
        # không cần mang thông tin phổ biến nữa — mà người đọc mới *chỉ có*
        # đường đó. Đúng lập luận của DropoutNet (Volkovs, Yu & Poutanen 2017),
        # chỉ là áp cho cổng chứ không cho vector ID.
        #
        #   g_u = base(log(1+n_u)) + delta_u
        #
        # `base` là hàm của độ dài lịch sử, dùng được cho *mọi* người kể cả
        # người mới — nó mã hoá thẳng vào cấu trúc quan hệ đã đo được (g_u giảm
        # đơn điệu qua cả sáu bậc độ dài, tương quan hạng −0,51) thay vì khớp
        # lại sau bằng hồi quy. `delta_u` là phần riêng của từng người và bị bỏ
        # ngẫu nhiên lúc huấn luyện, nên những bước đó buộc đường gu phải tự
        # gánh — y như cách vector ID của truyện bị bỏ ngẫu nhiên.
        gated = bool(time_dim and trend_gate)
        self.trend = nn.Embedding(n_users, 1) if gated else None
        self.trend_base = nn.Linear(1, 1) if gated else None
        self.trend_dropout = trend_dropout

        nn.init.normal_(self.item_id.weight, std=0.05)
        nn.init.normal_(self.user_id.weight, std=0.05)
        nn.init.normal_(self.content.weight, std=0.05)
        nn.init.eye_(self.history.weight)
        if self.time is not None:
            nn.init.zeros_(self.time.weight)
        if self.trend is not None:
            # Phần riêng khởi tạo 0, phần chung khởi tạo hằng 1: lúc chưa học
            # cổng mở hoàn toàn và mô hình trùng khít bản không cổng, nên mọi
            # khác biệt sau đó là do học được.
            nn.init.zeros_(self.trend.weight)
            nn.init.zeros_(self.trend_base.weight)
            nn.init.ones_(self.trend_base.bias)

    # ---- tháp truyện -------------------------------------------------
    def item_vectors(self, content, item_ids=None, drop_id=False):
        """`v_i = e_id[i] * mask + P c_i`.

        `drop_id=True` tắt vector ID một cách ngẫu nhiên theo từng truyện. Đây
        là đường đi của truyện mới: lúc suy luận một truyện chưa ai đọc chỉ có
        `P c_i`, nên lúc huấn luyện phải có những bước mà nó cũng chỉ có thế.
        """
        vectors = self.content(content)
        if item_ids is not None:
            identity = self.item_id(item_ids)
            if drop_id and self.id_dropout > 0:
                keep = (torch.rand(len(item_ids), 1, device=identity.device)
                        >= self.id_dropout).float()
                identity = identity * keep
            vectors = vectors + identity
        return vectors

    # ---- tháp người đọc ----------------------------------------------
    def user_vectors(self, pooled, user_ids=None, drop_id=False):
        """Lịch sử đã gộp, cộng vector ID của người đọc nếu có.

        Người đọc mới không có vector ID, nên cũng bỏ ngẫu nhiên lúc huấn
        luyện, cùng lý do như truyện mới.
        """
        vectors = self.history(pooled)
        if user_ids is not None:
            identity = self.user_id(user_ids)
            if drop_id and self.user_dropout > 0:
                keep = (torch.rand(len(user_ids), 1, device=identity.device)
                        >= self.user_dropout).float()
                identity = identity * keep
            vectors = vectors + identity
        return unit(vectors)

    def time_bias(self, time_features):
        if self.time is None or time_features is None:
            return 0.0
        return self.time(time_features).squeeze(-1)

    def trend_weight(self, user_ids=None, n_history=None, drop_delta=False):
        """Cổng độ phổ biến: `base(log(1+n)) + delta_u`.

        `user_ids=None` chỉ còn phần chung — đó chính là đường đi của người đọc
        mới, và nó *khớp đúng* với những bước huấn luyện đã bỏ `delta_u`, nên
        không còn khoảng lệch giữa lúc học và lúc phục vụ như bản dùng hồi quy
        hậu kiểm (`fit_trend_prior`).
        """
        if self.trend is None:
            return 1.0
        if n_history is None:
            length = torch.zeros(1, 1)
        elif torch.is_tensor(n_history):
            length = torch.log1p(n_history.float()).reshape(-1, 1)
        else:
            length = torch.log1p(torch.tensor([[float(n_history)]]))
        gate = self.trend_base(length.to(self.trend_base.weight.device))
        if user_ids is None:
            return gate
        delta = self.trend(user_ids)
        if drop_delta and self.trend_dropout > 0:
            keep = (torch.rand(len(user_ids), 1, device=delta.device)
                    >= self.trend_dropout).float()
            delta = delta * keep
        return gate + delta


def pool_history(item_vectors, hist_index, hist_weight):
    """Gộp lịch sử có trọng số, đã chuẩn hoá theo hàng.

    Trọng số giảm theo thời gian: người đọc "ăn tạp" (độ chụm chỉ 1,2–1,6 lần
    ngẫu nhiên, mục 3 của chẩn đoán), nên một vector trung bình phẳng trên 179
    mục bị loãng. Ưu tiên mục gần đây làm hồ sơ sắc hơn mà không cần tách
    nhiều sở thích — chẩn đoán đã cho thấy tách cụm không phải đòn bẩy lớn.
    """
    safe = hist_index.clamp(min=0)
    gathered = item_vectors[safe]
    mask = (hist_index >= 0).float().unsqueeze(-1)
    weight = (hist_weight.unsqueeze(-1) * mask)
    total = weight.sum(dim=1).clamp(min=1e-6)
    return (gathered * weight).sum(dim=1) / total


def build_history_tensors(history_positions, event_times, cutoff, n_users,
                          max_len=300, half_life_days=None):
    """Lịch sử dạng ma trận chữ nhật, cắt còn `max_len` mục gần nhất.

    `half_life_days=None` nghĩa là không giảm theo thời gian, và đó là mặc
    định — đo được chứ không chọn bừa (mục I, `docs/research-notes.md`): cosine
    chủ đề giữa hai mục của cùng một người cách nhau 24 tháng chỉ giảm 11%, tức
    gu gần như đứng yên qua nhiều năm.

    Lần chạy đầu dùng bán rã 180 ngày và đó là một lỗi: tuổi trung vị của một
    mục trong train là 821 ngày, nên 58% số mục nhận trọng số dưới 0,1 và tổng
    trọng số hiệu dụng chỉ còn 0,232 — hồ sơ người đọc bị dựng từ một lát mỏng
    gần đây, trong khi ALS được dùng trọn 1,52 triệu tương tác. Thêm nữa 9,7%
    số mục nằm ở mốc 1970 (`created_at = 0`) nên bị triệt tiêu hoàn toàn.

    Phần "đang ra, đang nổi" là việc của đầu thời điểm, không phải của tháp gu.
    Trộn nó vào đây làm mờ đúng cái phép tách công mà thiết kế dựa vào.
    """
    index = np.full((n_users, max_len), -1, dtype=np.int64)
    weight = np.zeros((n_users, max_len), dtype=np.float32)
    for user_position, (items, times) in enumerate(zip(history_positions, event_times)):
        if not len(items):
            continue
        items, times = items[-max_len:], times[-max_len:]
        index[user_position, :len(items)] = items
        if half_life_days:
            age = np.maximum((cutoff - np.asarray(times)) / 86400.0, 0.0)
            weight[user_position, :len(items)] = 0.5 ** (age / half_life_days)
        else:
            weight[user_position, :len(items)] = 1.0
    return torch.from_numpy(index), torch.from_numpy(weight)


def log_q(train_counts, batch_size, n_negatives, n_items):
    """Hiệu chỉnh logQ cho mẫu âm (Yi et al. 2019).

    Mẫu âm trong lô được rút theo tần suất lượt đọc, nên truyện phổ biến xuất
    hiện làm mẫu âm thường hơn và bị phạt quá tay. Trừ `log q_i` khỏi logit để
    bù lại. Không hiệu chỉnh thì mô hình bị đẩy khỏi truyện phổ biến — mà tập
    test lại nghiêng về truyện phổ biến, nên đây là hướng lệch ngược hẳn với
    thứ đang đo.

    `q_i` là xác suất truyện i xuất hiện làm ứng viên, trộn hai nguồn: trong
    lô (tỉ lệ với lượt đọc) và đều (mẫu âm ngẫu nhiên).
    """
    frequency = train_counts / max(train_counts.sum(), 1)
    uniform = 1.0 / n_items
    total = batch_size + n_negatives
    q = (batch_size * frequency + n_negatives * uniform) / total
    return torch.from_numpy(np.log(np.maximum(q, 1e-12)).astype(np.float32))


def fit_trend_prior(trend_weight, hist_index):
    """Hồi quy hậu kiểm cho cổng — **thiết kế cũ**, giữ để nạp lại blob đã lưu.

    Bản hiện tại không cần hàm này nữa: quan hệ với độ dài lịch sử đã nằm thẳng
    trong `TwoTower.trend_base`, nên lúc phục vụ người đọc mới dùng *đúng* công
    thức mà huấn luyện đã luyện. Hồi quy sau khi huấn luyện xong thì không —
    đường gu vẫn học ở điều kiện có `delta_u`, và đó mới là chỗ mất thật
    (mục 9.7 `docs/research-notes.md`).

    Khớp `g_u ≈ a + b·log(1+độ dài lịch sử)`, chỉ trên người đọc train.

    Người đọc mới không có `g_u` nào được học. Bản đầu phát cho họ *trung bình
    toàn dân*, và điều đó làm `new_user` rơi 0,3771 → 0,2499. Chẩn đoán
    (`scripts/diagnose_trend_gate.py`) chỉ ra vì sao, chỉ bằng dữ liệu train:

        độ dài lịch sử   1–4    5–14   15–39  40–99  100–199  ≥200
        g_u trung bình   1,455  1,443  1,406  1,354   1,306   1,247

    đơn điệu qua cả sáu bậc, tương quan hạng −0,51. Người đọc ít lịch sử bám
    truyện đang nổi *nhiều hơn*, điều hợp lý vì họ chưa có gu để bám. Trung
    bình toàn dân (1,323) chính là đường khớp đọc tại độ dài *trung bình của
    train* (~130 mục) — tức là đọc hồi quy ở sai chỗ trên trục x. Một người vừa
    chọn 3 truyện phải được đọc tại x = 3, cho 1,505.

    Nên đây không phải dò một hằng số cho vừa điểm test: `g_u` là biến ẩn không
    quan sát được ở người mới, nhưng *độ dài lịch sử thì quan sát được*, và
    quan hệ giữa hai thứ được đo trên train. Cắt vào khoảng `[min, max]` của
    `g_u` đã học để không ngoại suy ra ngoài vùng có dữ liệu.
    """
    gate = trend_weight.detach().cpu().numpy().ravel().astype(np.float64)
    length = (hist_index.cpu().numpy() >= 0).sum(axis=1).astype(np.float64)
    x = np.log1p(length)
    design = np.stack([np.ones_like(x), x], axis=1)
    coefficients, *_ = np.linalg.lstsq(design, gate, rcond=None)
    return {
        "intercept": float(coefficients[0]),
        "slope": float(coefficients[1]),
        "lo": float(gate.min()),
        "hi": float(gate.max()),
        "mean": float(gate.mean()),
    }


class ServedTwoTower:
    """Mô hình đã huấn luyện, nạp lại để chấm điểm — dùng cho cả đánh giá và API.

    Có lớp này để *đúng một* đường chấm điểm tồn tại. Trước đây script đánh giá
    tự chấm trong bộ nhớ rồi ném mô hình đi, còn API phục vụ hệ hai tầng cũ —
    nên con số trong báo cáo và hành vi của sản phẩm là hai thứ rời nhau. Giờ
    script đánh giá cũng đi qua lớp này, nên bảng kết quả đo đúng cái được phục
    vụ.

    Lưu sẵn bảng vector toàn danh mục (`item_raw` để gộp lịch sử, `item_unit`
    để chấm cosine) thay vì lưu ma trận nội dung rồi chiếu lại lúc nạp: chiếu
    lại cho 34.608 truyện mất thời gian khởi động mà kết quả không đổi. Hệ quả
    phải biết: thêm truyện mới vào danh mục thì phải huấn luyện/chiếu lại, bảng
    này không tự mọc thêm hàng.
    """

    def __init__(self, blob):
        config = blob["config"]
        self.config = config
        self.model = TwoTower(config["n_users"], config["n_items"],
                              config["content_dim"],
                              time_dim=config["time_dim"], dim=config["dim"],
                              trend_gate=config.get("trend_gate", False))
        # Blob lưu trước khi cổng tách thành `base + delta` chỉ có `trend.weight`.
        # Vẫn nạp được, và đi đường cũ (hồi quy hậu kiểm `trend_fit`) — ghi rõ
        # bằng một cờ chứ không im lặng dùng lẫn hai cách tính.
        self.legacy_gate = ("trend.weight" in blob["state_dict"]
                            and "trend_base.weight" not in blob["state_dict"])
        self.model.load_state_dict(blob["state_dict"],
                                   strict=not self.legacy_gate)
        self.model.eval()
        self.user_index = blob["user_index"]
        self.hist_index = blob["hist_index"]
        self.hist_weight = blob["hist_weight"]
        self.item_raw = blob["item_raw"]
        self.item_unit = blob["item_unit"]
        self.item_bias = blob["item_bias"]
        self.trend_fit = blob.get("trend_fit")

    @classmethod
    def load(cls, path):
        return cls(torch.load(path, map_location="cpu", weights_only=False))

    def gate(self, user_ids=None, n_history=0):
        """Cổng độ phổ biến lúc phục vụ.

        Bản hiện tại: `base(log(1+n)) + delta_u`, và người đọc mới nhận đúng
        `base` — cùng một công thức mà huấn luyện đã luyện (những bước bỏ
        `delta_u`), nên không có khoảng lệch học/phục vụ.

        Bản cũ (`legacy_gate`): bảng tra theo người, người mới nhận hồi quy hậu
        kiểm `trend_fit`. Giữ để blob đã lưu vẫn chấm lại được y hệt.
        """
        if self.model.trend is None:
            return 1.0
        if self.legacy_gate:
            if user_ids is not None:
                return float(self.model.trend(user_ids).detach().squeeze())
            if not self.trend_fit:
                return float(self.model.trend.weight.mean())
            fit = self.trend_fit
            value = fit["intercept"] + fit["slope"] * float(np.log1p(n_history))
            return float(np.clip(value, fit["lo"], fit["hi"]))
        with torch.no_grad():
            return float(self.model.trend_weight(user_ids, n_history).squeeze())

    @torch.no_grad()
    def scores(self, user=None, history_positions=()):
        """Điểm cho toàn bộ danh mục. `user=None` là người đọc mới.

        Người đọc đã có trong train dùng lịch sử đã lưu và vector ID của mình.
        Người đọc mới không có vector ID — hồ sơ dựng từ các truyện họ vừa chọn,
        đúng như lúc huấn luyện đã bỏ ngẫu nhiên vector ID để tập đường đi này.
        """
        position = self.user_index.get(user) if user is not None else None
        if position is not None:
            index = self.hist_index[position:position + 1]
            weight = self.hist_weight[position:position + 1]
            user_ids = torch.tensor([position])
            n_history = int((index >= 0).sum())
        else:
            length = self.hist_index.shape[1]
            index = torch.full((1, length), -1, dtype=torch.long)
            weight = torch.zeros((1, length), dtype=torch.float32)
            seeds = list(history_positions)[-length:]
            if seeds:
                index[0, :len(seeds)] = torch.tensor(seeds, dtype=torch.long)
                weight[0, :len(seeds)] = 1.0
            user_ids = None
            n_history = len(seeds)

        pooled = pool_history(self.item_raw, index, weight)
        vector = self.model.user_vectors(pooled, user_ids, drop_id=False)
        scores = self.item_unit @ vector.squeeze(0) / 0.07
        if self.item_bias is not None:
            scores = scores + self.gate(user_ids, n_history) * self.item_bias
        return scores.numpy()

    def top_k(self, user=None, history_positions=(), k=10):
        scores = self.scores(user, history_positions)
        blocked = np.asarray(list(history_positions), dtype=int)
        if blocked.size:
            scores = scores.copy()
            scores[blocked] = -np.inf
        top = np.argpartition(-scores, k)[:k]
        return top[np.argsort(-scores[top])]


class RoutedTwoTower:
    """Hai mô hình, chọn theo việc người đọc CÓ lịch sử hay KHÔNG.

    Không phải một ý tưởng thêm cho đủ engine: phép kiểm ghép cặp 4 seed
    (mục 9.9 `docs/research-notes.md`) cho hai kết quả **lập được** và chúng
    ngược chiều nhau —

        lát `all`       cổng hơn  +1,9%  (t = +4,27, 4/4 seed)
        lát `new_user`  cổng kém  −5,2%  (t = −5,79, 0/4 seed)

    tức không tồn tại một mô hình tốt hơn ở cả hai vùng. Ranh giới đo được
    **không** phải "ít hay nhiều lịch sử" (`sparse_user` nghiêng về cổng) mà
    đúng là "có hay không có": người đọc mới không có `delta_u`, không có vector
    ID, và frontier của cổng nằm trong `feed` 7,8% ở vùng đó.

    Định tuyến là một thay đổi ở **vế phục vụ**, nên đo được ghép cặp hoàn hảo:
    cùng trọng số, chỉ khác đường chấm. Vì vậy con số của nó không cần huấn
    luyện lại và không chịu nhiễu seed.
    """

    def __init__(self, cold, warm):
        if cold.user_index != warm.user_index:
            raise ValueError("hai blob không cùng bảng người đọc, không ghép được")
        self.cold, self.warm = cold, warm
        self.user_index = cold.user_index
        self.config = dict(warm.config)
        self.config["routed"] = True

    def pick(self, user=None):
        """Mô hình nào sẽ chấm — tách ra để test và log soi được quyết định."""
        position = self.user_index.get(user) if user is not None else None
        if position is None:
            return self.cold, 0
        length = int((self.warm.hist_index[position:position + 1] >= 0).sum())
        return (self.warm if length > 0 else self.cold), length

    def scores(self, user=None, history_positions=()):
        served, _ = self.pick(user)
        return served.scores(user, history_positions)

    def top_k(self, user=None, history_positions=(), k=10):
        served, _ = self.pick(user)
        return served.top_k(user, history_positions, k)


def pack_for_serving(model, user_index, hist_index, hist_weight,
                     item_raw, item_unit, item_bias, use_time):
    """Gói đúng những gì cần để chấm lại y hệt, không gói cả tập huấn luyện."""
    return {
        "state_dict": model.state_dict(),
        # None với thiết kế hiện tại: phần chung của cổng đã là một hàm của độ
        # dài lịch sử nằm trong mô hình, không cần hồi quy thêm sau. Chỉ blob
        # cũ (cổng một phần) mới mang khoá này.
        "trend_fit": None,
        "user_index": dict(user_index),
        "hist_index": hist_index,
        "hist_weight": hist_weight,
        "item_raw": item_raw,
        "item_unit": item_unit,
        "item_bias": item_bias,
        "config": {
            "n_users": model.user_id.num_embeddings,
            "n_items": model.item_id.num_embeddings,
            "content_dim": model.content.in_features,
            "time_dim": model.time.in_features if model.time is not None else 0,
            "dim": model.dim,
            "use_time": bool(use_time),
            "trend_gate": model.trend is not None,
        },
    }


def sampled_softmax_loss(user_vec, pos_vec, neg_vec, confidence, temperature=0.07):
    """Softmax có mẫu âm: mẫu âm trong lô cộng mẫu âm đều.

    Mẫu âm trong lô được rút theo phân phối lượt đọc, tức là truyện phổ biến
    xuất hiện làm mẫu âm thường hơn. Đó chính là phần hiệu chỉnh độ phổ biến
    cần có để `<q_u, v_i>` đo gu chứ không đo độ nổi — còn độ nổi thì đã có
    đầu thời điểm lo riêng.
    """
    positive = (user_vec * pos_vec).sum(-1, keepdim=True)
    negative = user_vec @ neg_vec.T
    logits = torch.cat([positive, negative], dim=1) / temperature
    target = torch.zeros(len(logits), dtype=torch.long, device=logits.device)
    losses = F.cross_entropy(logits, target, reduction="none")
    return (losses * confidence).sum() / confidence.sum()
