"""Năm mô hình nơ-ron của thang L3, L6, L9, L10 — cùng một giao diện.

Máy ở đây **không có GPU** (torch 2.10 CPU, 14 luồng), nên mọi lựa chọn dưới đây
nghiêng về rẻ: chiều 64, chuỗi dài tối đa 50, lấy mẫu âm đồng đều, và **dừng
sớm theo validation** thay vì dò lưới siêu tham số. Dừng sớm theo val vẫn là
"chọn trên validation" đúng nghĩa, và nó tốn một lần huấn luyện thay vì sáu.

Giao diện chung, để một vòng đánh giá dùng được cho cả năm mô hình:

- `kind`: `"pairwise"` (NeuMF, LightGCN) hoặc `"sequence"` (GRU4Rec, SASRec,
  CL4SRec, NRMS) — quyết định cách dựng lô.
- `loss(batch)`: trả về một tensor vô hướng.
- `scores(rows)`: ma trận `(len(rows), n_items)` dạng numpy, dùng để xếp hạng.

Một chỗ dễ sai mà tôi viết sẵn ở đây: **điểm phải chấm trên toàn danh mục**.
Mọi mô hình chuỗi đều dễ cài thành "chấm trên tập ứng viên lấy mẫu" vì nó nhanh
hơn hàng chục lần, và con số thu được không so được với bất kỳ dòng nào trong
bảng của đồ án này (xem Krichene & Rendle, KDD 2020).
"""

import numpy as np
import torch
from torch import nn


def set_threads(threads=None):
    if threads:
        torch.set_num_threads(int(threads))


class NeuMF(nn.Module):
    """GMF nhân từng chiều, ghép một MLP, rồi một lớp tuyến tính hợp hai nhánh.

    Hai nhánh có **embedding riêng**: dùng chung embedding thì MLP và GMF kéo
    cùng một vectơ về hai hướng và mô hình tệ hơn cả hai nhánh đơn lẻ — đó là
    điểm chính của bài NeuMF.
    """

    kind = "pairwise"

    def __init__(self, n_users, n_items, factors=32, layers=(64, 32, 16)):
        super().__init__()
        self.gmf_user = nn.Embedding(n_users, factors)
        self.gmf_item = nn.Embedding(n_items, factors)
        self.mlp_user = nn.Embedding(n_users, layers[0] // 2)
        self.mlp_item = nn.Embedding(n_items, layers[0] // 2)
        blocks = []
        for inside, outside in zip(layers, layers[1:]):
            blocks += [nn.Linear(inside, outside), nn.ReLU()]
        self.mlp = nn.Sequential(*blocks)
        self.out = nn.Linear(factors + layers[-1], 1)
        for embedding in (self.gmf_user, self.gmf_item, self.mlp_user,
                          self.mlp_item):
            nn.init.normal_(embedding.weight, std=0.01)

    def logits(self, users, items):
        gmf = self.gmf_user(users) * self.gmf_item(items)
        mlp = self.mlp(torch.cat([self.mlp_user(users), self.mlp_item(items)],
                                 dim=-1))
        return self.out(torch.cat([gmf, mlp], dim=-1)).squeeze(-1)

    def loss(self, batch):
        users, positives, negatives = batch
        positive = self.logits(users, positives)
        negative = self.logits(users, negatives)
        return nn.functional.binary_cross_entropy_with_logits(
            torch.cat([positive, negative]),
            torch.cat([torch.ones_like(positive), torch.zeros_like(negative)]))

    @torch.no_grad()
    def scores(self, rows):
        n_items = self.gmf_item.weight.shape[0]
        items = torch.arange(n_items)
        out = np.zeros((len(rows), n_items), dtype=np.float32)
        for position, row in enumerate(rows):
            users = torch.full((n_items,), int(row), dtype=torch.long)
            out[position] = self.logits(users, items).numpy()
        return out


class LightGCN(nn.Module):
    """Lan truyền trên đồ thị hai phía, không phép biến đổi phi tuyến nào.

    `propagate()` phải được gọi lại mỗi bước vì embedding gốc đổi sau mỗi bước
    cập nhật; gọi một lần rồi dùng mãi là một lỗi im lặng khiến mô hình đứng yên.
    """

    kind = "pairwise"

    def __init__(self, n_users, n_items, factors=64, layers=3,
                 adjacency=None):
        super().__init__()
        self.n_users, self.n_items, self.layers = n_users, n_items, layers
        self.embedding = nn.Embedding(n_users + n_items, factors)
        nn.init.normal_(self.embedding.weight, std=0.1)
        self.adjacency = adjacency

    def train(self, mode=True):
        # Bo nho dem chi song trong eval; moi lan quay lai train thi xoa.
        self._cache = None
        return super().train(mode)

    def propagate(self):
        if not self.training and getattr(self, "_cache", None) is not None:
            return self._cache
        state = self.embedding.weight
        total = state
        for _ in range(self.layers):
            state = torch.sparse.mm(self.adjacency, state)
            total = total + state
        total = total / (self.layers + 1)
        out = total[:self.n_users], total[self.n_users:]
        if not self.training:
            # Trong eval trọng số đứng yên, nên lan truyền lại cho **từng**
            # người là lãng phí thuần: lần chạy đầu mất 2.378 giây chấm 2.000
            # người, gần bằng một vòng huấn luyện, và cho đúng cùng con số.
            self._cache = out
        return out

    def loss(self, batch):
        users, positives, negatives = batch
        user_vectors, item_vectors = self.propagate()
        chosen = user_vectors[users]
        positive = (chosen * item_vectors[positives]).sum(-1)
        negative = (chosen * item_vectors[negatives]).sum(-1)
        regular = 1e-4 * (chosen.pow(2).sum() + item_vectors[positives].pow(2).sum()
                          + item_vectors[negatives].pow(2).sum()) / len(users)
        return -nn.functional.logsigmoid(positive - negative).mean() + regular

    @torch.no_grad()
    def scores(self, rows):
        user_vectors, item_vectors = self.propagate()
        chosen = user_vectors[torch.as_tensor(np.asarray(rows), dtype=torch.long)]
        return (chosen @ item_vectors.T).numpy().astype(np.float32)


class GRU4Rec(nn.Module):
    """GRU trên chuỗi mục, mất mát BPR với mẫu âm đồng đều."""

    kind = "sequence"

    def __init__(self, n_items, factors=64, hidden=64, dropout=0.2):
        super().__init__()
        self.n_items = n_items
        self.item = nn.Embedding(n_items + 1, factors, padding_idx=n_items)
        nn.init.normal_(self.item.weight, std=0.01)
        self.gru = nn.GRU(factors, hidden, batch_first=True)
        self.drop = nn.Dropout(dropout)
        self.project = nn.Linear(hidden, factors)

    def encode(self, sequences, pool=True):
        hidden, _ = self.gru(self.drop(self.item(sequences)))
        state = self.project(hidden)
        return state[:, -1] if pool else state

    def loss(self, batch):
        """Mất mát ở **mọi** vị trí của chuỗi, không chỉ vị trí cuối.

        Lần chạy đầu tôi lấy mất mát chỉ ở mục cuối, và GRU4Rec sập xuống
        NDCG 0,0200 với độ phủ 0,0006 — 22 mục cho 2.000 người. Nguyên nhân
        không phải kiến trúc mà là số tín hiệu: một chuỗi cho **một** cặp
        (dương, âm), nên 58.413 chuỗi x 8 vòng chỉ là 467 nghìn cặp, trong
        khi SASRec cùng dữ liệu nhận gấp 49 lần vì nó học ở mọi vị trí. Bài
        GRU4Rec gốc cũng dự đoán mục kế tiếp ở mọi bước của phiên, nên đây là
        trả lại đúng bài chứ không phải vá cho số đẹp.
        """
        sequences, positives, negatives = batch
        if positives.dim() == 1:
            state = self.encode(sequences)
            positive = (state * self.item(positives)).sum(-1)
            negative = (state * self.item(negatives)).sum(-1)
            return -nn.functional.logsigmoid(positive - negative).mean()
        state = self.encode(sequences, pool=False)
        positive = (state * self.item(positives)).sum(-1)
        negative = (state * self.item(negatives)).sum(-1)
        valid = (positives != self.n_items).float()
        loss = -(nn.functional.logsigmoid(positive - negative) * valid).sum()
        return loss / valid.sum().clamp(min=1.0)

    @torch.no_grad()
    def scores(self, sequences):
        state = self.encode(sequences)
        return (state @ self.item.weight[:-1].T).numpy().astype(np.float32)


class SASRec(nn.Module):
    """Self-attention nhân quả, dự đoán mục kế tiếp ở MỌI vị trí của chuỗi.

    Dự đoán ở mọi vị trí là lý do nó học nhanh hơn GRU4Rec với cùng số epoch:
    một chuỗi dài 50 cho 50 tín hiệu huấn luyện thay vì một.
    """

    kind = "sequence"

    def __init__(self, n_items, factors=64, blocks=2, heads=2, length=50,
                 dropout=0.2):
        super().__init__()
        self.n_items, self.length = n_items, length
        self.item = nn.Embedding(n_items + 1, factors, padding_idx=n_items)
        self.position = nn.Embedding(length, factors)
        nn.init.normal_(self.item.weight, std=0.01)
        nn.init.normal_(self.position.weight, std=0.01)
        self.drop = nn.Dropout(dropout)
        encoder = nn.TransformerEncoderLayer(
            factors, heads, dim_feedforward=factors * 2, dropout=dropout,
            batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(encoder, blocks)
        self.norm = nn.LayerNorm(factors)

    def encode(self, sequences, pool=True):
        length = sequences.shape[1]
        positions = torch.arange(length).unsqueeze(0)
        state = self.drop(self.item(sequences) + self.position(positions))
        mask = nn.Transformer.generate_square_subsequent_mask(length)
        padding = sequences == self.n_items
        state = self.encoder(state, mask=mask, src_key_padding_mask=padding,
                             is_causal=True)
        state = self.norm(state)
        return state[:, -1] if pool else state

    def loss(self, batch):
        sequences, positives, negatives = batch
        state = self.encode(sequences, pool=False)
        positive = (state * self.item(positives)).sum(-1)
        negative = (state * self.item(negatives)).sum(-1)
        valid = (positives != self.n_items).float()
        loss = -(nn.functional.logsigmoid(positive) * valid
                 + nn.functional.logsigmoid(-negative) * valid).sum()
        return loss / valid.sum().clamp(min=1.0)

    @torch.no_grad()
    def scores(self, sequences):
        state = self.encode(sequences)
        return (state @ self.item.weight[:-1].T).numpy().astype(np.float32)


class CL4SRec(SASRec):
    """SASRec cộng một mất mát tương phản giữa hai biến thể tăng cường.

    Ba phép tăng cường của bài gốc: cắt đoạn, che mục, đảo thứ tự một đoạn. Mất
    mát NT-Xent kéo hai biến thể của *cùng* một chuỗi lại gần nhau và đẩy biến
    thể của các chuỗi khác ra xa.
    """

    kind = "sequence"

    def __init__(self, *args, contrast=0.1, temperature=1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.contrast, self.temperature = contrast, temperature

    def augment(self, sequences, generator):
        out = sequences.clone()
        n_rows, length = out.shape
        pick = torch.randint(0, 3, (n_rows,), generator=generator)
        for row in range(n_rows):
            real = (out[row] != self.n_items).nonzero().flatten()
            if len(real) < 3:
                continue
            start, stop = int(real[0]), int(real[-1]) + 1
            span = stop - start
            if pick[row] == 0:  # cắt: giữ 70% đoạn cuối
                keep = max(2, int(span * 0.7))
                out[row, start:stop - keep] = self.n_items
            elif pick[row] == 1:  # che 30% vị trí
                hide = torch.randperm(span, generator=generator)[
                    :max(1, int(span * 0.3))]
                out[row, start + hide] = self.n_items
            else:  # đảo thứ tự một đoạn 40%
                size = max(2, int(span * 0.4))
                begin = start + int(torch.randint(
                    0, max(1, span - size + 1), (1,), generator=generator))
                piece = out[row, begin:begin + size].flip(0)
                out[row, begin:begin + size] = piece
        return out

    def loss(self, batch):
        sequences = batch[0]
        base = super().loss(batch)
        generator = torch.Generator().manual_seed(
            int(torch.randint(0, 2 ** 31 - 1, (1,)).item()))
        first = nn.functional.normalize(
            self.encode(self.augment(sequences, generator)), dim=-1)
        second = nn.functional.normalize(
            self.encode(self.augment(sequences, generator)), dim=-1)
        logits = first @ second.T / self.temperature
        target = torch.arange(len(sequences))
        contrast = nn.functional.cross_entropy(logits, target)
        return base + self.contrast * contrast


class NRMSStyle(nn.Module):
    """Bộ mã mục từ NỘI DUNG, bộ mã người đọc bằng attention trên lịch sử.

    Khác NRMS gốc ở một chỗ phải ghi rõ: NRMS chạy attention trên **từ** của
    tiêu đề, còn ở đây mục đã là một vectơ nội dung dày (LSA của TF-IDF), nên
    attention chỉ còn ở tầng người đọc. Tầng người đọc thì đúng NRMS: multi-head
    self-attention trên các mục đã đọc, rồi additive attention để gộp.

    Hệ quả đáng giá: mục chưa ai đọc vẫn có vectơ, nên mô hình này là mô hình
    duy nhất trong năm mô hình ở đây chấm được lát `new_item`.
    """

    kind = "sequence"

    def __init__(self, content, factors=64, heads=2, dropout=0.2):
        super().__init__()
        self.register_buffer("content", torch.as_tensor(content,
                                                        dtype=torch.float32))
        self.n_items = content.shape[0]
        self.encode_item = nn.Sequential(
            nn.Linear(content.shape[1], factors), nn.ReLU(),
            nn.Dropout(dropout))
        self.attention = nn.MultiheadAttention(factors, heads, dropout=dropout,
                                               batch_first=True)
        self.query = nn.Sequential(nn.Linear(factors, factors), nn.Tanh())
        self.weigh = nn.Linear(factors, 1, bias=False)

    def item_vectors(self, items):
        flat = items.reshape(-1).clamp(max=self.n_items - 1)
        return self.encode_item(self.content[flat]).reshape(
            *items.shape, -1)

    def encode(self, sequences):
        vectors = self.item_vectors(sequences)
        padding = sequences >= self.n_items
        attended, _ = self.attention(vectors, vectors, vectors,
                                     key_padding_mask=padding)
        weights = self.weigh(self.query(attended)).squeeze(-1)
        weights = weights.masked_fill(padding, float("-inf"))
        weights = torch.softmax(weights, dim=-1)
        weights = torch.nan_to_num(weights)
        return (attended * weights.unsqueeze(-1)).sum(dim=1)

    def loss(self, batch):
        sequences, positives, negatives = batch
        state = self.encode(sequences)
        if positives.dim() > 1:
            positives, negatives = positives[:, -1], negatives[:, -1]
        positive = (state * self.item_vectors(positives)).sum(-1)
        negative = (state * self.item_vectors(negatives)).sum(-1)
        return -nn.functional.logsigmoid(positive - negative).mean()

    @torch.no_grad()
    def scores(self, sequences):
        state = self.encode(sequences)
        every = self.encode_item(self.content)
        return (state @ every.T).numpy().astype(np.float32)


def normalized_adjacency(matrix):
    """Â = D^(−1/2) A D^(−1/2) cho đồ thị hai phía, dạng torch sparse."""
    n_users, n_items = matrix.shape
    coo = matrix.tocoo()
    rows = np.concatenate([coo.row, coo.col + n_users])
    cols = np.concatenate([coo.col + n_users, coo.row])
    values = np.ones(len(rows), dtype=np.float32)
    degree = np.bincount(rows, minlength=n_users + n_items).astype(np.float32)
    degree[degree == 0] = 1.0
    values = values / np.sqrt(degree[rows] * degree[cols])
    return torch.sparse_coo_tensor(
        np.vstack([rows, cols]), values,
        (n_users + n_items, n_users + n_items)).coalesce()
