"""Gợi ý theo nội dung, tính độ giống riêng cho từng mặt.

Giữ điểm số tách theo mặt thay vì cộng gộp ngay, vì phần giải thích cần biết
hai truyện giống nhau *ở chỗ nào* — nét vẽ, nhịp, chủ đề hay người sáng tác.
"""

import numpy as np


class FacetModel:
    def __init__(self, facets, index):
        self.index = index
        self.facets = {}
        self.available = {}
        for name, vectors in facets.items():
            self.facets[name] = vectors
            # Item thiếu mặt này (ví dụ không tải được ảnh bìa) có vector 0
            self.available[name] = np.linalg.norm(vectors, axis=1) > 1e-8

    @property
    def names(self):
        return list(self.facets)

    def profile(self, item_indices, weights=None):
        """Hồ sơ người đọc: trung bình có trọng số các truyện họ đã thích."""
        profile = {}
        item_indices = np.asarray(item_indices, dtype=int)
        for name, vectors in self.facets.items():
            usable = item_indices[self.available[name][item_indices]]
            if usable.size == 0:
                profile[name] = None
                continue
            block = vectors[usable]
            if weights is not None:
                weight = np.asarray(weights, dtype=float)[
                    self.available[name][item_indices]]
                centre = (block * weight[:, None]).sum(axis=0) / max(weight.sum(), 1e-9)
            else:
                centre = block.mean(axis=0)
            norm = np.linalg.norm(centre)
            profile[name] = centre / norm if norm > 1e-9 else None
        return profile

    def score(self, profile, candidates=None):
        """Điểm cosine theo từng mặt. Mặt thiếu trả về NaN để tầng trên biết
        mà chuẩn hoá lại trọng số thay vì coi như điểm 0."""
        scores = {}
        for name, vectors in self.facets.items():
            centre = profile.get(name)
            block = vectors if candidates is None else vectors[candidates]
            mask = (self.available[name] if candidates is None
                    else self.available[name][candidates])
            if centre is None:
                scores[name] = np.full(block.shape[0], np.nan)
                continue
            values = block @ centre
            values[~mask] = np.nan
            scores[name] = values
        return scores

    def coherence(self, item_indices):
        """Độ chụm của một tập truyện trong từng mặt.

        Người đọc chọn truyện theo nét vẽ sẽ có tập truyện chụm ở mặt art;
        người chọn theo cốt sẽ chụm ở mặt theme. Đây là cơ sở để học trọng số
        từng mặt riêng cho mỗi người.
        """
        item_indices = np.asarray(item_indices, dtype=int)
        result = {}
        for name, vectors in self.facets.items():
            usable = item_indices[self.available[name][item_indices]]
            if usable.size < 2:
                result[name] = np.nan
                continue
            block = vectors[usable]
            centre = block.mean(axis=0)
            norm = np.linalg.norm(centre)
            result[name] = float(norm) if norm > 1e-9 else 0.0
        return result
