"""Ghép lọc cộng tác với các mặt nội dung, hai tầng: lấy ứng viên rồi xếp lại.

Trọng số từng mặt học riêng cho mỗi người đọc: có người chọn truyện theo nét
vẽ, có người theo cốt. Dùng một bộ trọng số chung cho tất cả là bỏ mất khác
biệt đó.
"""

import numpy as np

DEFAULT_WEIGHTS = {"theme": 1.0, "pace": 0.6, "creator": 0.8, "art": 0.7}


def zscore(values):
    finite = np.isfinite(values)
    if finite.sum() < 2:
        return np.zeros_like(values)
    mean = values[finite].mean()
    std = values[finite].std()
    out = np.zeros_like(values)
    out[finite] = (values[finite] - mean) / max(std, 1e-9)
    return out


class HybridRecommender:
    def __init__(self, als, facet_model, popularity, user_index, item_index,
                 cf_weight=1.0, n_candidates=400, personalize=True,
                 creator_penalty=0.35, facet_weights=None):
        self.als = als
        self.facets = facet_model
        self.popularity = popularity
        self.user_index = user_index
        self.item_index = item_index
        self.reverse_item = {v: k for k, v in item_index.items()}
        self.cf_weight = cf_weight
        self.n_candidates = n_candidates
        self.personalize = personalize
        self.creator_penalty = creator_penalty
        self.base_weights = dict(facet_weights or DEFAULT_WEIGHTS)
        self.baseline_coherence = {}
        self.creator_groups = None

    def fit_weights(self, user_items):
        """Chuẩn tham chiếu độ chụm: dùng để biết một người chụm hơn mức
        thường thấy ở mặt nào."""
        samples = {name: [] for name in self.facets.names}
        for indices in list(user_items.values())[:4000]:
            if len(indices) < 3:
                continue
            for name, value in self.facets.coherence(indices).items():
                if np.isfinite(value):
                    samples[name].append(value)
        self.baseline_coherence = {
            name: (float(np.mean(values)) if values else 0.0)
            for name, values in samples.items()
        }
        return self

    def user_weights(self, item_indices):
        weights = dict(self.base_weights)
        if not self.personalize or len(item_indices) < 3:
            return weights
        coherence = self.facets.coherence(item_indices)
        lifts = {}
        for name, value in coherence.items():
            if not np.isfinite(value):
                continue
            lifts[name] = value - self.baseline_coherence.get(name, 0.0)
        if not lifts:
            return weights
        scale = max(np.std(list(lifts.values())), 1e-6)
        for name, lift in lifts.items():
            # Nhân tố trong [0.5, 2.0]: cá nhân hoá nhưng không lật ngược
            # hoàn toàn trọng số chung khi người đọc có ít lịch sử.
            weights[name] = weights.get(name, 1.0) * float(
                np.clip(np.exp(lift / (2 * scale)), 0.5, 2.0))
        return weights

    def set_creator_groups(self, media, index):
        """Gom truyện theo người sáng tác để chặn hiện tượng danh sách gợi ý
        toàn tác phẩm của cùng một hoạ sĩ."""
        groups = {}
        for row in media.itertuples(index=False):
            position = index.get(row.media_id)
            if position is None:
                continue
            key = tuple(sorted(set(row.staff_art_ids) | set(row.staff_story_ids)))
            groups[position] = key or (("none", position),)
        self.creator_groups = groups
        return self

    def _candidates(self, user_position, seen):
        if user_position is not None and self.als.user_factors is not None:
            scores = self.als.user_factors[user_position] @ self.als.item_factors.T
        else:
            scores = self.popularity.scores.astype(float)
        scores = scores.copy()
        scores[list(seen)] = -np.inf
        size = min(self.n_candidates, np.isfinite(scores).sum())
        if size <= 0:
            return np.array([], dtype=int), scores
        picked = np.argpartition(-scores, size - 1)[:size]
        return picked, scores

    def recommend(self, user, k=10, history=None, extra_profile=None,
                  return_detail=False):
        user_position = self.user_index.get(user)
        history = list(history or [])
        seen = set(history)

        candidates, cf_scores = self._candidates(user_position, seen)
        if candidates.size == 0:
            return ([], {}) if return_detail else []

        profile = (extra_profile if extra_profile is not None
                   else self.facets.profile(history) if history else {})
        weights = self.user_weights(history) if history else dict(self.base_weights)

        facet_scores = (self.facets.score(profile, candidates) if profile
                        else {name: np.full(len(candidates), np.nan)
                              for name in self.facets.names})

        total = np.zeros(len(candidates))
        if user_position is not None and self.als.user_factors is not None:
            total += self.cf_weight * zscore(cf_scores[candidates])
        else:
            total += 0.4 * zscore(np.log1p(self.popularity.scores[candidates]))

        facet_total = np.zeros(len(candidates))
        weight_sum = np.zeros(len(candidates))
        standardized = {}
        for name, values in facet_scores.items():
            weight = weights.get(name, 0.0)
            if weight <= 0:
                continue
            standardized[name] = values
            present = np.isfinite(values)
            facet_total[present] += weight * zscore(values)[present]
            weight_sum[present] += weight

        # Trung bình có trọng số, không phải tổng: truyện thiếu ảnh bìa chỉ
        # được chấm trên các mặt nó thực có, thay vì bị cộng thiếu một số hạng.
        active = weight_sum > 0
        facet_total[active] /= weight_sum[active]
        total += facet_total

        # Chọn tham lam, mỗi lần trừ đi hình phạt cho người sáng tác đã xuất
        # hiện: gợi ý toàn tác phẩm của cùng một hoạ sĩ thì đúng mà vô dụng.
        order = list(np.argsort(-total)[:max(4 * k, 60)])
        chosen, used_creators = [], {}
        while len(chosen) < k and order:
            best_position, best_score = None, -np.inf
            for position in order:
                group = (self.creator_groups or {}).get(int(candidates[position]))
                penalty = self.creator_penalty * used_creators.get(group, 0) if group else 0.0
                score = total[position] - penalty
                if score > best_score:
                    best_score, best_position = score, position
            order.remove(best_position)
            item = int(candidates[best_position])
            chosen.append(item)
            group = (self.creator_groups or {}).get(item)
            if group:
                used_creators[group] = used_creators.get(group, 0) + 1

        if not return_detail:
            return [self.reverse_item[i] for i in chosen]

        detail = {
            "weights": weights,
            "facet_scores": {name: dict(zip(candidates.tolist(), values.tolist()))
                             for name, values in standardized.items()},
        }
        return [self.reverse_item[i] for i in chosen], detail
