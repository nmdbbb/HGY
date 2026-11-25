"""Nạp artifact một lần và phục vụ gợi ý kèm lý do."""

import json
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from truyen_reco.explain.explainer import Explainer
from truyen_reco.features.art import FEATURE_NAMES
from truyen_reco.models.cold_start import onboarding_grid, profile_from_picks
from truyen_reco.models.collaborative import ALS, PopularityModel
from truyen_reco.models.content_based import FacetModel
from truyen_reco.models.hybrid import HybridRecommender
from truyen_reco.models.two_tower import RoutedTwoTower, ServedTwoTower

DEFAULT_DIR = Path("data/processed")

# Ba mô hình hai tháp đã lưu là ba sản phẩm, không phải một tốt hai kém. Phép
# quét ở mục 9.7 `docs/research-notes.md` cho thấy chúng là ba điểm trên **cùng
# một núm**: phục vụ bao nhiêu phần của đầu độ phổ biến. Đổi đơn điệu, không có
# điểm nào được cả hai đầu.
#   feed      chính xác nhất (+48% NDCG so với ALS trên nhánh taste) nhưng
#             không bao giờ gợi truyện mới — đầu độ phổ biến dìm truyện ít đọc.
#   discover  phủ catalog gấp 3,3× và là cấu hình duy nhất chấm được truyện mới
#             (new_item 0,0173 vs 0,0000 của mọi baseline), manhua trong top-10
#             lên 0,11 — đổi bằng 6% độ chính xác ở đầu phân phối.
#   gated     mỗi người một mức thích-truyện-phổ-biến riêng:
#             g_u = base(log(1+n_u)) + delta_u, với `delta_u` bị bỏ ngẫu nhiên
#             p=0,3 lúc huấn luyện (DropoutNet áp cho cổng, mục 9.8-9.9). Tương
#             quan +0,84 với độ nổi những bộ người đó đã đọc.
#             Số đo trên CÁC TRỌNG SỐ ĐÃ LƯU (một lần huấn luyện, KHÔNG phải
#             kết luận — phần trăm ở đây bị phóng đại, xem phép kiểm bên dưới):
#               all         0,1169 -> 0,1202
#               sparse_user 0,1268 -> 0,1387
#               warm        0,1039 -> 0,1051
#               new_user    0,3771 -> 0,3533
#               phủ catalog 0,0336 -> 0,0347   manhua 0,01 -> 0,02
#             Phép kiểm ghép cặp 4 seed (mục 9.9) chốt lại: `all` +1,9%
#             (t=+4,27, 4/4 seed) và `new_user` −5,2% (t=−5,79, 0/4) đều LẬP
#             ĐƯỢC; `sparse_user` +4,4% và `warm` +1,9% thì 4/4 seed dương
#             nhưng t=2,60/2,62 CHƯA vượt ngưỡng df=3 là 3,18 — ghi là chưa lập
#             được, không gọi là "có xu hướng".
#   auto      ĐỊNH TUYẾN, mặc định. `feed` cho người chưa có lịch sử, `gated`
#             cho người đã có. Không phải mô hình thứ tư: dùng lại đúng hai
#             blob trên. Đo được (ghép cặp hoàn hảo, cùng trọng số chỉ khác
#             đường chấm) nó lấy MAX ở mọi lát — all 0,1202 / new_user 0,3771 /
#             sparse_user 0,1387 / warm 0,1051 — tức không kém `feed` ở đâu và
#             hơn ở lát `all`. Ranh giới là "CÓ hay KHÔNG CÓ lịch sử", không
#             phải "ít hay nhiều": `sparse_user` nghiêng về cổng. Đa dạng nước
#             không mất gì: ở 800 người, 1,742 so với 1,741 nước của `feed`
#             (ở 150 người có vẻ kém 1,6 điểm — đó là nhiễu lấy mẫu).
# `hybrid` giữ lại hệ hai tầng cũ để so sánh, không phải mặc định.
#
# Ghi cho rõ để không ai đọc lẫn: quy tắc chốt trước khi chạy là "chỉ đổi mặc
# định sang `gated` nếu `all` VÀ `sparse_user` đều vượt t=3,18". `sparse_user`
# không vượt, nên `gated` KHÔNG thành mặc định — quy tắc được giữ nguyên, không
# diễn giải lại. `auto` là một đối tượng khác và được xét theo tiêu chí riêng
# của nó: không hồi quy ở bất kỳ lát nào, cộng một mức hơn đã lập được ở `all`.
ENGINES = ("auto", "feed", "discover", "gated", "hybrid")
TOWER_ENGINES = ("feed", "discover", "gated")


class RecommenderService:
    def __init__(self, out_dir=DEFAULT_DIR):
        out_dir = Path(out_dir)
        blob = np.load(out_dir / "facets.npz", allow_pickle=False)
        meta = json.loads((out_dir / "index.json").read_text(encoding="utf-8"))

        self.facet_vectors = {k[len("facet_"):]: blob[k]
                              for k in blob.files if k.startswith("facet_")}
        self.item_index = {int(k): v for k, v in meta["item_index"].items()}
        self.user_index = meta["user_index"]
        self.media = pd.read_parquet(out_dir / "media_core.parquet")

        als = ALS()
        als.user_factors = blob["user_factors"]
        als.item_factors = blob["item_factors"]
        popularity = PopularityModel()
        popularity.scores = blob["popularity"]

        self.facets = FacetModel(self.facet_vectors, self.item_index)
        self.model = HybridRecommender(
            als, self.facets, popularity, self.user_index, self.item_index,
        ).set_creator_groups(self.media, self.item_index)

        train = pd.read_parquet(out_dir / "train.parquet")
        self.history = {
            user: [self.item_index[m] for m in items if m in self.item_index]
            for user, items in train.groupby("user")["media_id"].apply(list).items()
        }
        self.model.fit_weights(self.history)

        art_raw = blob["art_raw"] if "art_raw" in blob.files else None
        art_ids = blob["art_ids"] if "art_ids" in blob.files else None
        if art_raw is not None and art_raw.size == 0:
            art_raw = art_ids = None
        self.explainer = Explainer(self.facets, self.media, self.item_index,
                                   art_raw=art_raw, art_ids=art_ids,
                                   art_names=meta.get("art_feature_names",
                                                      FEATURE_NAMES))
        # Hai tháp: nạp nếu có file, không có thì rơi về hệ hai tầng và nói rõ
        self.towers = {}
        for engine in TOWER_ENGINES:
            path = out_dir / f"two_tower_{engine}.pt"
            if path.exists():
                self.towers[engine] = ServedTwoTower.load(path)
        # `auto` không phải mô hình thứ tư — nó dùng lại đúng hai blob đã nạp,
        # nên không tốn thêm bộ nhớ và không thêm độ trễ nào ngoài một phép tra.
        if "feed" in self.towers and "gated" in self.towers:
            self.towers["auto"] = RoutedTwoTower(self.towers["feed"],
                                                 self.towers["gated"])

        self.titles = dict(zip(self.media["media_id"], self.media["title"]))
        self.countries = dict(zip(self.media["media_id"], self.media["country"]))
        self.covers = dict(zip(self.media["media_id"], self.media["cover_url"]))
        self.reverse_item = {v: k for k, v in self.item_index.items()}

    def onboarding(self, n_options=12):
        positions = onboarding_grid(self.facets, self.media, self.item_index,
                                    n_options=n_options)
        return [self._card(self.reverse_item[p]) for p in positions]

    def _card(self, media_id):
        return {
            "media_id": int(media_id),
            "title": self.titles.get(media_id) or f"#{media_id}",
            "country": self.countries.get(media_id),
            "cover_url": self.covers.get(media_id),
        }

    def _tower_items(self, engine, user_id, history, k):
        """Gợi ý bằng hai tháp, trả về kèm điểm từng mặt để giải thích.

        Hai tháp chấm bằng một vector 64 chiều nên bản thân nó không nói được
        "giống ở chỗ nào". Điểm từng mặt vẫn tính từ `FacetModel` trên chính
        những truyện hai tháp đã chọn — tức lời giải thích là *hậu kiểm* trên
        cùng bốn mặt, không phải phân rã điểm của mô hình. Phải ghi rõ như vậy
        trong báo cáo, không được trình bày như thể mô hình tự giải thích.
        """
        tower = self.towers[engine]
        known = user_id if user_id in tower.user_index else None
        positions = tower.top_k(known, history, k=k)
        items = [self.reverse_item[int(p)] for p in positions]

        weights = self.model.user_weights(history) if history else dict(
            self.model.base_weights)
        profile = self.facets.profile(history) if history else {}
        scored = (self.facets.score(profile, np.asarray(positions, dtype=int))
                  if profile else
                  {name: np.full(len(positions), np.nan)
                   for name in self.facets.names})
        facet_scores = {name: {int(p): float(values[row])
                               for row, p in enumerate(positions)}
                        for name, values in scored.items()}
        return items, {"facet_scores": facet_scores, "weights": weights}

    def recommend(self, user_id=None, picked_ids=None, pace_preference=None,
                  top_k=10, countries=None, engine="auto"):
        picked = [self.item_index[m] for m in (picked_ids or [])
                  if m in self.item_index]
        history = self.history.get(user_id, []) if user_id else []
        profile = None

        if picked:
            profile = profile_from_picks(self.facets, picked,
                                         pace_preference=pace_preference)
            history = history or picked

        wide = top_k * 3 if countries else top_k
        # `auto` cần cả hai blob. Thiếu `gated` thì rơi về `feed`, không rơi
        # thẳng xuống hệ hai tầng cũ: `feed` vẫn tốt hơn nó ở mọi lát.
        if engine == "auto" and engine not in self.towers and "feed" in self.towers:
            engine = "feed"
        if engine in self.towers:
            items, detail = self._tower_items(engine, user_id, history, wide)
        else:
            items, detail = self.model.recommend(
                user_id, k=wide, history=history, extra_profile=profile,
                return_detail=True)
            engine = "hybrid"

        if countries:
            wanted = set(countries)
            items = [m for m in items if self.countries.get(m) in wanted][:top_k]

        history_ids = [self.reverse_item[p] for p in history]
        results = []
        for media_id in items[:top_k]:
            position = self.item_index.get(media_id)
            scores = {name: values.get(position)
                      for name, values in detail["facet_scores"].items()}
            card = self._card(media_id)
            card["reason"] = self.explainer.explain(
                media_id, history_ids, scores, detail["weights"])
            card["breakdown"] = self.explainer.facet_breakdown(
                scores, detail["weights"])
            results.append(card)
        return {"items": results, "facet_weights": detail["weights"],
                "engine": engine}


@lru_cache(maxsize=1)
def get_service(out_dir=str(DEFAULT_DIR)):
    return RecommenderService(out_dir)
