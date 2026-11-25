from typing import Literal

from pydantic import BaseModel, Field


class ComicCard(BaseModel):
    media_id: int
    title: str
    country: str | None = None
    cover_url: str | None = None
    reason: str | None = None
    breakdown: str | None = None


class OnboardingResponse(BaseModel):
    items: list[ComicCard]


class RecommendRequest(BaseModel):
    user_id: str | None = None
    picked_ids: list[int] = Field(default_factory=list,
                                  description="Truyện người mới chọn ở màn onboarding")
    pace_preference: Literal["short", "long"] | None = None
    countries: list[Literal["JP", "KR", "CN"]] | None = Field(
        default=None, description="Lọc manga (JP), manhwa (KR), manhua (CN)")
    top_k: int = 10
    engine: Literal["auto", "feed", "discover", "gated", "hybrid"] = Field(
        default="auto",
        description="auto (mặc định): định tuyến — feed cho người chưa có lịch "
                    "sử đọc, gated cho người đã có. Đo được nó lấy max ở mọi "
                    "lát, không kém feed ở đâu (mục 9.9 docs/research-notes.md). "
                    "Ba engine dưới là ba điểm trên cùng một núm: phục vụ bao "
                    "nhiêu phần của đầu độ phổ biến (mục 9.7). "
                    "feed: chính xác nhất với người đọc mới, không gợi truyện mới. "
                    "discover: phủ danh mục gấp 3,3×, chấm được truyện mới, "
                    "nhiều manhua hơn (0,11 vs 0,01), đổi bằng 6% độ chính xác. "
                    "gated: mỗi người một mức thích-truyện-phổ-biến riêng — hơn "
                    "feed 1,9% ở lát all (đã lập được, t=4,27 với 4 seed ghép "
                    "cặp) nhưng kém 5,2% ở người đọc mới hoàn toàn (t=-5,79). "
                    "hybrid: hệ hai tầng cũ, để so sánh.")


class RecommendResponse(BaseModel):
    items: list[ComicCard]
    facet_weights: dict[str, float]
    engine: str = Field(description="Mô hình thực sự đã dùng — rơi về 'hybrid' "
                                    "nếu chưa có trọng số hai tháp")
