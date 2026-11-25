from fastapi import APIRouter, HTTPException

from truyen_reco.api.schemas import OnboardingResponse, RecommendRequest, RecommendResponse
from truyen_reco.api.service import get_service

router = APIRouter(tags=["recommend"])


@router.get("/onboarding", response_model=OnboardingResponse)
def onboarding(n: int = 12):
    """Lưới ảnh bìa trải đều các cụm nét vẽ, cho người đọc mới chọn."""
    try:
        return OnboardingResponse(items=get_service().onboarding(n_options=n))
    except FileNotFoundError:
        raise HTTPException(503, "Chưa có artifact — chạy scripts/train.py trước.")


@router.post("/recommend", response_model=RecommendResponse)
def recommend(request: RecommendRequest):
    try:
        service = get_service()
    except FileNotFoundError:
        raise HTTPException(503, "Chưa có artifact — chạy scripts/train.py trước.")

    if not request.user_id and not request.picked_ids:
        raise HTTPException(400, "Cần user_id hoặc picked_ids từ màn onboarding.")

    result = service.recommend(
        user_id=request.user_id,
        picked_ids=request.picked_ids,
        pace_preference=request.pace_preference,
        top_k=request.top_k,
        countries=request.countries,
        engine=request.engine,
    )
    return RecommendResponse(**result)
