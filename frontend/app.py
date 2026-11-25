"""Demo: chọn vài ảnh bìa, nhận gợi ý kèm lý do.

    streamlit run frontend/app.py
"""

import sys
from pathlib import Path

import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from truyen_reco.api.service import RecommenderService  # noqa: E402

COUNTRY_LABEL = {"JP": "Manga", "KR": "Manhwa", "CN": "Manhua"}
FACET_LABEL = {"art": "nét vẽ", "pace": "nhịp truyện",
               "theme": "chủ đề", "creator": "người sáng tác"}


@st.cache_resource
def load_service():
    return RecommenderService()


st.set_page_config(page_title="Gợi ý truyện tranh", layout="wide")
st.title("Gợi ý truyện tranh")
st.caption("Manga · Manhwa · Manhua — chọn theo nét vẽ, nhịp truyện và người sáng tác")

try:
    service = load_service()
except FileNotFoundError:
    st.error("Chưa có artifact. Chạy `python scripts/train.py` trước.")
    st.stop()

if "picked" not in st.session_state:
    st.session_state.picked = []

with st.sidebar:
    st.header("Tuỳ chọn")
    wanted = st.multiselect(
        "Loại truyện", options=list(COUNTRY_LABEL),
        format_func=lambda c: COUNTRY_LABEL[c], default=list(COUNTRY_LABEL))
    pace = st.radio("Thích bộ như thế nào?",
                    options=[None, "short", "long"],
                    format_func=lambda v: {None: "Không quan trọng",
                                           "short": "Ngắn gọn, đã hoàn",
                                           "long": "Dài, nhiều chương"}[v])
    top_k = st.slider("Số gợi ý", 5, 30, 10)
    # Các lựa chọn là những đánh đổi đo được, không phải mức "chất lượng":
    # feed chính xác hơn 48% so với ALS nhưng không bao giờ gợi truyện mới;
    # discover gợi được truyện mới và nhiều manhua hơn, đổi bằng 6% độ chính xác;
    # gated hơn ở người đã có lịch sử và kém ở người mới, nên mặc định là bản
    # định tuyến giữa hai cái.
    engine = st.radio(
        "Kiểu gợi ý",
        options=["auto", "feed", "discover", "gated", "hybrid"],
        format_func=lambda v: {
            "auto": "Tự chọn theo lịch sử của bạn (mặc định)",
            "feed": "Sát gu nhất (bỏ truyện mới)",
            "discover": "Khám phá (có truyện mới, nhiều manhua)",
            "gated": "Theo mức thích truyện phổ biến của bạn",
            "hybrid": "Hai tầng cũ (để so sánh)"}[v])
    CAPTION = {
        "auto": "Người chưa có lịch sử đọc dùng 'sát gu nhất', người đã có dùng "
                "'theo mức thích truyện phổ biến'. Đo được bản này lấy kết quả "
                "tốt nhất ở mọi nhóm người đọc, không kém ở nhóm nào.",
        "gated": "Học riêng cho mỗi người mức thích truyện phổ biến. Đo trên 4 "
                 "lần huấn luyện ghép cặp: hơn 'sát gu' 1,9% trên tổng, nhưng "
                 "kém 5,2% với người mới hoàn toàn. Giá trị nằm ở người đã có "
                 "lịch sử đọc — vì vậy mặc định là bản tự chọn ở trên.",
    }
    if engine in CAPTION:
        st.caption(CAPTION[engine])
    if st.button("Chọn lại từ đầu"):
        st.session_state.picked = []
        st.rerun()

st.subheader("1. Chọn vài bìa truyện bạn thấy hợp mắt")
st.caption("Lưới này trải đều các cụm nét vẽ khác nhau, nên vài lựa chọn đã đủ "
           "để đoán gu — không cần lịch sử đọc.")

grid = service.onboarding(n_options=12)
columns = st.columns(6)
for position, card in enumerate(grid):
    with columns[position % 6]:
        if card["cover_url"]:
            st.image(card["cover_url"], use_container_width=True)
        label = COUNTRY_LABEL.get(card["country"], "?")
        chosen = card["media_id"] in st.session_state.picked
        if st.button(("✓ " if chosen else "") + f"{card['title'][:28]}",
                     key=f"pick-{card['media_id']}", use_container_width=True):
            if chosen:
                st.session_state.picked.remove(card["media_id"])
            else:
                st.session_state.picked.append(card["media_id"])
            st.rerun()
        st.caption(label)

if not st.session_state.picked:
    st.info("Chọn ít nhất một truyện để nhận gợi ý.")
    st.stop()

st.divider()
st.subheader(f"2. Gợi ý cho bạn ({len(st.session_state.picked)} truyện đã chọn)")

result = service.recommend(picked_ids=st.session_state.picked,
                           pace_preference=pace, top_k=top_k,
                           countries=wanted or None, engine=engine)

if result["engine"] != engine:
    st.warning(f"Chưa có trọng số hai tháp cho **{engine}** — đang dùng "
               f"`{result['engine']}`. Chạy "
               "`python scripts/train_embedding.py --save` để có.")

weights = result["facet_weights"]
if weights:
    ranked = sorted(weights.items(), key=lambda kv: -kv[1])
    top_facet = FACET_LABEL.get(ranked[0][0], ranked[0][0])
    st.caption("Hệ đang ưu tiên **" + top_facet + "** cho lựa chọn của bạn — "
               + " · ".join(f"{FACET_LABEL.get(k, k)} {v:.2f}" for k, v in ranked))

for card in result["items"]:
    left, right = st.columns([1, 5])
    with left:
        if card["cover_url"]:
            st.image(card["cover_url"], use_container_width=True)
    with right:
        st.markdown(f"**{card['title']}**  ·  {COUNTRY_LABEL.get(card['country'], '')}")
        st.write(card["reason"])
        if card["breakdown"]:
            st.caption(card["breakdown"])
    st.divider()
