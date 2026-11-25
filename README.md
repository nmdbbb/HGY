# Hệ gợi ý Truyện tranh — Manga · Manhwa · Manhua (IT4613)

Hệ gợi ý lai ghép cho truyện tranh châu Á, gợi ý theo **nét vẽ**, **nhịp truyện**, **chủ đề** và **người sáng tác**, có giải thích lý do và xử lý được người đọc mới.

Dữ liệu tự thu từ [AniList](https://anilist.co) — chọn nguồn này vì nó là nguồn duy nhất tách được manga / manhwa / manhua (`countryOfOrigin`), có tiến độ đọc theo chương, và tách riêng vai tác giả với hoạ sĩ. Lý do đầy đủ: [docs/data-sources.md](docs/data-sources.md).

## Chạy

```bash
pip install -e ".[dev,demo]"

# 1. Thu thập dữ liệu (resume được; lần đầy đủ ~6 tiếng, chạy qua đêm)
python scripts/crawl_anilist.py discover
python scripts/crawl_anilist.py users --target 10000 --weights "KR=0.35,CN=0.35,JP=0.30"
python scripts/crawl_anilist.py lists
python scripts/crawl_anilist.py media

# 2. Đặc trưng + ALS (gồm tải ảnh bìa và rút đặc trưng nét vẽ)
python scripts/train.py

# 3. Hai tháp: huấn luyện, đánh giá, lưu trọng số cho API
python scripts/train_embedding.py --epochs 30 --save              # mục tiêu "sắp đọc gì"
python scripts/train_embedding.py --split taste --epochs 15 --save  # mục tiêu "gu"

# 4. Đánh giá sâu
python scripts/train_embedding.py --extra-variants --facet-ablation  # ablation
python scripts/diagnose_data.py && python scripts/analyze_deeper.py  # chẩn đoán
python scripts/measure_product.py                                    # bậc 4
python scripts/diagnose_country.py                                   # vì sao thiếu manhua
python scripts/diagnose_trend_gate.py                                # cổng độ phổ biến
python scripts/diagnose_country_threshold.py                         # ngưỡng "≥2 nước" có đúng không

# Chấm lại trọng số đã lưu mà KHÔNG huấn luyện lại (0,2 phút thay vì 44 phút).
# Dùng cho mọi thay đổi ở vế phục vụ: huấn luyện lại là đổi cả seed, làm phép
# so hết còn ghép cặp.
python scripts/eval_saved.py two_tower_feed.pt two_tower_gated.pt
python scripts/eval_saved.py two_tower_gated.pt --gate-sweep          # quét núm
python scripts/eval_saved.py two_tower_feed.pt two_tower_gated.pt \
       --route two_tower_feed.pt two_tower_gated.pt                   # engine auto

# Phép kiểm ghép cặp theo seed — bắt buộc khi chênh lệch cần kết luận nhỏ hơn
# nhiễu khởi tạo. Mọi biến thể chạy qua CÙNG danh sách seed, biến thể đầu làm mốc.
python scripts/train_embedding.py --epochs 15 --variants feed feed_gated --seeds 0 1 2 3

# 4b. Nhánh đối chứng Goodreads — CÙNG pipeline, chỉ đổi tầng đọc dữ liệu.
# Dùng để kiểm phương pháp có lặp lại trên bộ dữ liệu khác hay không, không
# phải để tìm bộ nào cho điểm cao hơn. Giấy phép học thuật + trích dẫn bắt
# buộc: xem docs/goodreads-arm.md.
JOBS=16 bash scripts/fetch_goodreads.sh                   # tải (resume được)
python scripts/scale_curve_goodreads.py                   # quy mô nào khớp AniList
python scripts/train.py --source goodreads --out-dir data/processed_gr        --cover-dir data/raw/covers_gr --sample-users 160000 --reuse-ingest
python scripts/diagnose_goodreads.py                      # so bản chất hai bộ
python scripts/train_embedding.py --out-dir data/processed_gr --facet-ablation
python scripts/diagnose_trend_gate.py --out-dir data/processed_gr   # cổng học gì trên bộ này

# Nghĩa vụ đã chốt trước: dự đoán nào BỊ BÁC BỎ ở quy mô khớp thì phải chạy lại
# đúng nó ở quy mô đầy đủ, để biết nó bị bác bỏ vì dữ liệu hay vì quy mô (P3 và
# P5, xem docs/goodreads-arm.md mục 7 và 9). Dùng lại parquet thô đã đọc nên
# không phải tải lại dữ liệu nguồn.
mkdir -p data/processed_gr_full
cp data/processed_gr/interactions.parquet data/processed_gr/media.parquet data/processed_gr_full/
python scripts/train.py --source goodreads --out-dir data/processed_gr_full --cover-dir data/raw/covers_gr --reuse-ingest
python scripts/train_embedding.py --out-dir data/processed_gr_full --epochs 15 --variants feed feed_gated --save --out-name embedding_temporal_p3p5.json
# --out-name là bắt buộc trên mọi lần chạy phụ: tên mặc định
# embedding_<split>.json bị GHI ĐÈ mỗi lần, nên chạy một tập biến thể khác
# trên cùng thư mục sẽ xoá kết quả cũ mà không báo gì. Script cũng ghi điểm
# kiểm tra sau MỖI biến thể, nên một lần chạy bị giết giữa chừng vẫn giữ
# được phần đã đánh giá xong (hơn bốn giờ cho hai biến thể ở quy mô này).

# 4c. Nhánh "hướng ban đầu" — CÙNG dữ liệu, đổi GIAO THỨC (đề xuất đồ án):
# dương = rating >= 4, k-core 10/5, chia 70/10/20 theo thời gian của TỪNG người,
# và tuỳ chọn gộp tập thành bộ truyện. Dùng để đối chiếu với mốc HGN và để biết
# phần nào của những con số thấp là tính chất của giao thức, không của mô hình.
# Xem docs/protocol-arm.md.
python scripts/build_series_map.py                  # bản đồ tác phẩm -> bộ
python scripts/prepare_protocol.py                  # dựng cả hai biến thể gộp/không
python scripts/run_protocol_baselines.py --dir data/processed_hgn_none
python scripts/run_protocol_baselines.py --dir data/processed_hgn_volume

# Chọn siêu tham số trên VAL rồi đọc test đúng một lần (mục 12 và 13 của
# docs/protocol-arm.md). Chạy cho CẢ HAI biến thể: con số để nộp là của bản
# gộp tập, vì đó là cấu hình đề xuất yêu cầu.
python scripts/tune_protocol.py --dir data/processed_hgn_none
python scripts/tune_protocol.py --dir data/processed_hgn_volume

# Hai phép đo trả lời "con số vừa chọn có nghĩa không", cả hai CHỈ nhìn val:
#  - tuning_noise.py: sai số của một con số val (±0,007) so với hiệu số ghép
#    cặp theo mẫu (±0,0001) — cho biết lưới phân giải được tới đâu, và cấu
#    hình val chọn có nằm trong một mảng bẹt hay không.
#  - diagnose_shrink.py: vì sao hạ trọng số cặp mỏng làm xấu đi. Nó cũng bác
#    bỏ lời giải thích đầu tiên của chính tôi (mục 14).
python scripts/tuning_noise.py --dir data/processed_hgn_volume
python scripts/diagnose_shrink.py --dir data/processed_hgn_none

# 4d. SO HAI HƯỚNG (docs/so-sanh-hai-huong.md). Con số thô của hai nhánh không
# so thẳng được vì năm thứ khác nhau cùng lúc, nên script này chạy đúng
# Item-KNN/Session-KNN của nhánh giao thức DƯỚI giao thức của nhánh dữ liệu —
# cùng train/test, cùng lát, cùng phép loại "đã đọc". Khác biệt giao thức bị
# triệt tiêu, thứ còn lại chỉ là họ mô hình.
python scripts/crossarm_knn.py --out-dir data/processed            # AniList
python scripts/crossarm_knn.py --out-dir data/processed_gr         # Goodreads khớp quy mô
python scripts/crossarm_knn.py --out-dir data/processed_gr_full --lengths 2 5 10

# `--sample-users 160000` không phải con số tròn cho đẹp: đó là mức ĐO ĐƯỢC để
# khớp AniList cả về số tương tác (0,96x) lẫn cỡ danh mục (0,99x) mà vẫn giữ
# nguyên hình dạng lịch sử riêng của Goodreads (mục 4 docs/goodreads-arm.md).
# `--reuse-ingest` dùng lại parquet đã đọc, và tự từ chối nếu tầng ingest đã
# sửa sau khi cache được ghi.

# 5. Demo
streamlit run frontend/app.py
uvicorn truyen_reco.api.main:app --reload   # hoặc API, xem /docs
```

Bước 3 bắt buộc nếu muốn demo chạy mô hình hai tháp: không có
`data/processed/two_tower_*.pt` thì API tự rơi về hệ hai tầng cũ và ghi rõ
`engine: "hybrid"` trong phản hồi.

## Thiết kế

Mỗi truyện được biểu diễn bằng **bốn vector tách riêng** thay vì một vector nối dài:

| Mặt | Nguồn |
|---|---|
| `art` — nét vẽ | 15 thống kê mực và tông rút từ ảnh bìa: độ dày nét, mật độ screentone, tỉ lệ mực, độ bão hoà |
| `pace` — nhịp truyện | Số chương, thời gian serialize, đã hoàn chưa, **tỉ lệ bỏ dở và bỏ ở chương nào** |
| `theme` — chủ đề | Tag cộng đồng có trọng số phần trăm, thể loại |
| `creator` — người sáng tác | Hai không gian riêng: người viết truyện và người vẽ |

Tách mặt vì ba lý do: phần **giải thích** cần biết hai truyện giống nhau *ở chỗ nào*; **trọng số từng mặt học riêng cho mỗi người đọc** (có người chọn theo nét vẽ, có người theo cốt); và truyện thiếu dữ liệu ở một mặt vẫn dùng được các mặt còn lại.

Cả bốn mặt đều đóng góp biên dương ở lát truyện mới — thang ablation 128 → 192 → 205 → 220 chiều cho NDCG@10 0,0055 → 0,0073 → 0,0149 → 0,0165, tức **đủ bốn mặt gấp 3 lần chỉ dùng chủ đề**. Bậc cuối (`art`) nhỏ hơn dao động giữa các seed nên phải kiểm riêng: huấn luyện ghép cặp 5 seed cho chênh +0,00208 ± 0,00113, **5/5 seed dương**, `t = 4,10` / `df = 4` / p ≈ 0,015 — nét vẽ là mặt thật, không phải nhiễu. Chi tiết và cả phép kiểm *sai* từng cho kết luận ngược: [research-notes.md](docs/research-notes.md) mục 10.

Chấm điểm bằng **hai tháp**, chấm toàn bộ 34.608 truyện cho mỗi người thay vì xếp lại top-400 ứng viên: `v_i = e_id[i] + P·c_i` cho truyện, lịch sử gộp tuyến tính cho người đọc, cộng một đầu tuyến tính riêng cho thời điểm phát hành. Bản hai tầng cũ (ALS lấy ứng viên rồi xếp lại) vẫn giữ làm mốc so sánh — chỉ 35,6% truyện trong test lọt vào top-400 nên trần đó chặn mọi cải tiến.

API phục vụ **bốn mô hình, bốn mục đích**, chọn bằng tham số `engine`:

| `engine` | all (temporal) | người đọc mới | người đọc thưa | warm | truyện mới | phủ danh mục | manhua |
|---|---|---|---|---|---|---|---|
| **`auto`** (mặc định) | **0,1202** | **0,3771** | **0,1387** | **0,1051** | 0,0000 | 0,035 | 0,02 |
| `feed` | 0,1169 | **0,3771** | 0,1268 | 0,1039 | 0,0000 | 0,034 | 0,01 |
| `gated` | **0,1202** | 0,3533 | **0,1387** | **0,1051** | 0,0000 | 0,035 | 0,02 |
| `discover` | — | — | — | — | **0,0173** | **0,188** | **0,11** |

`gated` học riêng cho mỗi người một mức thích-truyện-phổ-biến. Phép kiểm **ghép cặp 4 seed** chốt hai điều ngược chiều: nó hơn `feed` **+1,9%** ở lát `all` (`t = +4,27`, 4/4 seed) và kém **−5,2%** ở người đọc mới hoàn toàn (`t = −5,79`, 0/4 seed). Mức +4,4% ở người đọc thưa thì 4/4 seed dương nhưng `t = 2,60` **chưa vượt** ngưỡng 3,18 — ghi là chưa lập được. Vì hai kết quả lập được ngược chiều nhau nên câu "chọn mô hình nào" là câu sai: mặc định là **`auto`**, định tuyến `feed` cho người chưa có lịch sử và `gated` cho người đã có, đo được lấy max ở mọi lát mà không tốn gì về đa dạng nước hay độ trễ. Trên nhánh `taste`, `feed` đạt NDCG@10 **0,2327** (+48% so ALS) và `discover` 0,1229.

Không phải một tốt hai kém — đo được rằng cả ba là **ba điểm trên cùng một núm**: phục vụ bao nhiêu phần của đầu độ phổ biến. Quét chín giá trị trên cùng một mô hình cho một đường đổi **đơn điệu cả hai chiều**: NDCG người-đọc-mới 0,048 → 0,395 trong khi phủ danh mục 0,125 → 0,021 và manhua 0,13 → 0,01. Không có điểm nào được cả hai, nên đây là một lựa chọn sản phẩm phải công bố chứ không phải một mô hình "đúng".

Nhưng *vị trí* của đường cong đó không phải định luật: áp **DropoutNet cho chính cái cổng** (`g_u = base(log(1+n_u)) + delta_u`, bỏ `delta_u` với p=0,3 lúc huấn luyện) đẩy cả frontier ra ngoài — tại cùng mức phủ danh mục, NDCG người-đọc-mới đi từ 0,2944 lên 0,3498, đóng 67% khoảng cách tới `feed`. Và riêng việc bỏ ngẫu nhiên làm **toàn bộ** việc đó: cùng kiến trúc, cùng khởi tạo đến từng bit, chỉ khác có bỏ `delta_u` hay không, `new_user` đi từ 0,1774 lên 0,3533 (**+99%**). Đưa quan hệ vào cấu trúc mà không bỏ ngẫu nhiên thì **tệ hơn** cả bản cũ — điều quyết định là huấn luyện trong đúng điều kiện lúc phục vụ. Chi tiết cơ chế: [docs/research-notes.md](docs/research-notes.md) mục 6 và 9.

**Người đọc mới** không phải chọn thể loại mà chọn vài ảnh bìa từ lưới trải đều các cụm nét vẽ — vài giây là dựng được hồ sơ về mặt nét vẽ, thứ mà "chọn 3 thể loại" không lấy được.

Một chỉ tiêu chốt trước đã bị **chính dữ liệu bác bỏ cả về giá trị lẫn về dấu**: "ít nhất 2 nước trong top-10". Người đọc thật chỉ đạt **1,94** nước trong 10 bộ gần nhất (29,6% chỉ đọc một nước), nên ngưỡng đó đòi hệ đa dạng *hơn* người nó phục vụ. Và qua cả ba engine, thứ tự của nó **ngược** với độ khớp gu: `discover` khớp tỉ lệ nước của từng người sát nhất (khoảng cách 0,147, dưới cả mức sàn 0,182 đo từ hành vi đọc thật) nhưng có **ít** nước nhất (1,61), còn `feed` khớp tệ nhất (0,220) lại có **nhiều** nước nhất (1,76) — vì nó nhồi manhwa vào danh sách của mọi người. Chỉ tiêu đó thưởng cho sự lệch gu, nên đã được thay bằng khoảng cách hiệu chỉnh theo Steck 2018. Chi tiết: [docs/success-criteria.md](docs/success-criteria.md).

## Tài liệu

- [SPEC.md](SPEC.md) — kiến trúc và cấu trúc repo
- [docs/data-sources.md](docs/data-sources.md) — so sánh nguồn dữ liệu, các bẫy xử lý, pipeline
- [docs/goodreads-arm.md](docs/goodreads-arm.md) — **nhánh đối chứng Goodreads**: phương pháp có lặp lại trên bộ dữ liệu khác không; bộ chuyển đổi dùng chung pipeline, bảng kiểm dữ liệu thật, và các dự đoán chốt trước khi chạy mô hình
- [docs/content-features.md](docs/content-features.md) — thiết kế đặc trưng nét vẽ, nhịp truyện, tác giả
- [docs/success-criteria.md](docs/success-criteria.md) — **ngưỡng đạt/không đạt, chốt trước khi có kết quả**, kèm việc phải làm nếu không đạt
- [docs/data-diagnosis.md](docs/data-diagnosis.md) — vì sao ablation phẳng: mổ xẻ bản chất dữ liệu (mục C–G)
- [docs/research-notes.md](docs/research-notes.md) — đối chiếu văn liệu + phân tích sâu: nét vẽ sau khi trừ nhiễu franchise, gu trôi chậm, ALS theo nhóm người đọc, DROPPED (mục 1–5); logQ và `pace` đổi dấu (6–7); **vì sao manhua bị dìm 6,3 lần** (8); cổng độ phổ biến theo người đọc, gồm cả chỗ tôi đọc sai phép đo (9); **mặt `art`: cùng một bộ số, phép kiểm độc lập và phép kiểm ghép cặp cho hai kết luận trái ngược** (10)
