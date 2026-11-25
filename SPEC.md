# SPEC — Khung repo: Hệ gợi ý Truyện tranh (IT4613)

Nguồn: báo cáo đề xuất đồ án "Hệ gợi ý Truyện tranh — Báo cáo đề xuất đồ án IT4613" (Claude Docs, artifact `QiHky3odpY2z2V1xC3qpum`).

## 1. Bài toán (trích từ báo cáo đề xuất)

> Nhóm xây dựng một hệ gợi ý truyện tranh lai ghép (hybrid), giúp người đọc tìm được truyện hợp gu trong kho hàng chục nghìn đầu truyện, có giải thích lý do gợi ý và xử lý được người dùng mới.

Ba yêu cầu lõi rút ra trực tiếp từ câu trên, chi phối toàn bộ kiến trúc dưới đây:

1. **Lai ghép (hybrid)** — kết hợp ít nhất 2 nguồn tín hiệu: nội dung truyện (content-based) và hành vi người dùng (collaborative filtering).
2. **Giải thích được (explainable)** — mỗi gợi ý phải đi kèm lý do (ví dụ: "vì bạn đã đọc X", "cùng thể loại Y", "độc giả giống bạn cũng đọc").
3. **Cold-start** — người dùng mới (chưa có lịch sử đọc) và có thể cả truyện mới (chưa có lượt đọc/đánh giá) vẫn phải nhận được gợi ý hợp lý.

Thiết kế chi tiết cho phần content-based (nét vẽ, nhịp truyện, tác giả/họa sĩ): xem [docs/content-features.md](docs/content-features.md).

Nguồn dữ liệu — **AniList là nguồn chính, Goodreads Comics & Graphic là nhánh đối chứng** — các bẫy xử lý và pipeline: xem [docs/data-sources.md](docs/data-sources.md). Nhánh đối chứng (bộ chuyển đổi, bảng kiểm dữ liệu thật, các dự đoán chốt trước): xem [docs/goodreads-arm.md](docs/goodreads-arm.md).

> **Lưu ý:** báo cáo nguồn hiện mới có phần "1. Bài toán thực tế"; các phần mục tiêu, dữ liệu, phương pháp, tiêu chí đánh giá, kế hoạch chưa được viết. Spec này suy ra kiến trúc và khung thư mục từ bài toán đã nêu — các giả định được đánh dấu rõ ở mục 6 và cần nhóm xác nhận/sửa khi báo cáo được bổ sung.

## 2. Mục tiêu / phi mục tiêu

**Trong phạm vi:**
- Pipeline dữ liệu: thu thập/nhập metadata truyện + tương tác người dùng (đọc, rating, bookmark...).
- 3 thành phần model: content-based, collaborative filtering, hybrid kết hợp hai nguồn trên theo trọng số/stacking.
- Module cold-start riêng (dựa trên popularity + thể loại người dùng chọn lúc onboarding).
- Module giải thích (explain) gắn với từng gợi ý được trả về.
- API serving để demo (và để frontend/notebook gọi vào).
- Bộ đánh giá offline (precision@k, recall@k, NDCG, coverage, và đánh giá riêng cho cold-start).

**Ngoài phạm vi (giả định ban đầu, có thể sai nếu báo cáo nói khác):**
- Không tự thu thập dữ liệu qua scraping web truyện có bản quyền — dùng dataset có sẵn hoặc dữ liệu giả lập/được cấp.
- Không build app di động; chỉ cần 1 giao diện demo tối thiểu (web đơn giản hoặc notebook).
- Không triển khai real-time retraining; retrain theo batch/offline.

## 3. Kiến trúc tổng thể

```
┌─────────────┐     ┌──────────────┐     ┌────────────────────┐
│  Dữ liệu thô │ ──▶ │  Preprocess/  │ ──▶ │     Features        │
│ (truyện,     │     │  Feature      │     │ (embedding nội dung, │
│  tương tác)  │     │  engineering  │     │  ma trận user-item)  │
└─────────────┘     └──────────────┘     └─────────┬──────────┘
                                                      │
                     ┌────────────────────────────────┼────────────────────┐
                     ▼                                 ▼                    ▼
            ┌─────────────────┐              ┌──────────────────┐  ┌───────────────┐
            │ Content-based    │              │ Collaborative     │  │  Cold-start    │
            │ (similarity theo │              │ Filtering         │  │  (popularity + │
            │ thể loại/mô tả)  │              │ (matrix factor./  │  │  thể loại chọn │
            │                  │              │  item-item CF)    │  │  lúc onboard)  │
            └────────┬─────────┘              └─────────┬─────────┘  └───────┬───────┘
                     └──────────────────┬─────────────────┴────────────────────┘
                                         ▼
                                ┌────────────────────┐
                                │   Hybrid combiner    │
                                │ (weighted/ stacking)  │
                                └─────────┬───────────┘
                                          ▼
                                ┌────────────────────┐
                                │   Explain layer      │
                                │ (lý do cho mỗi item)  │
                                └─────────┬───────────┘
                                          ▼
                                ┌────────────────────┐
                                │   API (FastAPI)       │
                                └─────────┬───────────┘
                                          ▼
                                ┌────────────────────┐
                                │  Frontend demo tối    │
                                │  thiểu                │
                                └────────────────────┘
```

Offline: `scripts/train.py` chạy pipeline từ dữ liệu → features → train từng model → lưu artifact (`models/*.pkl` hoặc tương đương) → `scripts/evaluate.py` chấm điểm trên tập test.

Online: `api/main.py` load các artifact đã train, nhận `user_id` (+ optional context lúc cold-start: thể loại đã chọn), chạy qua hybrid combiner + explain layer, trả về danh sách gợi ý kèm lý do.

## 4. Ngăn xếp công nghệ đề xuất

| Thành phần | Lựa chọn | Lý do |
|---|---|---|
| Ngôn ngữ chính | Python 3.11+ | Hệ sinh thái ML/RecSys mạnh nhất, phù hợp đồ án |
| Xử lý dữ liệu | pandas, numpy | Chuẩn, đủ cho quy mô hàng chục nghìn item |
| Content-based | scikit-learn (TF-IDF/cosine) hoặc sentence-transformers nếu cần embedding ngữ nghĩa | TF-IDF đủ nhanh để làm baseline; embedding nếu cần chất lượng cao hơn |
| Collaborative filtering | `implicit` (ALS) hoặc `lightfm`/`surprise` | Có sẵn, nhẹ, phù hợp đồ án môn học |
| Serving | FastAPI + uvicorn | Nhanh để viết, tự sinh docs API (`/docs`), dễ demo |
| Demo UI | Streamlit (ưu tiên, ít code) hoặc React tối giản nếu cần tùy biến | Streamlit đủ cho mục đích báo cáo/demo |
| Thực nghiệm | Jupyter notebook (`notebooks/`) | EDA và so sánh model trước khi đóng gói vào `src/` |
| Quản lý phụ thuộc | `pyproject.toml` (hoặc `requirements.txt` nếu nhóm quen pip thuần) | Chọn 1 trong 2, tránh trùng lặp |
| Test | pytest | Chuẩn cho Python |
| CI | GitHub Actions | Chạy lint + test mỗi push/PR |

> Đây là lựa chọn mặc định hợp lý cho một đồ án học phần — nhóm có thể đổi (ví dụ dùng PyTorch nếu cần deep learning cho content embedding) miễn giữ đúng ranh giới module trong mục 5.

## 5. Cấu trúc thư mục

```
truyen/
├── README.md                  # Giới thiệu, hướng dẫn chạy nhanh
├── SPEC.md                    # File này
├── pyproject.toml             # Deps + cấu hình project
├── .gitignore
├── .env.example                # Biến môi trường mẫu (đường dẫn data, API key nếu có)
├── configs/
│   └── config.yaml            # Tham số pipeline: đường dẫn data, trọng số hybrid, k...
├── data/                       # KHÔNG commit dữ liệu thật (trừ sample nhỏ)
│   ├── raw/                    # Dữ liệu gốc, chưa xử lý
│   ├── interim/                # Dữ liệu trung gian
│   ├── processed/              # Dữ liệu đã sẵn sàng để train
│   └── external/               # Dữ liệu/tài nguyên bên ngoài (ví dụ embedding pretrained)
├── notebooks/                  # EDA, thử nghiệm, so sánh model (không chứa logic production)
├── src/truyen_reco/            # Package chính — mọi logic production nằm ở đây
│   ├── data/                   # Nạp & làm sạch dữ liệu
│   ├── features/               # Sinh đặc trưng (TF-IDF, ma trận user-item...)
│   ├── models/
│   │   ├── content_based.py
│   │   ├── collaborative.py
│   │   ├── hybrid.py
│   │   └── cold_start.py
│   ├── explain/                # Sinh lý do cho từng gợi ý
│   ├── evaluation/             # Metric: precision@k, recall@k, NDCG, coverage
│   ├── api/
│   │   ├── main.py             # FastAPI app
│   │   ├── routers/recommend.py
│   │   └── schemas.py          # Pydantic request/response models
│   └── utils/                  # Logging, helper chung
├── frontend/                   # Demo UI tối giản (Streamlit hoặc web nhẹ)
├── scripts/
│   ├── train.py                # Entry point train toàn bộ pipeline
│   └── evaluate.py             # Entry point đánh giá offline
├── tests/                      # Unit test cho src/truyen_reco
├── docs/
│   ├── architecture.md         # Chi tiết kiến trúc (mở rộng từ SPEC.md)
│   ├── data_dictionary.md      # Định nghĩa schema dữ liệu
│   └── report/                 # Bản thảo báo cáo đồ án (slide, docx/pdf export...)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
└── .github/workflows/ci.yml    # Lint + test tự động
```

Ranh giới quan trọng cần giữ:
- `notebooks/` chỉ để khám phá/thử nghiệm — logic chạy được (đã kiểm chứng) phải chuyển vào `src/truyen_reco/`.
- `models/` trong `src/` chứa **code định nghĩa model**, không chứa **artifact đã train** (file `.pkl`/`.bin` nên lưu ngoài git, ví dụ `data/processed/models/` hoặc git-lfs/external storage, tuỳ nhóm quyết định).
- `cold_start.py` là module riêng, không lồng vào trong `hybrid.py`, vì đây là yêu cầu lõi của bài toán và cần test độc lập (test với user chưa có lịch sử).
- `explain/` tách khỏi `models/` vì lý do gợi ý phải map được với *từng loại model* (content/CF/cold-start) sinh ra gợi ý đó.

## 6. Giả định cần nhóm xác nhận

Vì báo cáo nguồn hiện chỉ có phần "Bài toán thực tế", các điểm sau là giả định hợp lý rút ra từ câu đề bài, không phải trích dẫn từ tài liệu — cần nhóm chốt lại khi báo cáo đầy đủ hơn:

- ~~Nguồn dữ liệu cụ thể~~ — **đã chốt: AniList tự thu thập (chính), Goodreads Comics & Graphic (đối chứng).** Xem [docs/data-sources.md](docs/data-sources.md).
- ~~Tín hiệu đo "hợp gu"~~ — **đã rõ từ dữ liệu**: có cả hai. Điểm tường minh (`score`, ~34% số mục) và implicit 5 mức (CURRENT / COMPLETED / DROPPED / PAUSED / PLANNING). Lưu ý PLANNING là "muốn đọc", không phải đã đọc.
- Hình thức giải thích (text tự nhiên hay nhãn có cấu trúc như tag thể loại) — ảnh hưởng `explain/`.
- Có cần serving thời gian thực hay chỉ cần batch + demo tĩnh cho báo cáo.

## 7. Quy ước làm việc

- Nhánh: `main` ổn định, nhánh tính năng `feat/<ten>`, nhánh fix `fix/<ten>`.
- Commit: ngắn gọn, mô tả *tại sao* khi không hiển nhiên.
- Cấu hình (đường dẫn, trọng số hybrid, k cho top-k) đọc từ `configs/config.yaml`, không hard-code trong `src/`.
- Dữ liệu thật không commit vào git (`.gitignore` đã chặn `data/raw`, `data/interim`, `data/processed`); chỉ giữ `.gitkeep` để giữ cấu trúc thư mục.

## 8. Bước tiếp theo

1. Nhóm bổ sung phần mục tiêu/dữ liệu/phương pháp vào báo cáo gốc trên Claude Docs.
2. Xác nhận/sửa các giả định ở mục 6.
3. Lấp phần `src/truyen_reco/data/ingest.py` theo nguồn dữ liệu thật đã chốt.
4. Làm baseline content-based trước (đơn giản nhất, không cần dữ liệu tương tác) rồi mới sang CF và hybrid.
