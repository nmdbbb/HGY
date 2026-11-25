"""Ngưỡng "≥ 2 nước trong top-10" có phải là ngưỡng đúng không.

    python scripts/diagnose_country_threshold.py

Bậc 4 đo được 1,76 nước trong top-10 (`feed`), dưới ngưỡng ≥ 2 mà tiêu chí đã
chốt trước khi có kết quả. Fallback đã chốt sẵn là "thêm ràng buộc đa dạng theo
nước". Nhưng §9.7 đo được rằng đa dạng và độ chính xác đổi nhau **đơn điệu**,
nên ràng buộc đó không tạo ra giá trị, nó chỉ dịch điểm làm việc — và §8 đã chỉ
ra nguyên nhân thật nằm ở đầu độ phổ biến, không nằm ở việc thiếu ràng buộc.

Trước khi làm hay bỏ, phải trả lời một câu khác hẳn: **chính người đọc có đạt
ngưỡng đó không?** Nếu 10 bộ liên tiếp mà một người thật đọc cũng chỉ chứa
trung bình ~1,8 nước, thì ngưỡng ≥ 2 đang đòi mô hình đa dạng *hơn* chính người
nó phục vụ, và cái sai là ngưỡng chứ không phải mô hình. Đây là lập luận hiệu
chỉnh của Steck 2018 áp cho ngưỡng: mốc đúng là hành vi của người đọc, không
phải một con số tròn.

Phép đo không cần mô hình, không nạp trọng số: chỉ lịch sử thật + nước của từng
truyện. Ba cách đếm, vì "số nước trong 10 bộ" phụ thuộc cách chọn 10 bộ nào:

  1. 10 bộ **gần nhất** của mỗi người — sát nhất với một lần mở app.
  2. 10 bộ **bất kỳ** (ngẫu nhiên không lặp) — bỏ yếu tố thời điểm.
  3. **Toàn bộ** lịch sử — chặn trên: ai cả đời chỉ đọc một nước thì không có
     cách chọn 10 bộ nào cho ra 2 nước.

Quy tắc đọc, chốt trước:

  - Người đọc ở cách 1 trung bình < 2,0 → ngưỡng ≥ 2 **mis-specified**, ghi lại
    và thay bằng mốc hiệu chỉnh (khoảng cách tới phân phối của chính người đó).
  - Người đọc ≥ 2,0 rõ ràng mà mô hình 1,76 → thiếu hụt là thật, fallback giữ.
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

COUNTRIES = ("JP", "KR", "CN")


def summarise(label, counts, total_users):
    counts = np.asarray(counts, dtype=float)
    share_ge2 = float((counts >= 2).mean())
    print(f"  {label:<34} {counts.mean():.2f} nước  "
          f"| ≥2 nước: {share_ge2:5.1%}  | n = {total_users}")
    return {"mean": float(counts.mean()), "share_ge2": share_ge2,
            "hist": {str(k): int(v) for k, v in
                     zip(*np.unique(counts.astype(int), return_counts=True))}}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    media = pd.read_parquet(args.out_dir / "media_core.parquet",
                            columns=["media_id", "country"])
    country_of = dict(zip(media["media_id"], media["country"]))

    frame = pd.read_parquet(args.out_dir / "interactions.parquet")
    frame = frame[frame["media_id"].isin(country_of)].copy()
    frame["country"] = frame["media_id"].map(country_of)
    frame = frame[frame["country"].isin(COUNTRIES)]

    stamp = next((c for c in ("updated_at", "completed_at", "started_at")
                  if c in frame.columns), None)
    print(f"{len(frame):,} tương tác có nước | cột thời điểm: {stamp}")

    rng = np.random.default_rng(args.seed)
    recent, anyk, lifetime, keys = [], [], [], []
    for key, group in frame.groupby("user", sort=False):
        if len(group) < args.k:
            continue
        keys.append(key)
        lifetime.append(group["country"].nunique())
        ordered = group.sort_values(stamp) if stamp else group
        recent.append(ordered["country"].tail(args.k).nunique())
        pick = rng.choice(len(group), size=args.k, replace=False)
        anyk.append(group["country"].iloc[pick].nunique())

    print(f"\nSố nước khác nhau trong {args.k} bộ — ĐO TRÊN NGƯỜI ĐỌC THẬT")
    print("-" * 78)
    result = {
        "k": args.k, "users": len(recent),
        f"recent_{args.k}": summarise(f"{args.k} bộ gần nhất", recent, len(recent)),
        f"any_{args.k}": summarise(f"{args.k} bộ bất kỳ", anyk, len(anyk)),
        "lifetime": summarise("toàn bộ lịch sử (chặn trên)", lifetime, len(lifetime)),
    }

    # Mô hình đo được ở bậc 4, để so cùng đơn vị. Mốc so đúng là cột "10 bộ gần
    # nhất": nó là cái một người thật *vừa làm*, tức mức đa dạng mà hành vi của
    # chính họ bảo chứng, chứ không phải một con số tròn chọn trước.
    benchmark = result[f"recent_{args.k}"]["mean"]
    product = args.out_dir / "product_metrics.json"
    if product.exists():
        blob = json.loads(product.read_text(encoding="utf-8"))
        rows = blob.get("engines", [])
        print(f"\nSo với mốc hành vi người đọc ({benchmark:.2f} nước):")
        print("-" * 78)
        print(f"  {'engine':<12}{'nước/top-10':>13}{'so mốc':>10}"
              f"{'so ngưỡng ≥2':>15}")
        for row in rows:
            got = row.get("countries")
            if got is None:
                continue
            print(f"  {row.get('engine', '?'):<12}{got:>13.2f}"
                  f"{got / benchmark - 1:>+9.1%}{got / 2.0 - 1:>+14.1%}")
        result["benchmark_recent"] = benchmark
        result["engines"] = {r.get("engine"): r.get("countries") for r in rows}

    # `discover` phủ danh mục gấp 5 lần và có manhua gấp 11 lần `feed`, nhưng
    # lại ÍT nước hơn (1,61 so 1,76). Nghe như nghịch lý, nên phải đo xem nó có
    # nằm sẵn trong dữ liệu hay là lỗi mô hình: nếu người đọc càng đi vào đuôi
    # dài thì gu càng bó vào một nước, thì 1,61 là tính chất của vùng đuôi dài
    # chứ không phải của `discover`.
    rank = frame["media_id"].value_counts().rank(pct=True)   # 1,0 = phổ biến nhất
    frame["pop_pct"] = frame["media_id"].map(rank)
    pop = frame.groupby("user")["pop_pct"].mean()
    per_user = pd.DataFrame({"pop": pop.reindex(keys).to_numpy(),
                             "countries": np.asarray(recent, dtype=float)},
                            index=pd.Index(keys, name="user"))
    rho = float(np.corrcoef(per_user["pop"].rank(),
                            per_user["countries"].rank())[0, 1])
    tertile = pd.qcut(per_user["pop"], 3, labels=["đuôi dài", "giữa", "phổ biến"])
    print(f"\nĐuôi dài có bó vào một nước không? (tương quan hạng ρ = {rho:+.3f})")
    print("-" * 78)
    for name, sub in per_user.groupby(tertile, observed=True):
        print(f"  {str(name):<12} độ nổi TB {sub['pop'].mean():.3f}"
              f" → {sub['countries'].mean():.2f} nước  (n = {len(sub):,})")
    result["tail_vs_country_rho"] = rho

    # "10 bộ gần nhất" ÍT nước hơn "10 bộ bất kỳ" trên cùng những người đó —
    # nếu chênh này thật thì người đọc đi theo **đợt theo nước**, và lúc đó một
    # top-10 trộn đều là sai với người đang trong đợt manhwa. Kiểm ghép cặp,
    # vì hai cột đo trên đúng cùng một tập người.
    pair = np.asarray(recent, dtype=float) - np.asarray(anyk, dtype=float)
    n_pos, n_neg = int((pair > 0).sum()), int((pair < 0).sum())
    mean = pair.mean()
    se = pair.std(ddof=1) / np.sqrt(len(pair))
    print(f"\nĐợt theo nước: gần nhất − bất kỳ = {mean:+.3f} ± {se:.3f}"
          f"  (t = {mean / se:+.1f}, n = {len(pair):,})")
    print(f"  ít nước hơn: {n_neg:,} người | nhiều hơn: {n_pos:,} người"
          f" | như nhau: {len(pair) - n_pos - n_neg:,}")
    result["phase_effect"] = {"mean": float(mean), "se": float(se),
                              "t": float(mean / se), "n": len(pair)}

    out = args.out_dir / "country_threshold.json"
    out.write_text(json.dumps(result, ensure_ascii=False, indent=2),
                   encoding="utf-8")
    print(f"\n-> {out}")


if __name__ == "__main__":
    main()
