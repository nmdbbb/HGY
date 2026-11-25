"""Đo bậc 4 của `docs/success-criteria.md`: dùng được như sản phẩm chưa.

    python scripts/measure_product.py
    python scripts/measure_product.py --engines feed discover hybrid --users 200

Bậc 4 bị để trắng suốt quá trình vì tôi đo mô hình chứ không đo sản phẩm. Một
hệ đúng về số mà chậm, hoặc gợi 10 bộ cùng một hoạ sĩ, hoặc không nói được vì
sao, thì vẫn là thất bại — nên bốn ngưỡng dưới đây chốt từ trước và đo qua đúng
`RecommenderService` mà API dùng, không phải qua một đường tắt trong script.

Thời gian đo là thời gian *trong tiến trình*, không qua HTTP: nó đo mô hình và
việc dựng lý do, không đo mạng. Ngưỡng 300 ms vì vậy là ngưỡng dễ hơn thực tế,
phải nói rõ như thế.
"""

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.stdout.reconfigure(encoding="utf-8")

from truyen_reco.api.service import RecommenderService  # noqa: E402

THRESHOLDS = {
    "creator_conc": ("trùng người sáng tác trong top-10", 0.30, "<="),
    "countries": ("số nước trong top-10", 2.0, ">="),
    "reason_rate": ("tỉ lệ gợi ý có lý do", 1.0, ">="),
    "p95_ms": ("thời gian mỗi lần gợi ý (p95)", 300.0, "<="),
    "onboard_picks": ("số bìa người mới phải chọn", 5.0, "<="),
}

# Chỉ tiêu nào vô nghĩa khi danh mục không có trường xuất xứ. Trên bộ Goodreads
# `country` rỗng toàn bộ, nên "số nước trong top-10" sẽ luôn bằng 0 và bảng sẽ
# báo KHÔNG ĐẠT — một kết luận sai về mô hình khi nguyên nhân nằm ở dữ liệu.
# Thiếu dữ liệu phải hiện ra là N/A, không phải là trượt.
NEEDS_COUNTRY = {"countries"}


def passed(name, value):
    """None = không đo được (trượt). 'N/A' = chỉ tiêu không áp dụng cho bộ này."""
    _, threshold, direction = THRESHOLDS[name]
    if value == "N/A":
        return "N/A"
    if value is None:
        return False
    return value <= threshold if direction == "<=" else value >= threshold


def measure_engine(service, engine, users, seed=0):
    rng = np.random.default_rng(seed)
    pool = [u for u in service.history if len(service.history[u]) >= 5]
    picked_users = rng.choice(pool, size=min(users, len(pool)), replace=False)

    timings, creator_rates, country_counts, reason_hits, total = [], [], [], 0, 0
    # Ngưỡng "≥2 nước trong top-10" chốt từ trước, nhưng nó có thể sai kiểu như
    # ngưỡng bậc 0: người chỉ đọc manga mà nhận toàn manga là *đúng*, không
    # phải thiếu đa dạng. Nên đo thêm phần có nghĩa: so số nước trong top-10
    # với số nước trong chính lịch sử người đó. Thiếu đa dạng thật là khi người
    # đọc nhiều nước mà chỉ nhận về một nước.
    matched, shortfalls = [], []
    for user in picked_users:
        started = time.perf_counter()
        result = service.recommend(user_id=str(user), top_k=10, engine=engine)
        timings.append((time.perf_counter() - started) * 1000.0)

        items = result["items"]
        if not items:
            continue
        total += len(items)
        reason_hits += sum(1 for card in items if card.get("reason"))

        # Trùng người sáng tác: tỉ lệ vị trí bị chiếm bởi một người đã xuất hiện
        groups = [service.model.creator_groups.get(service.item_index[c["media_id"]])
                  for c in items]
        seen, repeats = set(), 0
        for key in groups:
            if key is None:
                continue
            if key in seen:
                repeats += 1
            seen.add(key)
        creator_rates.append(repeats / len(items))
        got_countries = {c.get("country") for c in items if c.get("country")}
        country_counts.append(len(got_countries))

        read = {service.countries.get(service.reverse_item[p])
                for p in service.history[user]}
        read = {c for c in read if c}
        if read:
            wanted = min(len(read), 3)
            matched.append(len(got_countries & read) / len(read))
            shortfalls.append(max(0, min(wanted, 3) - len(got_countries)))

    return {
        "engine": result.get("engine", engine),
        "requested": engine,
        "users": len(picked_users),
        "median_ms": statistics.median(timings),
        "p95_ms": float(np.percentile(timings, 95)),
        "creator_conc": float(np.mean(creator_rates)),
        "countries": float(np.mean(country_counts)),
        "country_recall": float(np.mean(matched)) if matched else None,
        "country_shortfall": float(np.mean(shortfalls)) if shortfalls else None,
        "reason_rate": reason_hits / max(total, 1),
    }


def measure_onboarding(service, engine, trials=40, seed=0):
    """Ít nhất bao nhiêu bìa thì hồ sơ người mới mới thực sự khác mặc định.

    Phép đo: với k bìa chọn ngẫu nhiên từ lưới onboarding, top-10 lệch bao
    nhiêu so với top-10 khi chưa chọn gì. Nếu không lệch thì màn onboarding chỉ
    là trang trí — người dùng bấm xong vẫn nhận đúng danh sách phổ biến.
    """
    rng = np.random.default_rng(seed)
    grid = [card["media_id"] for card in service.onboarding(n_options=24)]
    baseline = None
    if engine in service.towers:
        tower = service.towers[engine]
        baseline = {service.reverse_item[int(p)]
                    for p in tower.top_k(None, [], k=10)}

    rows = {}
    for k in range(1, 6):
        shifts, usable = [], 0
        for _ in range(trials):
            picks = list(rng.choice(grid, size=k, replace=False))
            result = service.recommend(picked_ids=[int(m) for m in picks],
                                       top_k=10, engine=engine)
            got = {c["media_id"] for c in result["items"]}
            if baseline is not None:
                shifts.append(1.0 - len(got & baseline) / max(len(got), 1))
            positions = [service.item_index[m] for m in picks
                         if m in service.item_index]
            profile = service.facets.profile(positions)
            usable += sum(1 for v in profile.values() if v is not None) >= 3
        rows[k] = {"shift": float(np.mean(shifts)) if shifts else None,
                   "facets_ok": usable / trials}
    return rows


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", default="data/processed", type=Path)
    parser.add_argument("--engines", nargs="+", default=["feed", "discover", "hybrid"])
    parser.add_argument("--users", type=int, default=150)
    args = parser.parse_args()

    started = time.monotonic()
    service = RecommenderService(args.out_dir)
    load_seconds = time.monotonic() - started
    print(f"nạp service {load_seconds:.1f}s | "
          f"hai tháp có sẵn: {sorted(service.towers) or 'không có'}")

    rows = [measure_engine(service, engine, args.users) for engine in args.engines]

    print("\n" + "=" * 92)
    print("BẬC 4 — dùng được như sản phẩm chưa")
    print("=" * 92)
    head = (f"{'engine':<12}{'median ms':>11}{'p95 ms':>9}"
            f"{'trùng tác giả':>15}{'số nước':>9}{'phủ nước đã đọc':>17}"
            f"{'có lý do':>10}")
    print(head)
    print("-" * len(head))
    for row in rows:
        mark = "" if row["engine"] == row["requested"] else "  (rơi về hybrid)"
        print(f"{row['engine']:<12}{row['median_ms']:>11.1f}{row['p95_ms']:>9.1f}"
              f"{row['creator_conc']:>15.3f}{row['countries']:>9.2f}"
              f"{row['country_recall'] or 0:>17.3f}"
              f"{row['reason_rate']:>10.3f}{mark}")
    print("\n'phủ nước đã đọc' = tỉ lệ các nước trong lịch sử người đó có mặt "
          "trong top-10.\nSố này mới nói được thiếu đa dạng hay không; 'số "
          "nước' trung bình bị kéo xuống\nbởi những người thật sự chỉ đọc một "
          "loại truyện.")

    onboard = {}
    for engine in args.engines:
        onboard[engine] = measure_onboarding(service, engine)
    print("\n" + "=" * 92)
    print("ONBOARDING — top-10 lệch bao nhiêu so với khi chưa chọn gì")
    print("=" * 92)
    print(f"{'engine':<12}" + "".join(f"{f'{k} bìa':>12}" for k in range(1, 6)))
    print("-" * 72)
    for engine, rows_k in onboard.items():
        line = f"{engine:<12}"
        for k in range(1, 6):
            shift = rows_k[k]["shift"]
            line += f"{'—' if shift is None else f'{shift:.2f}':>12}"
        print(line)

    print("\n" + "=" * 92)
    print("ĐỐI CHIẾU NGƯỠNG (engine mặc định: engines[0])")
    print("=" * 92)
    best = rows[0]
    first_shift = next((k for k in range(1, 6)
                        if (onboard[args.engines[0]][k]["shift"] or 0) >= 0.5), None)
    checks = dict(best)
    checks["onboard_picks"] = first_shift
    has_country = any(c is not None for c in service.countries.values())
    if not has_country:
        print("  [danh mục này KHÔNG có trường xuất xứ: chỉ tiêu theo nước là N/A]")
        for name in NEEDS_COUNTRY:
            checks[name] = "N/A"
    for name, (label, threshold, direction) in THRESHOLDS.items():
        value = checks.get(name)
        state = passed(name, value)
        state = "N/A" if state == "N/A" else ("đạt" if state else "KHÔNG ĐẠT")
        shown = ("không đo được" if value is None
                 else value if isinstance(value, str) else f"{value:.3f}")
        print(f"  {label:<38} {direction} {threshold:<6} | {shown:>14}  {state}")

    report = {"engines": rows, "onboarding": onboard,
              "load_seconds": load_seconds}
    path = args.out_dir / "product_metrics.json"
    path.write_text(json.dumps(report, ensure_ascii=False, indent=1),
                    encoding="utf-8")
    print(f"\n-> {path}   ({(time.monotonic() - started) / 60:.1f} phút)")


if __name__ == "__main__":
    main()
