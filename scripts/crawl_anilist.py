"""Thu thập dữ liệu manga/manhwa/manhua từ AniList GraphQL API.

Bốn giai đoạn, chạy tuần tự. Mỗi giai đoạn resume được: ngắt giữa chừng rồi
chạy lại sẽ tiếp tục từ chỗ dừng, không mất việc đã làm.

    python scripts/crawl_anilist.py discover
    python scripts/crawl_anilist.py users --target 10000
    python scripts/crawl_anilist.py lists
    python scripts/crawl_anilist.py media

Giới hạn API là 30 request/phút (đã kiểm qua header X-RateLimit-Limit), nên
giai đoạn `lists` với 10.000 user mất khoảng 9 tiếng. Chạy qua đêm.

Username được băm trước khi lưu; file muối nằm ngoài git.
"""

import argparse
import hashlib
import json
import secrets
import sys
import time
import urllib.error
import urllib.request
from collections import Counter
from pathlib import Path

sys.stdout.reconfigure(encoding="utf-8")

API = "https://graphql.anilist.co"
MIN_INTERVAL = 2.1  # 30 request/phút -> 2.0s; thêm biên an toàn
USER_AGENT = "truyen-reco-academic/0.1 (IT4613 course project)"
COUNTRIES = ("KR", "CN", "JP")


class Client:
    """Gọi AniList, tự giữ nhịp và tự lùi khi bị chặn."""

    def __init__(self):
        self.last_call = 0.0
        self.calls = 0

    def query(self, query, variables=None, retries=6):
        payload = json.dumps({"query": query, "variables": variables or {}}).encode()
        for attempt in range(retries):
            gap = time.monotonic() - self.last_call
            if gap < MIN_INTERVAL:
                time.sleep(MIN_INTERVAL - gap)
            self.last_call = time.monotonic()
            request = urllib.request.Request(
                API, data=payload,
                headers={"Content-Type": "application/json",
                         "Accept": "application/json",
                         "User-Agent": USER_AGENT},
            )
            try:
                with urllib.request.urlopen(request, timeout=30) as response:
                    self.calls += 1
                    return json.loads(response.read())
            except urllib.error.HTTPError as err:
                if err.code == 429:
                    wait = int(err.headers.get("Retry-After", 60)) + 1
                    print(f"    [429] chờ {wait}s", flush=True)
                    time.sleep(wait)
                elif err.code >= 500:
                    time.sleep(5 * (attempt + 1))
                else:
                    body = err.read()[:300].decode("utf-8", "replace")
                    print(f"    [HTTP {err.code}] {body}", flush=True)
                    return None
            except (urllib.error.URLError, TimeoutError, OSError) as err:
                print(f"    [mạng] {err}, thử lại", flush=True)
                time.sleep(5 * (attempt + 1))
        return None


def out_dir(base):
    path = Path(base) / "anilist"
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_json(path, default):
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return default


def save_json(path, data):
    path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")


def get_salt(path):
    """Muối cố định để băm username nhất quán giữa các lần chạy."""
    if not path.exists():
        path.write_text(secrets.token_hex(16), encoding="utf-8")
        print(f"Đã tạo muối mới tại {path} (KHÔNG commit file này)")
    return path.read_text(encoding="utf-8").strip()


def anonymize(username, salt):
    return hashlib.sha256((salt + username).encode("utf-8")).hexdigest()[:16]


def done_keys(path, field):
    """Đọc lại các bản ghi đã ghi để resume."""
    seen = set()
    if path.exists():
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        seen.add(json.loads(line)[field])
                    except (json.JSONDecodeError, KeyError):
                        continue
    return seen


# --------------------------------------------------------------------------
# Giai đoạn 1: tìm truyện phổ biến theo từng nước
# --------------------------------------------------------------------------

MEDIA_IDS_QUERY = """
query ($country: CountryCode, $page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    media(type: MANGA, countryOfOrigin: $country, sort: POPULARITY_DESC) { id }
  }
}
"""


def stage_discover(client, data_dir, per_country):
    path = data_dir / "media_ids.json"
    found = load_json(path, {})

    for country in COUNTRIES:
        if len(found.get(country, [])) >= per_country:
            print(f"[{country}] đã có {len(found[country])} id, bỏ qua")
            continue
        ids, page = [], 1
        while len(ids) < per_country:
            result = client.query(MEDIA_IDS_QUERY, {"country": country, "page": page})
            if not result or "data" not in result:
                break
            block = result["data"]["Page"]
            ids.extend(m["id"] for m in block["media"])
            print(f"[{country}] trang {page}: tổng {len(ids)} id", flush=True)
            if not block["pageInfo"]["hasNextPage"]:
                break
            page += 1
        found[country] = ids[:per_country]
        save_json(path, found)

    print("\nKết quả:", {k: len(v) for k, v in found.items()})
    print(f"-> {path}")


# --------------------------------------------------------------------------
# Giai đoạn 2: thu username từ activity trên chính các truyện đó
# --------------------------------------------------------------------------

ACTIVITY_QUERY = """
query ($ids: [Int], $page: Int) {
  Page(page: $page, perPage: 50) {
    pageInfo { hasNextPage }
    activities(mediaId_in: $ids, type: MANGA_LIST, sort: ID_DESC) {
      ... on ListActivity { user { name } media { countryOfOrigin } }
    }
  }
}
"""


def stage_users(client, data_dir, target, weights, pages_per_chunk=3):
    media_ids = load_json(data_dir / "media_ids.json", {})
    if not media_ids:
        raise SystemExit("Chưa có media_ids.json — chạy `discover` trước.")

    path = data_dir / "usernames.json"
    users = load_json(path, {})

    for country in COUNTRIES:
        per_country = int(target * weights[country])
        have = sum(1 for v in users.values() if v == country)
        if have >= per_country:
            print(f"[{country}] đã đủ {have}/{per_country} user")
            continue

        ids = media_ids.get(country, [])
        # Chia nhỏ danh sách truyện để mỗi truy vấn phủ nhiều truyện khác nhau
        chunks = [ids[i:i + 25] for i in range(0, len(ids), 25)]
        chunk_index = 0

        while have < per_country and chunk_index < len(chunks):
            chunk = chunks[chunk_index]
            for page in range(1, pages_per_chunk + 1):  # đo được: 3 trang bỏ sót
                result = client.query(ACTIVITY_QUERY, {"ids": chunk, "page": page})
                if not result or "data" not in result:
                    break
                block = result["data"]["Page"]
                for activity in block["activities"]:
                    if not activity or not activity.get("user"):
                        continue
                    name = activity["user"]["name"]
                    if name not in users:
                        users[name] = country
                        have += 1
                if have >= per_country or not block["pageInfo"]["hasNextPage"]:
                    break
            chunk_index += 1
            save_json(path, users)
            print(f"[{country}] {have}/{per_country} user "
                  f"(tổng {len(users)})", flush=True)

    print("\nPhân bố nguồn user:", dict(Counter(users.values())))
    print(f"Tổng: {len(users)} -> {path}")


# --------------------------------------------------------------------------
# Giai đoạn 3: lấy toàn bộ danh sách đọc của từng user
# --------------------------------------------------------------------------

LIST_QUERY = """
query ($userName: String) {
  MediaListCollection(userName: $userName, type: MANGA) {
    lists {
      entries {
        mediaId
        status
        score(format: POINT_100)
        progress
        progressVolumes
        repeat
        startedAt { year month day }
        completedAt { year month day }
        createdAt
        updatedAt
      }
    }
  }
}
"""


def stage_lists(client, data_dir, salt):
    users = load_json(data_dir / "usernames.json", {})
    if not users:
        raise SystemExit("Chưa có usernames.json — chạy `users` trước.")

    path = data_dir / "lists.jsonl"
    already = done_keys(path, "user")
    todo = [u for u in users if anonymize(u, salt) not in already]

    print(f"Tổng {len(users)} user | đã xong {len(already)} | còn {len(todo)}")
    print(f"Ước tính {len(todo) * MIN_INTERVAL / 3600:.1f} giờ\n")

    entries_total = 0
    with path.open("a", encoding="utf-8") as out:
        for i, username in enumerate(todo, 1):
            result = client.query(LIST_QUERY, {"userName": username})
            if not result or not result.get("data", {}).get("MediaListCollection"):
                # List riêng tư hoặc user đã xoá: ghi nhận để không thử lại
                out.write(json.dumps({"user": anonymize(username, salt),
                                      "source": users[username],
                                      "entries": []}, ensure_ascii=False) + "\n")
                out.flush()
                continue

            entries = [e for lst in result["data"]["MediaListCollection"]["lists"]
                       for e in lst["entries"]]
            out.write(json.dumps({"user": anonymize(username, salt),
                                  "source": users[username],
                                  "entries": entries}, ensure_ascii=False) + "\n")
            out.flush()
            entries_total += len(entries)

            if i % 25 == 0:
                done = len(already) + i
                left = (len(todo) - i) * MIN_INTERVAL / 3600
                print(f"  {done}/{len(users)} user | {entries_total:,} tương tác "
                      f"| còn ~{left:.1f}h", flush=True)

    print(f"\nXong. Thu thêm {entries_total:,} tương tác -> {path}")


# --------------------------------------------------------------------------
# Giai đoạn 4: metadata cho mọi truyện đã gặp
# --------------------------------------------------------------------------

META_QUERY = """
query ($ids: [Int]) {
  Page(page: 1, perPage: 50) {
    media(id_in: $ids, type: MANGA) {
      id
      title { romaji english native }
      countryOfOrigin format status
      chapters volumes
      startDate { year month } endDate { year month }
      genres
      tags { name rank isMediaSpoiler }
      coverImage { extraLarge }
      staff { edges { role node { id name { full } } } }
      description(asHtml: false)
      averageScore meanScore popularity favourites
      isAdult
    }
  }
}
"""


def stage_media(client, data_dir):
    lists_path = data_dir / "lists.jsonl"
    if not lists_path.exists():
        raise SystemExit("Chưa có lists.jsonl — chạy `lists` trước.")

    wanted = set()
    with lists_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                for entry in json.loads(line).get("entries", []):
                    wanted.add(entry["mediaId"])

    path = data_dir / "media.jsonl"
    already = done_keys(path, "id")
    todo = sorted(wanted - already)

    print(f"Tổng {len(wanted):,} truyện | đã có {len(already):,} | còn {len(todo):,}")
    print(f"Ước tính {len(todo) / 50 * MIN_INTERVAL / 60:.1f} phút\n")

    with path.open("a", encoding="utf-8") as out:
        for i in range(0, len(todo), 50):
            batch = todo[i:i + 50]
            result = client.query(META_QUERY, {"ids": batch})
            if not result or "data" not in result:
                continue
            for media in result["data"]["Page"]["media"]:
                out.write(json.dumps(media, ensure_ascii=False) + "\n")
            out.flush()
            if (i // 50) % 10 == 0:
                print(f"  {min(i + 50, len(todo)):,}/{len(todo):,}", flush=True)

    print(f"\nXong -> {path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("stage", choices=["discover", "users", "lists", "media"])
    parser.add_argument("--data-dir", default="data/raw", type=Path)
    parser.add_argument("--per-country", type=int, default=2000,
                        help="Số truyện phổ biến lấy mỗi nước (giai đoạn discover)")
    parser.add_argument("--target", type=int, default=10_000,
                        help="Tổng số user cần thu (giai đoạn users)")
    parser.add_argument("--weights", default="KR=0.35,CN=0.35,JP=0.30",
                        help="Tỉ lệ user lấy theo nước nguồn. Manhua (CN) có catalog "
                             "nhỏ và ít người đọc hơn, nên cần quota cao mới đủ dày.")
    parser.add_argument("--pages-per-chunk", type=int, default=3,
                        help="Số trang hoạt động quét mỗi nhóm 25 truyện (giai đoạn "
                             "users). Đo được: quét thêm trang 4-6 trên một nhóm đã "
                             "quét vẫn ra 46-106 người MỚI, nên 3 là tự đặt ra, "
                             "không phải giới hạn thật của API.")
    args = parser.parse_args()

    weights = {c: 0.0 for c in COUNTRIES}
    for part in args.weights.split(","):
        country, _, value = part.partition("=")
        country = country.strip().upper()
        if country not in weights:
            raise SystemExit(f"Nước không hợp lệ: {country}. Chỉ nhận {COUNTRIES}")
        weights[country] = float(value)

    data_dir = out_dir(args.data_dir)
    client = Client()
    started = time.monotonic()

    if args.stage == "discover":
        stage_discover(client, data_dir, args.per_country)
    elif args.stage == "users":
        stage_users(client, data_dir, args.target, weights, args.pages_per_chunk)
    elif args.stage == "lists":
        stage_lists(client, data_dir, get_salt(data_dir / "salt.txt"))
    elif args.stage == "media":
        stage_media(client, data_dir)

    print(f"\n{client.calls} request trong {(time.monotonic() - started) / 60:.1f} phút")


if __name__ == "__main__":
    main()
