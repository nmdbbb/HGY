#!/usr/bin/env bash
# Tải bộ Goodreads UCSD (Comics & Graphic) về data/external/goodreads/.
#
# GIẤY PHÉP: chỉ dùng cho mục đích học thuật, KHÔNG phát hành lại, KHÔNG dùng
# thương mại. Phải trích dẫn Wan & McAuley (RecSys'18) và Wan, Misra, Nakashole
# & McAuley (ACL'19) — xem docs/goodreads-arm.md.
#
# Tải song song theo đoạn (range request). Đo được: máy chủ giới hạn ~21 KB/s
# trên MỖI kết nối nhưng không giới hạn tổng, nên 16 kết nối cho ~16x. Một
# kết nối đơn mất ~7 tiếng cho 549 MB; chia đoạn còn khoảng 35-40 phút.
#
# Resume được ở mức từng đoạn: ngắt giữa đường chạy lại chỉ tải phần thiếu.
set -u
BASE="https://mcauleylab.ucsd.edu/public_datasets/gdrive/goodreads"
OUT="data/external/goodreads"
PARTS="$OUT/.parts"
JOBS="${JOBS:-16}"
mkdir -p "$PARTS"

# Bỏ goodreads_reviews_comics_graphic.json.gz (139 MB): nhánh AniList không
# dùng đặc trưng văn bản nào, nên để đối chứng ngang nhau thì cũng không dùng.
FILES=(
  "byGenre/goodreads_books_comics_graphic.json.gz"
  "byGenre/goodreads_interactions_comics_graphic.json.gz"
  "goodreads_book_works.json.gz"
  "goodreads_book_authors.json.gz"
  "goodreads_book_series.json.gz"
)

fetch_part() {   # $1=url $2=dest $3=start $4=end
  local want=$(( $4 - $3 + 1 ))
  local have=$(stat -c %s "$2" 2>/dev/null || echo 0)
  [ "$have" = "$want" ] && return 0
  for attempt in 1 2 3 4 5; do
    curl -s -S --retry 2 -m 1800 -r "$3-$4" -o "$2" "$1" 2>/dev/null
    have=$(stat -c %s "$2" 2>/dev/null || echo 0)
    [ "$have" = "$want" ] && return 0
    sleep 3
  done
  echo "[LOI doan] $2 ($have/$want)" >&2
  return 1
}

for rel in "${FILES[@]}"; do
  name=$(basename "$rel")
  dest="$OUT/$name"
  url="$BASE/$rel"
  want=$(curl -sI -m 60 "$url" | grep -i '^content-length' | tr -d '\r' | awk '{print $2}')
  if [ -z "${want:-}" ]; then echo "[LOI] khong doc duoc kich thuoc $name"; continue; fi
  have=$(stat -c %s "$dest" 2>/dev/null || echo 0)
  if [ "$have" = "$want" ]; then echo "[xong roi] $name ($(( have / 1048576 )) MB)"; continue; fi

  # Chia thành các đoạn 4 MB, chạy JOBS đoạn cùng lúc.
  chunk=$((4 * 1048576))
  n=$(( (want + chunk - 1) / chunk ))
  echo "[tai] $name  $(( want / 1048576 )) MB, $n doan x 4 MB, $JOBS luong song song"
  rm -f "$PARTS/$name".*.tmp
  running=0
  for ((i = 0; i < n; i++)); do
    s=$((i * chunk)); e=$((s + chunk - 1)); [ $e -ge $want ] && e=$((want - 1))
    fetch_part "$url" "$(printf '%s/%s.%05d.part' "$PARTS" "$name" "$i")" "$s" "$e" &
    running=$((running + 1))
    if [ $running -ge $JOBS ]; then wait -n 2>/dev/null || wait; running=$((running - 1)); fi
  done
  wait

  # Ghép lại, chỉ khi đủ đoạn và đủ byte.
  got=$(ls "$PARTS/$name".*.part 2>/dev/null | wc -l)
  if [ "$got" != "$n" ]; then echo "[LOI] $name thieu doan: $got/$n, chay lai de tai tiep"; continue; fi
  cat "$PARTS/$name".*.part > "$dest"
  final=$(stat -c %s "$dest" 2>/dev/null || echo 0)
  if [ "$final" != "$want" ]; then
    echo "[LOI] $name ghep xong lech kich thuoc: $final vs $want"; continue
  fi
  if gzip -t "$dest" 2>/dev/null; then
    echo "[gzip ok] $name ($(( final / 1048576 )) MB)"
    rm -f "$PARTS/$name".*.part
  else
    echo "[GZIP LOI] $name - giu lai cac doan de chay lai"
  fi
done

echo "=== ket qua ==="
ls -la "$OUT"/*.json.gz 2>/dev/null
for f in "$OUT"/*.json.gz; do
  [ -e "$f" ] || continue
  if gzip -t "$f" 2>/dev/null; then echo "[ok] $(basename "$f")"; else echo "[LOI] $(basename "$f")"; fi
done
