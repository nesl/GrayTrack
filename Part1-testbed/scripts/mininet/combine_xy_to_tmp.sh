#!/usr/bin/env bash

set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <root_dir>"
  echo "Example: $0 /media/ubuntu/research/carla1"
  exit 2
fi

root_dir="$1"
out_dir="/media/ubuntu/research/carla_data_aug_combined"

if [[ ! -d "$root_dir" ]]; then
  echo "error: root directory not found: $root_dir" >&2
  exit 2
fi

mkdir -p "$out_dir"
shopt -s nullglob

processed=0
skipped_missing_y=0

for episode_dir in "$root_dir"/*; do
  [[ -d "$episode_dir" ]] || continue

  episode_x_dir="$episode_dir/x"
  episode_y_dir="$episode_dir/y"
  if [[ ! -d "$episode_x_dir" || ! -d "$episode_y_dir" ]]; then
    continue
  fi

  mapfile -t x_files < <(rg --files -g '**/*_x.csv' "$episode_x_dir")
  for x_csv in "${x_files[@]}"; do
    [[ "$x_csv" = /* ]] || x_csv="$episode_x_dir/$x_csv"

    rel_in_x="${x_csv#"$episode_x_dir"/}"
    y_csv="$episode_y_dir/${rel_in_x%_x.csv}_y.csv"

    if [[ ! -f "$y_csv" ]]; then
      echo "skip: missing pair for $x_csv" >&2
      ((skipped_missing_y += 1))
      continue
    fi

    folder_id="$(basename "$episode_dir")"
    x_base="$(basename "${x_csv%_x.csv}")"
    ts="$(date +%Y%m%d_%H%M%S_%N)"
    out_csv="$out_dir/${folder_id}_${x_base}_${ts}.csv"

    awk -F',' '
      BEGIN {
        header = "frame.number,frame.time_epoch,frame.time_relative,frame.len,rtp.seq,rtp.timestamp,rtp.marker"
      }
      NR == FNR {
        if (FNR == 1) next
        gsub(/\r/, "", $0)
        xn++
        xt[xn] = $1 + 0
        xlen[xn] = $2
        next
      }
      {
        if (FNR == 1) next
        gsub(/\r/, "", $0)
        yn++
        ymark[yn] = $2
        next
      }
      END {
        n = (xn < yn ? xn : yn)
        if (n == 0) {
          print header
          exit
        }
        first = xt[1]
        print header
        for (i = 1; i <= n; i++) {
          rel = xt[i] - first
          printf "%d,%.9f,%.9f,%s,,,%s\n", i, xt[i], rel, xlen[i], ymark[i]
        }
      }
    ' "$x_csv" "$y_csv" > "$out_csv"

    echo "wrote: $out_csv"
    ((processed += 1))
  done
done

echo "done: processed=$processed skipped_missing_y=$skipped_missing_y"
