#!/usr/bin/env bash
# For each *.csv in carla_data_aug_combined, find the matching *_truth.csv under
# /home/ubuntu/carla_data_aug/carla1..8. (Same stem as batch_infer_rtp_marker.py uses.)
# Truth layout (per root):
#   <carla_root>/<run_id>/y/camera_N_truth.csv
# e.g. .../carla_data_aug/carla5/2026_07_28_16_07_15_126/y/camera_1_truth.csv
#
# Combined / infer stems look like:
#   2026_03_03_02_54_14_43_camera_1_20260505_134042_263380767[_rtp_marker_pred].csv
# Match key is only run_id + camera (timestamp after camera_* is ignored):
#   2026_03_03_02_54_14_43_camera_1
#
# Phase 1: index every *_truth.csv under each root's */y/ by "${run_id}_${camera_stem}".
# Phase 2: for each combined csv basename (stem), parse the same key and copy the truth file.
#
# Output: /media/ubuntu/research/carla_data_aug_infer_truth/<stem>_rtp_marker_pred_truth.csv
#
# Usage: copy_truth_csv_for_xy_infer.sh

set -euo pipefail

readonly COMBINED_DIR="/media/ubuntu/research/carla_data_aug_combined"
readonly OUT_DIR="/media/ubuntu/research/carla_data_aug_infer_truth"
readonly CARLA_ROOTS=(
  /home/ubuntu/carla_data_aug/carla1
  /home/ubuntu/carla_data_aug/carla2
  /home/ubuntu/carla_data_aug/carla3
  /home/ubuntu/carla_data_aug/carla4
  /home/ubuntu/carla_data_aug/carla5
  /home/ubuntu/carla_data_aug/carla6
  /home/ubuntu/carla_data_aug/carla7
  /home/ubuntu/carla_data_aug/carla8
)

if [[ ! -d "$COMBINED_DIR" ]]; then
  echo "error: combined csv directory not found: $COMBINED_DIR" >&2
  exit 2
fi

mkdir -p "$OUT_DIR"

shopt -s nullglob

# episode_key -> absolute path to truth csv (last wins on duplicate key; we warn)
declare -A truth_by_key=()

for carla in "${CARLA_ROOTS[@]}"; do
  [[ -d "$carla" ]] || continue

  for truth in "$carla"/*/y/*_truth.csv; do
    [[ -f "$truth" ]] || continue

    run_id="$(basename "$(dirname "$(dirname "$truth")")")"
    truth_bn="$(basename "$truth")"
    cam_stem="${truth_bn%_truth.csv}"
    key="${run_id}_${cam_stem}"

    if [[ -n "${truth_by_key[$key]+isset}" ]]; then
      echo "warning: duplicate key ${key}; replacing ${truth_by_key[$key]} with ${truth}" >&2
    fi
    truth_by_key["$key"]="$truth"
  done
done

if ((${#truth_by_key[@]} == 0)); then
  echo "warning: no truth csv files indexed under carla roots (check paths)" >&2
fi

infer_episode_key_from_stem() {
  # stem from combined basename, e.g. 2026_03_03_02_54_14_43_camera_1_20260505_134042_263380767
  # run_id is digits/underscores; require literal _camera_<digits>_ then timestamp.
  local stem="$1"
  if [[ "$stem" =~ ^([0-9_]+_camera_[0-9]+)_ ]]; then
    printf '%s' "${BASH_REMATCH[1]}"
    return 0
  fi
  return 1
}

copied=0
skipped_nomatch=0
skipped_exists=0
skipped_badstem=0

scanned=0
for combined in "$COMBINED_DIR"/*.csv; do
  [[ -f "$combined" ]] || continue
  ((scanned += 1)) || true

  combined_base="$(basename "$combined")"
  stem="${combined_base%.csv}"
  # Same basename pattern as batch_infer_rtp_marker_tmp.py output (stem + _rtp_marker_pred.csv).
  dest="$OUT_DIR/${stem}_rtp_marker_pred_truth.csv"

  if [[ -f "$dest" ]]; then
    echo "skip (already exists): $dest"
    ((skipped_exists += 1)) || true
    continue
  fi

  if ! episode_key="$(infer_episode_key_from_stem "$stem")"; then
    echo "warning: cannot parse run_id_camera_N from combined stem: $stem" >&2
    ((skipped_badstem += 1)) || true
    continue
  fi

  found="${truth_by_key[$episode_key]-}"

  if [[ -z "$found" ]]; then
    echo "warning: no truth for key=${episode_key} (combined=$combined_base)" >&2
    ((skipped_nomatch += 1)) || true
    continue
  fi

  cp -n -- "$found" "$dest"
  echo "copied: $found -> $dest (key=$episode_key)"
  ((copied += 1)) || true
done

echo "done: scanned_combined=$scanned copied=$copied skipped_existing_dest=$skipped_exists no_match=$skipped_nomatch bad_stem=$skipped_badstem truth_index_size=${#truth_by_key[@]}"
