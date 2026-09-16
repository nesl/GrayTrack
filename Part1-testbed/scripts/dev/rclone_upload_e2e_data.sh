#!/usr/bin/env bash
# Upload e2e pipeline data to Google Drive with README folder names.
# Local paths are never renamed; destination names match README.md.
#
# Drive layout:
#   <remote>:data/carla/...
#   <remote>:data/research/<carla_*>/...
#
# Usage:
#   bash scripts/dev/rclone_upload_e2e_data.sh
#   bash scripts/dev/rclone_upload_e2e_data.sh --dry-run
#   RCLONE_REMOTE=other-remote bash scripts/dev/rclone_upload_e2e_data.sh
#
# Config password is prompted once per run and kept in memory (RCLONE_CONFIG_PASS).
# Or export RCLONE_CONFIG_PASS yourself before running.

set -euo pipefail

REMOTE="${RCLONE_REMOTE:-google-drive}"
ROOT="data"
DRY_RUN=()

if [[ "${1:-}" == "--dry-run" ]]; then
  DRY_RUN=(--dry-run)
  shift
fi

if ! command -v rclone >/dev/null 2>&1; then
  echo "error: rclone not found on PATH" >&2
  exit 2
fi

# Prompt once per run; reuse via RCLONE_CONFIG_PASS for all rclone calls.
if [[ -z "${RCLONE_CONFIG_PASS:-}" ]]; then
  read -r -s -p "Enter rclone configuration password: " RCLONE_CONFIG_PASS
  echo
  export RCLONE_CONFIG_PASS
fi

if ! rclone listremotes | grep -qx "${REMOTE}:"; then
  echo "error: rclone remote '${REMOTE}:' not configured. Set RCLONE_REMOTE or run: rclone config" >&2
  rclone listremotes >&2 || true
  exit 2
fi

# rclone copy SRC DEST  — DEST is the renamed folder on Drive under data/
copy_dir() {
  local src="$1"
  local dest_rel="$2" # path under data/, e.g. carla or research/carla_combined

  if [[ ! -d "$src" ]]; then
    echo "SKIP (missing): $src" >&2
    return 0
  fi

  local dest="${REMOTE}:${ROOT}/${dest_rel}"
  echo "UPLOAD: $src  ->  $dest"
  rclone copy "$src" "$dest" \
    --create-empty-src-dirs \
    --progress \
    --transfers 8 \
    --checkers 16 \
    "${DRY_RUN[@]}"
}

echo "Remote: ${REMOTE}:  Root: ${ROOT}/"
echo "Dry-run: ${DRY_RUN[*]:-no}"
echo

# --- Steps 1–5: merge carla1..carla8 into data/carla ---
for i in 1 2 3 4 5 6 7 8; do
  copy_dir "/home/ubuntu/carla_data_aug/carla${i}" "carla"
done

# --- Step 6–7: combined packet CSVs ---
copy_dir "/media/ubuntu/research/carla_data_aug_combined" "research/carla_combined"

# --- Step 8: RTP-marker infer (packet-level) ---
copy_dir "/media/ubuntu/research/carla_data_aug_combined_x_infer" "research/carla_x_infer"

# --- Step 9: video-frame truth for infer ---
copy_dir "/media/ubuntu/research/carla_data_aug_infer_truth" "research/carla_truth"

# --- Step 10: stage-1 preprocess ---
copy_dir "/media/ubuntu/research/carla_data_aug_combined_x_infer_preprocessed" "research/carla_x_infer_preprocessed"
copy_dir "/media/ubuntu/research/carla_data_aug_infer_truth_interpolated" "research/carla_truth_interpolated"

# --- Step 11–12: no-warmup X/Y ---
copy_dir "/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup" "research/carla_x_infer_no_warmup"
copy_dir "/media/ubuntu/research/carla_data_aug_infer_truth_no_warmup" "research/carla_truth_no_warmup"

# --- Step 13: held-out test split ---
copy_dir "/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup_7_test" "research/carla_x_infer_test"
copy_dir "/media/ubuntu/research/carla_data_aug_infer_truth_7_test" "research/carla_truth_test"

# --- Step 14: car_visible preds on test ---
copy_dir "/media/ubuntu/research/carla_data_aug_car_visible_pred_7_test" "research/carla_car_visible_pred_test"

# --- Step 15: event eval ---
copy_dir "/media/ubuntu/research/carla_data_aug_event_eval_7_test" "research/carla_event_eval"

# --- Steps 16–17: passage-events export (grouped) ---
copy_dir "/media/ubuntu/research/passage_events_export_7_test" "research/carla_passage_events"

echo
echo "Done. Drive tree under ${REMOTE}:${ROOT}/:"
echo "  carla/"
echo "  research/carla_combined/"
echo "  research/carla_x_infer/"
echo "  research/carla_truth/"
echo "  research/carla_x_infer_preprocessed/"
echo "  research/carla_truth_interpolated/"
echo "  research/carla_x_infer_no_warmup/"
echo "  research/carla_truth_no_warmup/"
echo "  research/carla_x_infer_test/"
echo "  research/carla_truth_test/"
echo "  research/carla_car_visible_pred_test/"
echo "  research/carla_event_eval/"
echo "  research/carla_passage_events/"
echo
echo "Local restore mapping (README paths):"
echo "  ${REMOTE}:${ROOT}/carla                         -> /home/ubuntu/carla"
echo "  ${REMOTE}:${ROOT}/research/<name>               -> /media/ubuntu/research/<name>"
