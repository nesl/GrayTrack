#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 <root_folder_path>"
  exit 1
fi

ROOT_DIR="$1"

# Hardcoded GOP setup: one I-frame every N seconds.
FPS=20
IFRAME_INTERVAL_SEC=1
GOP=$((FPS * IFRAME_INTERVAL_SEC))

if [[ ! -d "$ROOT_DIR" ]]; then
  echo "Error: root folder not found: $ROOT_DIR"
  exit 1
fi

if ffmpeg -hide_banner -encoders 2>/dev/null | rg -q 'h264_nvenc'; then
  ENCODER="h264_nvenc"
  echo "Using GPU encoder: $ENCODER"
  CODEC_ARGS=(-c:v "$ENCODER" -preset p5 -bf 0 -g "$GOP" -keyint_min "$GOP")
else
  ENCODER="libx264"
  echo "GPU encoder unavailable, falling back to: $ENCODER"
  CODEC_ARGS=(-c:v "$ENCODER" -preset medium -bf 0 -x264-params "keyint=${GOP}:min-keyint=${GOP}:scenecut=0")
fi

shopt -s nullglob globstar
VIDEO_DIRS=("$ROOT_DIR"/**/videos)
shopt -u nullglob globstar

if [[ ${#VIDEO_DIRS[@]} -eq 0 ]]; then
  echo "No videos folders found under: $ROOT_DIR"
  exit 0
fi

reencoded_any=0
for video_dir in "${VIDEO_DIRS[@]}"; do
  shopt -s nullglob
  inputs=("$video_dir"/*.mp4)
  shopt -u nullglob

  for in_file in "${inputs[@]}"; do
    [[ "$in_file" == *_reenc.mp4 ]] && continue
    out_file="${in_file%.*}_reenc.mp4"

    echo "Re-encoding: $in_file -> $out_file"
    ffmpeg -y -i "$in_file" \
      -map 0:v:0 -an \
      -r "$FPS" \
      "${CODEC_ARGS[@]}" \
      -pix_fmt yuv420p \
      -force_key_frames "expr:gte(t,n_forced*${IFRAME_INTERVAL_SEC})" \
      "$out_file"
    reencoded_any=1
  done
done

if [[ $reencoded_any -eq 0 ]]; then
  echo "No matching video inputs found (*.mp4)."
  exit 0
fi

echo "Done."
