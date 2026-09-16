#!/usr/bin/env bash

set -euo pipefail

# Hardcoded input video path. Update this path as needed.
INPUT_MP4="/media/ubuntu/research/carla3/2026_03_04_01_22_34_85/videos/camera_1.mp4"

ffmpeg -hide_banner -i "$INPUT_MP4" -vf "showinfo" -f null - 2>&1 \
| awk '
  /showinfo/ {
    frame_id = ""
    frame_type = ""
    for (i = 1; i <= NF; i++) {
      if ($i == "n:" && (i + 1) <= NF) {
        frame_id = $(i + 1)
      }
      if ($i ~ /^type:/) {
        frame_type = tolower(substr($i, 6, 1))
      }
    }
    if (frame_id != "" && frame_type != "") {
      print "frame_id=" frame_id ", frame_type=" frame_type
    }
  }
'
