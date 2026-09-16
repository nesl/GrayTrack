#!/usr/bin/env bash
# Regroup a flat export_passage_events.py output into one folder per capture run.
#
#   <export>/x/<seq>.csv                          -> <export>/<run>/x/<seq>.csv
#   <export>/y_truth/<seq>_rtp_marker_pred_truth.csv -> <export>/<run>/y/<seq>_...csv
#
# The run key is the sequence stem up to "_camera_". passage_events_index.csv is
# rewritten in place with a leading `run` column and the new relative paths.
#
# Usage: group_passage_events_by_run.sh [/media/ubuntu/research/passage_events_export_7]

set -euo pipefail

export_dir="${1:-/media/ubuntu/research/passage_events_export_7}"
cd "$export_dir"

run_of() {
    # Strip everything from "_camera_" onward.
    local base="$1"
    printf '%s' "${base%%_camera_*}"
}

moved_x=0
for f in x/*.csv; do
    base="$(basename "$f" .csv)"
    run="$(run_of "$base")"
    mkdir -p "$run/x"
    mv "$f" "$run/x/"
    moved_x=$((moved_x + 1))
done

moved_y=0
for f in y_truth/*.csv; do
    base="$(basename "$f")"
    run="$(run_of "$base")"
    mkdir -p "$run/y"
    mv "$f" "$run/y/"
    moved_y=$((moved_y + 1))
done

rmdir x y_truth

awk 'BEGIN { FS = OFS = "," }
     NR == 1 { $4 = "y_csv"; print "run", $0; next }
     {
         i = index($1, "_camera_")
         run = substr($1, 1, i - 1)
         n = split($3, p, "/"); $3 = run "/x/" p[n]
         m = split($4, q, "/"); $4 = run "/y/" q[m]
         print run, $0
     }' passage_events_index.csv > passage_events_index.csv.tmp
mv passage_events_index.csv.tmp passage_events_index.csv

echo "Moved $moved_x x files and $moved_y y files into $(find . -mindepth 1 -maxdepth 1 -type d | wc -l) run folders"
