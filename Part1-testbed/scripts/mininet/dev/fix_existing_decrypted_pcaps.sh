#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  fix_existing_decrypted_pcaps.sh <carla_folder>

Description:
  Repairs already-generated *_decrypted.pcap files that accidentally include
  leading packets (for example from a previous frame-number filtering bug).

  For each *_decrypted.pcap:
    - Compares packet count with its matching original .pcap.
    - If decrypted_count > original_count, trims the extra leading packets.
    - Replaces the original decrypted file in place.

Examples:
  ./fix_existing_decrypted_pcaps.sh ~/carla1
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "" ]]; then
  echo "Error: missing <carla_folder>" >&2
  usage >&2
  exit 2
fi

CARLA_FOLDER="$1"
if [[ "${2:-}" != "" ]]; then
  echo "Error: unknown argument: ${2}" >&2
  usage >&2
  exit 2
fi

if [[ ! -d "$CARLA_FOLDER" ]]; then
  echo "Error: carla folder not found: $CARLA_FOLDER" >&2
  exit 2
fi

packet_count() {
  local pcap_path="$1"
  capinfos -c "$pcap_path" | awk -F': *' '
    /Number of packets/ {
      gsub(/[^0-9]/, "", $2)
      print $2
      exit
    }
  '
}

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

processed=0
fixed=0
skipped=0

while IFS= read -r -d '' DECRYPTED_PCAP; do
  processed=$((processed + 1))

  if [[ "$DECRYPTED_PCAP" != *_decrypted.pcap ]]; then
    continue
  fi

  ORIGINAL_PCAP="${DECRYPTED_PCAP%_decrypted.pcap}.pcap"
  if [[ ! -f "$ORIGINAL_PCAP" ]]; then
    echo "Skipping (missing original): $DECRYPTED_PCAP" >&2
    skipped=$((skipped + 1))
    continue
  fi

  decrypted_count="$(packet_count "$DECRYPTED_PCAP")"
  original_count="$(packet_count "$ORIGINAL_PCAP")"

  if [[ "$decrypted_count" == "" || "$original_count" == "" ]]; then
    echo "Skipping (count unavailable): $DECRYPTED_PCAP" >&2
    skipped=$((skipped + 1))
    continue
  fi

  if [[ "$decrypted_count" -le "$original_count" ]]; then
    echo "OK: $DECRYPTED_PCAP ($decrypted_count packets)"
    continue
  fi

  trim_count=$((decrypted_count - original_count))
  temp_out="$tmp_dir/trimmed.pcap"
  rm -f "$temp_out"

  # Drop the extra leading packets so count matches original capture.
  tshark -r "$DECRYPTED_PCAP" \
    -Y "frame.number > ${trim_count}" \
    -w "$temp_out"

  mv "$temp_out" "$DECRYPTED_PCAP"
  out_path="$DECRYPTED_PCAP"

  fixed=$((fixed + 1))
  echo "Fixed: $DECRYPTED_PCAP -> $out_path (trimmed $trim_count packets)"
done < <(find "$CARLA_FOLDER" -type f -name '*_decrypted.pcap' -print0)

echo "Processed: $processed, Fixed: $fixed, Skipped: $skipped"
