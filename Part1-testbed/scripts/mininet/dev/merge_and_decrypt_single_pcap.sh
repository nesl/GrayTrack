#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  merge_and_decrypt_pcap.sh <video_pcap> [decrypted_pcap]

Examples:
  ./merge_and_decrypt_pcap.sh camera_1.pcap decrypted.pcap
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ "${1:-}" == "" ]]; then
  echo "Error: missing <video_pcap>" >&2
  usage >&2
  exit 2
fi

VIDEO_PCAP="$1"
OUT_PCAP="${2:-${VIDEO_PCAP%.pcap}-decrypted.pcap}"

# Hardcoded decryption keys (matches your provided tshark reference).
WPA_PWD="123456780"
WPA_SSID="ssid-wifi"

if [[ ! -f "$VIDEO_PCAP" ]]; then
  echo "Error: video pcap not found: $VIDEO_PCAP" >&2
  exit 2
fi

infer_handshake_pcap() {
  local video="$1"
  local abs_video=""

  # If readlink -f is available, use it so regex matching is more reliable.
  if abs_video="$(readlink -f -- "$video" 2>/dev/null)"; then
    : # abs_video set
  else
    abs_video="$video"
  fi

  # Expected layout: .../carla<NUM>/.../<some_video>.pcap
  # Handshake:        .../carla<NUM>/handshake.pcap
  if [[ "$abs_video" =~ ^(.*?/carla[0-9]+)/ ]]; then
    echo "${BASH_REMATCH[1]}/handshake.pcap"
    return 0
  fi

  return 1
}

if ! HANDSHAKE_PCAP="$(infer_handshake_pcap "$VIDEO_PCAP")"; then
  echo "Error: could not infer handshake pcap from video path: $VIDEO_PCAP" >&2
  echo "Expected a path segment like: /carla<NUM>/" >&2
  exit 2
fi

if [[ ! -f "$HANDSHAKE_PCAP" ]]; then
  echo "Error: handshake pcap not found: $HANDSHAKE_PCAP" >&2
  exit 2
fi

key_opt="uat:80211_keys:\"wpa-pwd\",\"${WPA_PWD}:${WPA_SSID}\""

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

merged_pcap="$tmp_dir/merged.pcap"
mergecap -w "$merged_pcap" "$HANDSHAKE_PCAP" "$VIDEO_PCAP"
input_for_decryption="$merged_pcap"

echo "Decrypting: $input_for_decryption"
echo "Writing: $OUT_PCAP"
tshark -r "$input_for_decryption" \
  -o wlan.enable_decryption:TRUE \
  -o "$key_opt" \
  -w "$OUT_PCAP"

echo "Done."
