#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  merge_and_decrypt_pcap_all.sh <carla_folder>

Examples:
  ./merge_and_decrypt_pcap_all.sh ~/carla1
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

# Hardcoded decryption keys (matches your provided tshark reference).
WPA_PWD="123456780"
WPA_SSID="ssid-wifi"

if [[ ! -d "$CARLA_FOLDER" ]]; then
  echo "Error: carla folder not found: $CARLA_FOLDER" >&2
  exit 2
fi

HANDSHAKE_PCAP="$CARLA_FOLDER/handshake.pcap"
HANDSHAKE_PACKET_COUNT_FILE="$CARLA_FOLDER/handshake_packet_count.txt"

if [[ ! -f "$HANDSHAKE_PCAP" ]]; then
  echo "Error: handshake pcap not found: $HANDSHAKE_PCAP" >&2
  exit 2
fi

key_opt="uat:80211_keys:\"wpa-pwd\",\"${WPA_PWD}:${WPA_SSID}\""

HANDSHAKE_PACKET_COUNT="$(
  capinfos -c "$HANDSHAKE_PCAP" | awk -F': *' '
    /Number of packets/ {
      gsub(/[^0-9]/, "", $2)
      print $2
      exit
    }
  '
)"

if [[ "$HANDSHAKE_PACKET_COUNT" == "" || ! "$HANDSHAKE_PACKET_COUNT" =~ ^[0-9]+$ ]]; then
  echo "Error: could not determine packet count for handshake pcap: $HANDSHAKE_PCAP" >&2
  exit 2
fi

# Save handshake packet count once per run for downstream dataset generation.
printf '%s\n' "$HANDSHAKE_PACKET_COUNT" > "$HANDSHAKE_PACKET_COUNT_FILE"
echo "Handshake packet count: $HANDSHAKE_PACKET_COUNT (saved to $HANDSHAKE_PACKET_COUNT_FILE)"

tmp_dir="$(mktemp -d)"
cleanup() {
  rm -rf "$tmp_dir"
}
trap cleanup EXIT

while IFS= read -r -d '' VIDEO_PCAP; do
  if [[ "$VIDEO_PCAP" == "$HANDSHAKE_PCAP" ]]; then
    continue
  fi

  if [[ "$VIDEO_PCAP" == *_decrypted.pcap ]]; then
    continue
  fi

  OUT_PCAP="${VIDEO_PCAP%.pcap}_decrypted.pcap"
  merged_pcap="$tmp_dir/merged.pcap"
  decrypted_pcap="$tmp_dir/decrypted.pcap"

  rm -f "$merged_pcap" "$decrypted_pcap"

  mergecap -w "$merged_pcap" "$HANDSHAKE_PCAP" "$VIDEO_PCAP"
  input_for_decryption="$merged_pcap"

  echo "Decrypting: $input_for_decryption"
  echo "Writing: $OUT_PCAP"
  tshark -r "$input_for_decryption" \
    -o wlan.enable_decryption:TRUE \
    -o "$key_opt" \
    -w "$decrypted_pcap"

  cp "$decrypted_pcap" "$OUT_PCAP"
done < <(find "$CARLA_FOLDER" -type f -name '*.pcap' -print0)

echo "Done."
