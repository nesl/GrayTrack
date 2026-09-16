#!/usr/bin/env python3
"""RTP diagnostics from a PCAP: SSRC, marker counts, and timestamp checks."""

import argparse
import sys
from collections import Counter

import pyshark


def _open_capture(pcap_path, udp_port):
    return pyshark.FileCapture(
        pcap_path,
        decode_as={f"udp.port=={udp_port}": "rtp"},
    )


def analyze_rtp(pcap_path, udp_port=5000):
    """Single pass over RTP packets; mirrors the tshark checks in the module docstring."""
    cap = _open_capture(pcap_path, udp_port)
    ssrc_counts = Counter()
    rtp_packets = 0
    marker_packets = 0
    rtp_timestamps = set()
    marker_per_timestamp = Counter()

    try:
        for pkt in cap:
            if not hasattr(pkt, "rtp"):
                continue
            try:
                ssrc = str(pkt.rtp.ssrc)
                timestamp = str(pkt.rtp.timestamp)
                marker = int(pkt.rtp.marker)
            except Exception:
                continue

            rtp_packets += 1
            ssrc_counts[ssrc] += 1
            rtp_timestamps.add(timestamp)

            if marker == 1:
                marker_packets += 1
                marker_per_timestamp[timestamp] += 1
    finally:
        try:
            cap.close()
        except Exception:
            pass

    return {
        "ssrc_counts": ssrc_counts,
        "rtp_packets": rtp_packets,
        "marker_packets": marker_packets,
        "unique_timestamps": len(rtp_timestamps),
        "marker_per_timestamp": marker_per_timestamp,
    }


def print_marker_timestamp_duplicates(marker_per_timestamp, head):
    """Top timestamps by marker count (same as uniq -c | sort -nr | head)."""
    rows = [
        (count, timestamp)
        for timestamp, count in marker_per_timestamp.items()
    ]
    rows.sort(key=lambda row: (-row[0], row[1]))
    for count, timestamp in rows[:head]:
        print(f"{count:7d} {timestamp}")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "RTP diagnostics from a PCAP: SSRC, marker packet count, "
            "unique rtp.timestamp count, and duplicate markers per timestamp."
        ),
    )
    parser.add_argument("pcap", help="Path to the PCAP file")
    parser.add_argument(
        "--udp-port",
        type=int,
        default=5000,
        help="UDP port to decode as RTP (default: 5000)",
    )
    parser.add_argument(
        "--head",
        type=int,
        default=10,
        help="How many top duplicate-marker timestamps to print (default: 10)",
    )
    args = parser.parse_args()

    try:
        stats = analyze_rtp(args.pcap, udp_port=args.udp_port)
    except FileNotFoundError:
        print(f"error: pcap not found: {args.pcap}", file=sys.stderr)
        sys.exit(2)
    except Exception as exc:
        print(f"error: failed to read pcap: {exc!r}", file=sys.stderr)
        sys.exit(1)

    if stats["rtp_packets"] == 0:
        print("no RTP packets found")
        sys.exit(1)

    print("=== rtp.ssrc ===")
    for ssrc, count in stats["ssrc_counts"].most_common():
        print(f"rtp.ssrc={ssrc}  packets={count}")

    print()
    print("=== marker packets (rtp.marker == 1) ===")
    print(stats["marker_packets"])

    print()
    print("=== unique rtp.timestamp (all RTP packets) ===")
    print(stats["unique_timestamps"])

    print()
    print("=== markers per rtp.timestamp (count timestamp), top by count ===")
    print_marker_timestamp_duplicates(stats["marker_per_timestamp"], args.head)


if __name__ == "__main__":
    main()
