import argparse
import csv
import glob
import os
import shutil
import sys
import time

import pyshark

# If True, skip cameras whose output CSVs already exist.
SKIP_IF_OUTPUT_EXISTS = True

# Wireshark display filter: uplink frames from the virtual STA.
PCAP_DISPLAY_FILTER = (
    "wlan.fc.tods == 1 && wlan.fc.fromds == 0 && wlan.ta == 02:00:00:00:00:00"
)


def packet_matches_pcap_filter(pkt):
    if not hasattr(pkt, "wlan"):
        return False
    try:
        wlan = pkt.wlan
        return (
            int(wlan.fc_tods) == 1
            and int(wlan.fc_fromds) == 0
            and str(wlan.ta).lower() == "02:00:00:00:00:00"
        )
    except (AttributeError, ValueError):
        return False


def iter_decrypted_y_packets(decrypted_pcap_path, handshake_packet_count=0):
    """Skip handshake packets in raw file order, then yield WLAN-filtered packets."""
    cap = pyshark.FileCapture(
        decrypted_pcap_path,
        decode_as={"udp.port==5000": "rtp"},
    )
    try:
        raw_idx = 0
        for pkt in cap:
            raw_idx += 1
            if raw_idx <= handshake_packet_count:
                continue
            if not packet_matches_pcap_filter(pkt):
                continue
            yield pkt
    finally:
        try:
            cap.close()
        except Exception:
            pass


def iter_dataset_dirs(root_dir):
    for entry in sorted(glob.glob(os.path.join(root_dir, "*"))):
        if os.path.isdir(entry):
            yield entry


def extract_x_data(original_pcap_path):
    rows = []
    cap = pyshark.FileCapture(
        original_pcap_path,
        display_filter=PCAP_DISPLAY_FILTER,
    )
    try:
        for pkt in cap:
            try:
                timestamp = float(pkt.sniff_timestamp)
                size = int(pkt.length)
            except Exception:
                continue
            rows.append((timestamp, size))
    finally:
        try:
            cap.close()
        except Exception:
            pass
    return rows


def count_packets(pcap_path, display_filter=None):
    count = 0
    kwargs = {}
    if display_filter is not None:
        kwargs["display_filter"] = display_filter
    cap = pyshark.FileCapture(pcap_path, **kwargs)
    try:
        for _ in cap:
            count += 1
    finally:
        try:
            cap.close()
        except Exception:
            pass
    return count


def extract_y_data(decrypted_pcap_path, handshake_packet_count=0):
    rows = []
    for pkt in iter_decrypted_y_packets(decrypted_pcap_path, handshake_packet_count):
        try:
            timestamp = float(pkt.sniff_timestamp)
        except Exception:
            continue

        is_frame_boundary = 0
        if hasattr(pkt, "rtp"):
            try:
                is_frame_boundary = int(pkt.rtp.marker) == 1
            except Exception:
                is_frame_boundary = 0

        rows.append((timestamp, int(is_frame_boundary)))
    return rows


def write_csv(path, header, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows)


def count_truth_frames(truth_csv_path):
    with open(truth_csv_path, newline="", encoding="utf-8") as f:
        return sum(1 for _ in f) - 1


def truncate_truth_csv(truth_csv_path, n_frames):
    backup_path = f"{os.path.splitext(truth_csv_path)[0]}_backup.csv"
    shutil.copy2(truth_csv_path, backup_path)
    with open(truth_csv_path, newline="", encoding="utf-8") as f:
        rows = list(csv.reader(f))
    header, data = rows[0], rows[1 : n_frames + 1]
    with open(truth_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
    return backup_path


def trim_to_aligned_frames(x_rows, y_rows, n_truth):
    """Keep packets through marker N, N = min(n_markers, n_truth).

    Drops an incomplete trailing frame (packets after the last kept marker)
    and caps at the truth frame count when the capture is longer.
    """
    marker_idxs = [i for i, row in enumerate(y_rows) if int(row[1]) == 1]
    n_keep = min(len(marker_idxs), n_truth)
    end = marker_idxs[n_keep - 1] + 1
    return x_rows[:end], y_rows[:end], n_keep


def read_handshake_packet_count(root_dir):
    handshake_count_path = os.path.join(root_dir, "handshake_packet_count.txt")
    if os.path.exists(handshake_count_path):
        with open(handshake_count_path, "r", encoding="utf-8") as f:
            return int(f.read().strip())
    return 0


def process_dataset(dataset_dir, handshake_packet_count):
    videos_dir = os.path.join(dataset_dir, "videos")
    x_dir = os.path.join(dataset_dir, "x")
    y_dir = os.path.join(dataset_dir, "y")
    pcap_suffix = "_decrypted.pcap"

    if not os.path.isdir(videos_dir):
        return

    pcap_paths = sorted(glob.glob(os.path.join(videos_dir, f"*{pcap_suffix}")))
    if not pcap_paths:
        return

    print(f"dataset: {dataset_dir}")
    total_files = 0
    files_written = 0
    files_skipped_missing_original = 0
    files_skipped_errors = 0
    files_skipped_row_mismatch = 0
    files_skipped_existing_output = 0

    for decrypted_pcap_path in pcap_paths:
        total_files += 1
        camera_name = os.path.basename(decrypted_pcap_path)[: -len(pcap_suffix)]
        original_pcap_path = os.path.join(videos_dir, f"{camera_name}.pcap")
        truth_csv_path = os.path.join(y_dir, f"{camera_name}_truth.csv")
        x_csv_path = os.path.join(x_dir, f"{camera_name}_x.csv")
        y_csv_path = os.path.join(y_dir, f"{camera_name}_y.csv")

        print(f"original:  {original_pcap_path}")
        print(f"decrypted: {decrypted_pcap_path}")

        if SKIP_IF_OUTPUT_EXISTS and os.path.exists(x_csv_path) and os.path.exists(y_csv_path):
            files_skipped_existing_output += 1
            print(
                (
                    f"skip: output already exists for {camera_name}: "
                    f"{x_csv_path}, {y_csv_path}"
                )
            )
            continue

        if not os.path.exists(original_pcap_path):
            print(
                f"warning: original pcap missing for {camera_name}: {original_pcap_path}",
                file=sys.stderr,
            )
            files_skipped_missing_original += 1
            continue

        try:
            original_packet_count = count_packets(
                original_pcap_path,
                display_filter=PCAP_DISPLAY_FILTER,
            )
            decrypted_packet_count = sum(
                1 for _ in iter_decrypted_y_packets(
                    decrypted_pcap_path, handshake_packet_count
                )
            )
            print(f"original packet count:  {original_packet_count}")
            print(f"decrypted packet count: {decrypted_packet_count}")

            x_rows = extract_x_data(original_pcap_path)
            y_rows = extract_y_data(decrypted_pcap_path, handshake_packet_count)
        except Exception as exc:
            files_skipped_errors += 1
            print(
                (
                    f"error: failed to process camera {camera_name}. "
                    f"original={original_pcap_path}, decrypted={decrypted_pcap_path}. "
                    f"exception={exc!r}. waiting 30 seconds and skipping file."
                ),
                file=sys.stderr,
            )
            time.sleep(30)
            continue

        if len(x_rows) != len(y_rows):
            files_skipped_row_mismatch += 1
            print(
                (
                    f"error: row mismatch for {camera_name}. "
                    f"original={original_pcap_path}, decrypted={decrypted_pcap_path}. "
                    f"_x has {len(x_rows)} rows, _y has {len(y_rows)} rows. "
                    "waiting 30 seconds and skipping file."
                ),
                file=sys.stderr,
            )
            time.sleep(30)
            continue

        n_truth = count_truth_frames(truth_csv_path)
        x_rows, y_rows, n_frames = trim_to_aligned_frames(x_rows, y_rows, n_truth)
        if n_frames < n_truth:
            backup_path = truncate_truth_csv(truth_csv_path, n_frames)
            print(
                f"trimmed truth to {n_frames} frames (was {n_truth}); "
                f"backup: {backup_path}"
            )

        write_csv(x_csv_path, ["timestamp", "packet_size"], x_rows)
        write_csv(y_csv_path, ["timestamp", "is_frame_boundary"], y_rows)
        files_written += 1

        print(f"wrote:     {x_csv_path}")
        print(f"wrote:     {y_csv_path}")
        print(f"rows:      {len(x_rows)} (aligned frames: {n_frames})")

    print(
        (
            "dataset summary: "
            f"total={total_files}, "
            f"written={files_written}, "
            f"skipped_missing_original={files_skipped_missing_original}, "
            f"skipped_existing_output={files_skipped_existing_output}, "
            f"skipped_processing_errors={files_skipped_errors}, "
            f"skipped_row_mismatch={files_skipped_row_mismatch}"
        ),
        file=sys.stderr,
    )
    print()


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Generate _x and _y CSV datasets from original and decrypted PCAP files."
        )
    )
    parser.add_argument(
        "root_dir",
        help="Root directory containing <datetime_seed>/videos folder(s)",
    )
    args = parser.parse_args()

    root_dir = os.path.abspath(args.root_dir)
    if not os.path.isdir(root_dir):
        print(f"error: root directory not found: {root_dir}", file=sys.stderr)
        sys.exit(2)

    handshake_packet_count = read_handshake_packet_count(root_dir)
    print(f"handshake packet count to skip: {handshake_packet_count}")

    for dataset_dir in iter_dataset_dirs(root_dir):
        process_dataset(dataset_dir, handshake_packet_count)


if __name__ == "__main__":
    main()
