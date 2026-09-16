# Part 1 · Testbed and passage detection

CARLA capture → Mininet-WiFi encrypted replay → packet grouping → frame-size features → vehicle visibility → anonymous passage events.

## Setup

Use a separate Python environment compatible with the pinned dependencies and your CARLA Python API. The NumPy pin requires Python below 3.11. Install CARLA and Mininet-WiFi separately, along with `ffmpeg`, `ffprobe`, `tshark`, `mergecap`, `capinfos`, and `tcpdump`. Mininet-WiFi requires a Linux networking environment and root privileges for emulation.

From this folder:

```bash
python -m pip install -r requirements.txt
```

The original scripts contain machine-specific dataset and checkpoint paths, including `/media/ubuntu/research/` and `/home/ubuntu/carla`. Update the relevant script settings before running. Raw captures, trained weights, and normalization files are not bundled.

## Run the pipeline

Follow the [detailed guide](docs/guide.md) for the ordered capture, training, inference, and export commands.

| Folder | Purpose |
| --- | --- |
| `scripts/` | CARLA capture and preprocessing utilities. |
| `scripts/mininet/` | Wireless replay, PCAP processing, and PacketSegformer. |
| `scripts/camera_predict_model/` | CarVisibleLSTM, passage-event export, and evaluation. |
| `scripts/dev/`, `scripts/misc/` | Original environment and capture helpers. |
| `docs/` | Detailed guide. |

The final export groups passage events by capture run. [Part 2](../Part2-RoadPF/) evaluates tracking using the measured detector-error profile; its default experiments generate their own trajectories and events.
