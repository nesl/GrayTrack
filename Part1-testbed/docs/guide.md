# Testbed reproduction guide

Run the commands below from `Part1-testbed/`. Local paths are examples from the original setup.

Research pipeline: CARLA multi-camera capture, Mininet-WiFi encrypted replay, PacketSegformer RTP-marker inference, packet→video-frame aggregation, CarVisibleLSTM vehicle visibility, then passage-event export for particle filtering.

Paths are hardcoded under `/media/ubuntu/research/...` and `/home/ubuntu/carla`. Edit script headers before running on another machine.

---

## Prerequisites

CARLA and matching Python API; ffmpeg/ffprobe; Mininet-WiFi; tshark, mergecap, capinfos, tcpdump.

```bash
pip install -r requirements.txt
```

Install the CARLA Python API to match your simulator version, and install Mininet-WiFi (`mn_wifi`) separately.

---

## End-to-end reproduction

All capture and PCAP work uses `/home/ubuntu/carla`.

| Step | Stage | Command |
|------|--------|---------|
| 1 | CARLA capture (MP4 + video-frame truth) | `bash scripts/run_map_car_cam_loop.sh` (paths edited in `scripts/map_car_cam.py`) |
| 2 | Re-encode fixed GOP | `bash scripts/reencode_carla_videos.sh /home/ubuntu/carla` |
| 3 | Mininet-WiFi replay and PCAP capture | `sudo python3 scripts/mininet/two_stations_wifi.py /home/ubuntu/carla` |
| 4 | Merge handshake into decrypted PCAPs | `bash scripts/mininet/merge_and_decrypt_pcap_all_clean.sh /home/ubuntu/carla` |
| 5 | Build network-packet x/y CSVs | `python3 scripts/mininet/pcap_to_dataset.py /home/ubuntu/carla` |
| 6 | Combine X+Y into wide CSVs | `bash scripts/mininet/combine_xy_to_tmp.sh /home/ubuntu/carla` |
| 7 | Train PacketSegformer | `python3 scripts/mininet/frame_boundary_model/frame_boundary_model.py` (paths edited in file) |
| 8 | Batch infer RTP markers | `python3 scripts/mininet/frame_boundary_model/batch_infer_rtp_marker.py` (paths edited in file) |
| 9 | Copy video-frame truth CSVs for infer | `bash scripts/mininet/copy_truth_csv_for_xy_infer.sh` (paths edited in file) |
| 10 | Aggregate network packets to video frames | `python3 scripts/camera_predict_model/preprocess_stage_1.py` (paths edited in file) |
| 11 | Drop warmup video frames | `python3 scripts/camera_predict_model/preprocess_stage_2_warmup_interpolate.py` (paths edited in file) |
| 12 | Train CarVisibleLSTM | `python3 scripts/camera_predict_model/car_visible_lstm_model.py` (paths edited in file) |
| 13 | Materialize held-out test runs | `python3 scripts/camera_predict_model/extract_car_visible_test_split.py` (paths edited in file) |
| 14 | Infer car_visible on test split | `python3 scripts/camera_predict_model/car_visible_lstm_infer.py --x-dir /media/ubuntu/research/carla_x_infer_test --output /media/ubuntu/research/carla_car_visible_pred_test` |
| 15 | Event-level evaluation (writes `camera_hyperparams.csv`) | `python3 scripts/camera_predict_model/eval_car_visible_events.py --pred-dir /media/ubuntu/research/carla_car_visible_pred_test --truth-dir /media/ubuntu/research/carla_truth_test --output-dir /media/ubuntu/research/carla_event_eval` |
| 16 | Export passage events (`x/` + `y_truth/`) for the next stage in the pipeline | `python3 scripts/camera_predict_model/export_passage_events.py --pred-dir /media/ubuntu/research/carla_car_visible_pred_test --truth-dir /media/ubuntu/research/carla_truth_test --camera-hyperparams /media/ubuntu/research/carla_event_eval/camera_hyperparams.csv --output-dir /media/ubuntu/research/carla_passage_events` |
| 17 | Group export by capture run (one CARLA session; all cameras sharing the stem before `_camera_`) | `bash scripts/camera_predict_model/group_passage_events_by_run.sh /media/ubuntu/research/carla_passage_events` |

Step 17 output (`/media/ubuntu/research/carla_passage_events`) is the dataset handed to the next stage in the pipeline.

### Helper scripts used for organization, baselines, and reporting

| Script | Role |
|--------|------|
| `scripts/mininet/frame_boundary_model/infer_rtp_marker_from_csv.py` | Single-CSV PacketSegformer check |
| `scripts/mininet/frame_boundary_model/eval_rtp_marker_pred_accuracy.py` | RTP-marker accuracy |
| `scripts/mininet/frame_boundary_model/eval_boundary_timing_error.py` | Boundary timing error (packet index) |
| `scripts/mininet/frame_boundary_model/eval_frame_size_error.py` | Video-frame size error vs truth |
| `scripts/mininet/frame_boundary_model/fixed_period_packet_segmentation.py` | 1/FPS packet-grouping baseline |
| `scripts/camera_predict_model/sum_pred_marker.py` | Count predicted markers |
| `scripts/camera_predict_model/compare_xy_combined_pred_to_truth.py` | Pred vs truth overlap |
| `scripts/camera_predict_model/generate_xy_combined_x_truth.py` | Normalize CSV column layout |
| `scripts/camera_predict_model/car_visible_threshold_infer.py` | Non-ML threshold baseline |
| `scripts/camera_predict_model/plot_car_visible_event_diagnostics.py` | Event diagnostic plots |
| `scripts/camera_predict_model/run_car_visible_lstm_experiments.sh` | LSTM hyperparam sweep |
| `scripts/mininet/dev/merge_and_decrypt_single_pcap.sh` | Single-PCAP decrypt |
| `scripts/mininet/dev/clean_mininet.sh` | Clean Mininet state |
| `scripts/mininet/dev/fix_existing_decrypted_pcaps.sh` | Repair decrypted PCAPs |

