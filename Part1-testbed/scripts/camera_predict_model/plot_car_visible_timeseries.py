import pandas as pd
import matplotlib.pyplot as plt

STEM = "2026_07_28_16_07_15_126_camera_1_20260818_122811_434597926_rtp_marker_pred"
THR = 0.95
TRUTH_CSV = f"/media/ubuntu/research/carla_data_aug_infer_truth_no_warmup/{STEM}_truth.csv"
PRED_CSV = (
    f"/media/ubuntu/research/carla_data_aug_car_visible_pred_7/"
    f"{STEM}_car_visible_pred.csv"
)
X_TRUTH_CSV = f"/media/ubuntu/research/carla_data_aug_combined_x_infer_no_warmup/{STEM}.csv"

df_truth = pd.read_csv(TRUTH_CSV, usecols=["car_visible"])
df_pred = pd.read_csv(PRED_CSV, usecols=["frame_number_first", "car_visible_prob"])
df_x = pd.read_csv(X_TRUTH_CSV, usecols=["frame_number_first", "combined_frame_len_bytes"])

frame = df_pred["frame_number_first"]
truth_y = df_truth["car_visible"].astype(int)
pred_y = (df_pred["car_visible_prob"] > THR).astype(int)
frame_size = df_x["combined_frame_len_bytes"]

plt.figure(figsize=(16, 4))
plt.step(frame, truth_y, where="post", label="truth")
plt.step(frame, pred_y, where="post", label="pred")
plt.xlabel("frame_number_first")
plt.ylabel("car_visible")
plt.legend()
plt.title(f"car_visible (thr={THR})")
plt.ylim(-0.08, 1.08)

plt.figure(figsize=(16, 4))
plt.plot(df_x["frame_number_first"], frame_size, linewidth=0.8)
plt.xlabel("frame_number_first")
plt.ylabel("combined_frame_len_bytes")
plt.title("frame size (X ground truth)")

plt.show()
