#!/usr/bin/env bash
# Example: train with window/stride and stop at epoch 25 (saves checkpoint, then exits):
#   run_exp "exp_ws512_s1_e25" --window 512 --stride 1 --stop-at-epoch 25
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="${SCRIPT_DIR}/car_visible_lstm_model.py"

run_exp() {
  local name="$1"
  shift || true
  local log_path="${SCRIPT_DIR}/results_7/${name}.txt"
  mkdir -p "${SCRIPT_DIR}/results_7"
  echo "Running ${name}..."
  python3 "${TRAIN_SCRIPT}" "$@" 2>&1 | tee "${log_path}"
  echo "Finished ${name} -> ${log_path}"
}

# run_exp "exp_ws768_s1" --window 768 --stride 1
# run_exp "exp_ws768_s2" --window 768 --stride 2
# run_exp "exp_ws768_s4" --window 768 --stride 4
# run_exp "exp_ws768_s8" --window 768 --stride 8
# run_exp "exp_ws768_s16" --window 768 --stride 16

# run_exp "exp_ws512_s32" --window 512 --stride 32
# run_exp "exp_ws256_s32" --window 256 --stride 32
# run_exp "exp_ws768_s32" --window 768 --stride 32

# run_exp "exp_ws512_s64" --window 512 --stride 64
# run_exp "exp_ws256_s64" --window 256 --stride 64
# run_exp "exp_ws768_s64" --window 768 --stride 64

# run_exp "exp_ws512_s128" --window 512 --stride 128
# run_exp "exp_ws256_s128" --window 256 --stride 128
# run_exp "exp_ws768_s128" --window 768 --stride 128

# Baseline: default hyperparameters (no CLI args).
# run_exp "baseline_default"

# Window/stride sweeps (new warmup value).
# First chunk: stride=1 sweep from largest window size to smallest window size.
# run_exp "exp_ws768_s1_warm256" --window 768 --stride 1
run_exp "exp_ws512_s1_warm256" --window 512 --stride 1 # BEST ONE SO FAR
run_exp "exp_ws256_s1_warm256" --window 256 --stride 1
run_exp "exp_ws128_s1_warm256" --window 128 --stride 1
run_exp "exp_ws64_s1_warm256" --window 64 --stride 1
run_exp "exp_ws32_s1_warm256" --window 32 --stride 1
run_exp "exp_ws16_s1_warm256" --window 16 --stride 1

# Second chunk: a few variants with stride=2 and stride=4 (after first chunk).
# run_exp "exp_ws768_s2_warm256" --window 768 --stride 2
run_exp "exp_ws512_s2_warm256" --window 512 --stride 2
run_exp "exp_ws256_s2_warm256" --window 256 --stride 2
run_exp "exp_ws512_s4_warm256" --window 512 --stride 4
run_exp "exp_ws256_s4_warm256" --window 256 --stride 4

# run_exp "exp_ws16_s8" --window 16 --stride 8
# run_exp "exp_ws32_s8" --window 32 --stride 8
# run_exp "exp_ws64_s8" --window 64 --stride 8
# run_exp "exp_ws128_s8" --window 128 --stride 8
# run_exp "exp_ws256_s8" --window 256 --stride 8
# run_exp "exp_ws512_s8" --window 512 --stride 8

# run_exp "exp_ws16_s16" --window 16 --stride 16
# run_exp "exp_ws32_s16" --window 32 --stride 16
# run_exp "exp_ws64_s16" --window 64 --stride 16
# run_exp "exp_ws128_s16" --window 128 --stride 16
# run_exp "exp_ws256_s16" --window 256 --stride 16
# run_exp "exp_ws512_s16" --window 512 --stride 16

# Additional architecture experiments.
# run_exp "exp1_ws32_s4_h64_l2" --window 32 --stride 4 --hidden 64 --layers 2
# run_exp "exp2_ws32_s4_h128_l2_d02" --window 32 --stride 4 --hidden 128 --layers 2 --dropout 0.2
# run_exp "exp3_ws64_s8_h128_l2_d02" --window 64 --stride 8 --hidden 128 --layers 2 --dropout 0.2
# run_exp "exp4_ws32_s4_h64_l2_bi" --window 32 --stride 4 --hidden 64 --layers 2 --bidirectional
# run_exp "exp5_ws16_s2_h64_l2" --window 16 --stride 2 --hidden 64 --layers 2
# run_exp "exp6_ws16_s2_h128_l2_d02" --window 16 --stride 2 --hidden 128 --layers 2 --dropout 0.2
# run_exp "exp7_ws32_s2_h128_l2_bi" --window 32 --stride 2 --hidden 128 --layers 2 --bidirectional
# run_exp "exp8_ws64_s4_h128_l2_d02" --window 64 --stride 4 --hidden 128 --layers 2 --dropout 0.2
# run_exp "exp9_ws16_s4_h64_l3_d02" --window 16 --stride 4 --hidden 64 --layers 3 --dropout 0.2
# run_exp "exp10_ws64_s4_h64_l2_bi" --window 64 --stride 4 --hidden 64 --layers 2 --bidirectional

echo "All experiments completed."
