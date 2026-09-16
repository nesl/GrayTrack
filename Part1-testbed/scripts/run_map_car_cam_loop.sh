#!/usr/bin/env bash
# Run map_car_cam.py in a loop with configurable delay and incrementing seed.

DELAY_SEC=5
NUM_ITERATIONS=1000
START_SEED=124

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

for ((i = 0; i < NUM_ITERATIONS; i++)); do
  seed=$((START_SEED + i))
  echo "=== Iteration $((i + 1))/$NUM_ITERATIONS (seed=$seed) ==="
  python3 "$SCRIPT_DIR/map_car_cam.py" --seed "$seed" "$@"
  if [[ $i -lt $((NUM_ITERATIONS - 1)) ]]; then
    echo "Sleeping ${DELAY_SEC}s..."
    sleep "$DELAY_SEC"
  fi
done

echo "Done. Ran $NUM_ITERATIONS iterations."
