#!/bin/zsh
set -eu

ROOT_DIR="${0:A:h}"
OUTPUT_DIR="$ROOT_DIR/results/log_barrier/lunar_barrier/reliability"

cd "$ROOT_DIR"
mkdir -p "$OUTPUT_DIR"
exec >> "$OUTPUT_DIR/run.log" 2>&1
set -x
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

exec /usr/bin/caffeinate -dimsu \
  "$ROOT_DIR/.venv/bin/python" -u -m log_barrier.lunar_barrier.run reliability \
  --selection "$ROOT_DIR/results/log_barrier/lunar_barrier/smoke/selection.json" \
  --output "$OUTPUT_DIR" \
  --updates 1000 \
  --n-seeds 200 \
  --seed-start 10000 \
  --workers 10
