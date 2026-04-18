#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORLD_FILE="${1:-src/worlds/mazes/stage_1_open.sdf}"
TRAIN_CONFIG="${2:-stage_1_open_train.json}"

cd "$ROOT_DIR"

if [[ ! -f "$WORLD_FILE" ]]; then
  if [[ -f "src/worlds/mazes/$WORLD_FILE" ]]; then
    WORLD_FILE="src/worlds/mazes/$WORLD_FILE"
  else
    echo "[headless][error] World file not found: $WORLD_FILE"
    exit 1
  fi
fi

if [[ ! -f "$TRAIN_CONFIG" ]]; then
  if [[ -f "configs/$TRAIN_CONFIG" ]]; then
    TRAIN_CONFIG="configs/$TRAIN_CONFIG"
  else
    echo "[headless][error] Config file not found: $TRAIN_CONFIG"
    exit 1
  fi
fi

GZ_PID=""
BRIDGE_PID=""

cleanup() {
  set +e
  if [[ -n "$BRIDGE_PID" ]]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
  fi
  if [[ -n "$GZ_PID" ]]; then
    kill "$GZ_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT INT TERM

echo "[headless] Starting Gazebo server without GUI: $WORLD_FILE"
gz sim -s "$WORLD_FILE" > /tmp/gz_headless.log 2>&1 &
GZ_PID="$!"

sleep 3

echo "[headless] Starting ROS2 bridges"
bash src/scripts/setup_bridges.sh > /tmp/gz_bridges.log 2>&1 &
BRIDGE_PID="$!"

sleep 3

echo "[headless] Starting training with config: $TRAIN_CONFIG"
python3 train.py --config "$TRAIN_CONFIG"
