#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT_DIR"

print_usage() {
  cat <<'EOF'
Usage:
  bash train_curriculum_p0.sh [--start-stage 1|2|3] [--resume-ckpt <path>] [stage1_config] [stage2_config] [stage3_config]

Examples:
  bash train_curriculum_p0.sh
  bash train_curriculum_p0.sh --start-stage 2
  bash train_curriculum_p0.sh --start-stage 2 --resume-ckpt artifacts/qmix/<stage1_run>/best_model.pt

Notes:
  - --start-stage 2: resumes Stage2 from Stage1 checkpoint
    (auto-detected from stage1 config when --resume-ckpt is omitted).
  - --start-stage 3: resumes Stage3 from Stage2 checkpoint
    (auto-detected from stage2 config when --resume-ckpt is omitted).
EOF
}

START_STAGE="1"
RESUME_CKPT=""
POSITIONAL_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --start-stage)
      START_STAGE="${2:-}"
      shift 2
      ;;
    --resume-ckpt)
      RESUME_CKPT="${2:-}"
      shift 2
      ;;
    -h|--help)
      print_usage
      exit 0
      ;;
    *)
      POSITIONAL_ARGS+=("$1")
      shift
      ;;
  esac
done
set -- "${POSITIONAL_ARGS[@]}"

# P0 curriculum defaults:
#   Stage 1: stage_1_open (bootstrap)
#   Stage 2: stage_2_easy (obstacle transition)
#   Stage 3: maze_1 (target scene)
STAGE1_CONFIG="${1:-configs/p0_stage1_open.json}"
STAGE2_CONFIG="${2:-configs/p0_stage2_easy.json}"
STAGE3_CONFIG="${3:-configs/p0_stage3_maze1.json}"

if [[ "$START_STAGE" != "1" && "$START_STAGE" != "2" && "$START_STAGE" != "3" ]]; then
  echo "[P0][error] --start-stage must be one of: 1, 2, 3 (got: $START_STAGE)" >&2
  print_usage
  exit 1
fi

LOG_DIR="${LOG_DIR:-/tmp/qmix_curriculum}"
mkdir -p "$LOG_DIR"

GZ_PID=""
BRIDGE_PID=""

cleanup_runtime() {
  set +e
  if [[ -n "$BRIDGE_PID" ]]; then
    kill "$BRIDGE_PID" 2>/dev/null || true
    wait "$BRIDGE_PID" 2>/dev/null || true
    BRIDGE_PID=""
  fi
  if [[ -n "$GZ_PID" ]]; then
    kill "$GZ_PID" 2>/dev/null || true
    wait "$GZ_PID" 2>/dev/null || true
    GZ_PID=""
  fi
}

cleanup_all() {
  cleanup_runtime
}
trap cleanup_all EXIT INT TERM

require_file() {
  local path="$1"
  if [[ ! -f "$path" ]]; then
    echo "[P0][error] File not found: $path" >&2
    exit 1
  fi
}

config_get() {
  local cfg="$1"
  local key="$2"
  python3 - <<'PY' "$cfg" "$key"
import json, sys
with open(sys.argv[1], "r", encoding="utf-8") as f:
    data = json.load(f)
print(data.get(sys.argv[2], ""))
PY
}

find_latest_run_dir() {
  local output_root="$1"
  local run_base="$2"
  if [[ ! -d "$output_root" ]]; then
    return 1
  fi
  local latest
  latest="$(find "$output_root" -maxdepth 1 -mindepth 1 -type d -name "${run_base}*" -printf "%T@ %p\n" | sort -nr | head -n 1 | awk '{print $2}')"
  if [[ -z "$latest" ]]; then
    return 1
  fi
  printf "%s\n" "$latest"
}

resolve_resume_checkpoint_from_config() {
  local cfg_path="$1"
  require_file "$cfg_path"

  local output_root
  output_root="$(config_get "$cfg_path" "output_root")"
  local run_name
  run_name="$(config_get "$cfg_path" "run_name")"

  if [[ -z "$output_root" || -z "$run_name" ]]; then
    echo "[P0][error] Cannot resolve checkpoint from config (missing output_root/run_name): $cfg_path" >&2
    exit 1
  fi

  local latest_run_dir
  latest_run_dir="$(find_latest_run_dir "$output_root" "$run_name")" || {
    echo "[P0][error] Cannot find latest run dir for '$run_name' under '$output_root'" >&2
    exit 1
  }

  local best_model="$latest_run_dir/best_model.pt"
  local latest_model="$latest_run_dir/latest_model.pt"
  if [[ -f "$best_model" ]]; then
    printf "%s\n" "$best_model"
    return 0
  fi
  if [[ -f "$latest_model" ]]; then
    printf "%s\n" "$latest_model"
    return 0
  fi

  echo "[P0][error] No checkpoint found in latest run dir: $latest_run_dir" >&2
  exit 1
}

check_stage_summary() {
  local latest_run_dir="$1"
  local stage_name="$2"
  local summary_path="$latest_run_dir/summary.json"

  if [[ ! -f "$summary_path" ]]; then
    echo "[P0][$stage_name][warn] summary.json not found at $summary_path; skip interruption check." >&2
    return 0
  fi

  local status
  status="$(python3 - <<'PY' "$summary_path"
import json, sys
path = sys.argv[1]
with open(path, "r", encoding="utf-8") as f:
    s = json.load(f)
interrupted = bool(s.get("interrupted", False))
interrupt_type = s.get("interrupt_type")
if interrupted:
    print(f"FAIL:{interrupt_type}")
else:
    print("OK")
PY
)"

  if [[ "$status" == FAIL:* ]]; then
    echo "[P0][$stage_name][error] Training interrupted according to summary: ${status#FAIL:}" >&2
    echo "[P0][$stage_name][error] Summary: $summary_path" >&2
    exit 1
  fi
}

start_world_and_bridges() {
  local world_file="$1"
  local stage_name="$2"

  cleanup_runtime

  echo "[P0][$stage_name] Starting Gazebo headless world: $world_file" >&2
  gz sim -s "$world_file" > "$LOG_DIR/${stage_name}_gazebo.log" 2>&1 &
  GZ_PID="$!"

  sleep 4

  echo "[P0][$stage_name] Starting ROS2 bridges" >&2
  bash src/scripts/setup_bridges.sh > "$LOG_DIR/${stage_name}_bridges.log" 2>&1 &
  BRIDGE_PID="$!"

  sleep 4

  if ! kill -0 "$GZ_PID" 2>/dev/null; then
    echo "[P0][error] Gazebo exited unexpectedly for $stage_name. See $LOG_DIR/${stage_name}_gazebo.log" >&2
    exit 1
  fi
  if ! kill -0 "$BRIDGE_PID" 2>/dev/null; then
    echo "[P0][error] Bridge script exited unexpectedly for $stage_name. See $LOG_DIR/${stage_name}_bridges.log" >&2
    exit 1
  fi
}

run_stage() {
  local stage_name="$1"
  local config_path="$2"
  local resume_path="${3:-}"

  require_file "$config_path"

  local world_name
  world_name="$(config_get "$config_path" "world_name")"
  local output_root
  output_root="$(config_get "$config_path" "output_root")"
  local run_name
  run_name="$(config_get "$config_path" "run_name")"

  if [[ -z "$world_name" || -z "$output_root" || -z "$run_name" ]]; then
    echo "[P0][error] Config missing one of required keys: world_name, output_root, run_name -> $config_path" >&2
    exit 1
  fi

  local world_path="src/worlds/mazes/$world_name"
  require_file "$world_path"

  start_world_and_bridges "$world_path" "$stage_name"

  echo "[P0][$stage_name] Training with config: $config_path" >&2
  if [[ -n "$resume_path" ]]; then
    require_file "$resume_path"
    echo "[P0][$stage_name] Resume checkpoint: $resume_path" >&2
    python3 train.py --config "$config_path" --resume "$resume_path"
  else
    python3 train.py --config "$config_path"
  fi

  local latest_run_dir
  latest_run_dir="$(find_latest_run_dir "$output_root" "$run_name")" || {
    echo "[P0][error] Cannot locate run directory for base '$run_name' in '$output_root'" >&2
    exit 1
  }

  check_stage_summary "$latest_run_dir" "$stage_name"

  local best_model="$latest_run_dir/best_model.pt"
  local latest_model="$latest_run_dir/latest_model.pt"

  if [[ -f "$best_model" ]]; then
    STAGE_CKPT="$best_model"
    return 0
  fi

  if [[ -f "$latest_model" ]]; then
    STAGE_CKPT="$latest_model"
    return 0
  fi

  echo "[P0][error] No best_model.pt or latest_model.pt found in: $latest_run_dir" >&2
  exit 1
}

require_file "$STAGE1_CONFIG"
require_file "$STAGE2_CONFIG"
require_file "$STAGE3_CONFIG"

echo "[P0] Curriculum configs:"
echo "  Stage1: $STAGE1_CONFIG"
echo "  Stage2: $STAGE2_CONFIG"
echo "  Stage3: $STAGE3_CONFIG"
echo "  Start stage: $START_STAGE"
if [[ -n "$RESUME_CKPT" ]]; then
  echo "  Resume ckpt: $RESUME_CKPT"
fi
echo "  Logs:   $LOG_DIR"

STAGE_CKPT=""

if [[ "$START_STAGE" == "1" ]]; then
  echo "[P0] ===== Stage 1 ====="
  run_stage "stage1" "$STAGE1_CONFIG"
  STAGE1_CKPT="$STAGE_CKPT"
  echo "[P0] Stage 1 checkpoint: $STAGE1_CKPT"

  echo "[P0] ===== Stage 2 ====="
  run_stage "stage2" "$STAGE2_CONFIG" "$STAGE1_CKPT"
  STAGE2_CKPT="$STAGE_CKPT"
  echo "[P0] Stage 2 checkpoint: $STAGE2_CKPT"

  echo "[P0] ===== Stage 3 ====="
  run_stage "stage3" "$STAGE3_CONFIG" "$STAGE2_CKPT"
  STAGE3_CKPT="$STAGE_CKPT"
  echo "[P0] Stage 3 checkpoint: $STAGE3_CKPT"
elif [[ "$START_STAGE" == "2" ]]; then
  if [[ -z "$RESUME_CKPT" ]]; then
    RESUME_CKPT="$(resolve_resume_checkpoint_from_config "$STAGE1_CONFIG")"
  fi
  require_file "$RESUME_CKPT"
  echo "[P0] Starting from Stage 2 using checkpoint: $RESUME_CKPT"

  echo "[P0] ===== Stage 2 ====="
  run_stage "stage2" "$STAGE2_CONFIG" "$RESUME_CKPT"
  STAGE2_CKPT="$STAGE_CKPT"
  echo "[P0] Stage 2 checkpoint: $STAGE2_CKPT"

  echo "[P0] ===== Stage 3 ====="
  run_stage "stage3" "$STAGE3_CONFIG" "$STAGE2_CKPT"
  STAGE3_CKPT="$STAGE_CKPT"
  echo "[P0] Stage 3 checkpoint: $STAGE3_CKPT"
else
  if [[ -z "$RESUME_CKPT" ]]; then
    RESUME_CKPT="$(resolve_resume_checkpoint_from_config "$STAGE2_CONFIG")"
  fi
  require_file "$RESUME_CKPT"
  echo "[P0] Starting from Stage 3 using checkpoint: $RESUME_CKPT"

  echo "[P0] ===== Stage 3 ====="
  run_stage "stage3" "$STAGE3_CONFIG" "$RESUME_CKPT"
  STAGE3_CKPT="$STAGE_CKPT"
  echo "[P0] Stage 3 checkpoint: $STAGE3_CKPT"
fi

echo "[P0] Curriculum training completed."
