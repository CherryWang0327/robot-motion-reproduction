#!/usr/bin/env bash
# Train the two March references sequentially on one GPU:
#   1. finish the interrupted Pyroki run;
#   2. train the GMR run.
#
# The script is safe to restart. It resumes the newest model_*.pt in each
# run directory and only advances to the next job after train.py exits cleanly.

set -Eeuo pipefail

PROJECT_DIR=/home/unitree/projects/whole_body_tracking
PYTHON_BIN=/home/unitree/miniconda3/envs/whole_body_tracking/bin/python
TRAIN_SCRIPT="$PROJECT_DIR/scripts/rsl_rl/train.py"
LOG_ROOT="$PROJECT_DIR/logs/rsl_rl/g1_flat"
MAX_ITERATIONS="${MAX_ITERATIONS:-100000}"
RETRY_SECONDS="${RETRY_SECONDS:-60}"

PYROKI_MOTION_FILE="$PROJECT_DIR/inputs/march_video/proto_50fps/march_video_motionlib_pyroki_proto.npz"
PYROKI_RUN_DIR="$LOG_ROOT/2026-09-04_10-22-28_march_video_pyroki"

# The originally requested file is not currently present. Either put it at
# this path, or start the script with GMR_MOTION_FILE=/absolute/path/to/file.npz.
GMR_MOTION_FILE="${GMR_MOTION_FILE:-$PROJECT_DIR/inputs/march_video/march_video_gmr_50fps.npz}"
GMR_RUN_DIR="$LOG_ROOT/weekend_march_video_gmr"

mkdir -p "$LOG_ROOT" "$PYROKI_RUN_DIR" "$GMR_RUN_DIR"
cd "$PROJECT_DIR"

# Prevent two copies (manual launch plus systemd, for example) using the same GPU.
exec 9>"$LOG_ROOT/.weekend_march_training.lock"
flock -n 9 || { echo "Another March training scheduler is already running." >&2; exit 1; }

log() { printf '%s  %s\n' "$(date -Is)" "$*"; }

latest_checkpoint() {
    local run_dir="$1"
    find "$run_dir" -maxdepth 1 -type f -name 'model_*.pt' -printf '%f\n' 2>/dev/null \
        | sed -n 's/^model_\([0-9][0-9]*\)\.pt$/\1/p' \
        | sort -n \
        | tail -n 1
}

wait_for_gpu() {
    until timeout 10s nvidia-smi -L >/dev/null 2>&1; do
        log "CUDA driver is not ready; retrying in 30 seconds."
        sleep 30
    done
}

train_one() {
    local name="$1" motion_file="$2" run_dir="$3"
    local done_file="$run_dir/.training_complete" checkpoint

    if [[ -f "$done_file" ]]; then
        log "$name is already marked complete; skipping."
        return 0
    fi
    [[ -f "$motion_file" ]] || {
        log "ERROR: $name input is missing: $motion_file"
        return 2
    }

    while :; do
        wait_for_gpu
        checkpoint="$(latest_checkpoint "$run_dir" || true)"
        if [[ -n "$checkpoint" ]]; then
            log "Starting $name from model_${checkpoint}.pt toward total iteration $MAX_ITERATIONS."
            if "$PYTHON_BIN" "$TRAIN_SCRIPT" \
                --task Tracking-Flat-G1-v0 --motion_file "$motion_file" \
                --num_envs 4096 --max_iterations "$MAX_ITERATIONS" --seed 0 --headless \
                --logger tensorboard --run_name "$name" \
                --resume_checkpoint "$run_dir/model_${checkpoint}.pt" --resume_log_dir "$run_dir"; then
                break
            fi
        else
            log "Starting new $name run toward total iteration $MAX_ITERATIONS."
            if "$PYTHON_BIN" "$TRAIN_SCRIPT" \
                --task Tracking-Flat-G1-v0 --motion_file "$motion_file" \
                --num_envs 4096 --max_iterations "$MAX_ITERATIONS" --seed 0 --headless \
                --logger tensorboard --run_name "$name" --resume_log_dir "$run_dir"; then
                break
            fi
        fi
        log "$name stopped unexpectedly; retrying from its newest checkpoint in ${RETRY_SECONDS}s."
        sleep "$RETRY_SECONDS"
    done

    date -Is >"$done_file"
    log "$name completed successfully."
}

train_one march_video_pyroki "$PYROKI_MOTION_FILE" "$PYROKI_RUN_DIR"
train_one march_video_gmr "$GMR_MOTION_FILE" "$GMR_RUN_DIR"
log "Both March trainings completed."
