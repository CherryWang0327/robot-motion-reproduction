#!/bin/bash
# SPDX-FileCopyrightText: Copyright (c) 2025 The ProtoMotions Developers
# SPDX-License-Identifier: Apache-2.0
#
# Batch retarget packaged AMASS / ProtoMotions MotionLib .pt files to robot motions
# while preserving input directory structure.
#
# Usage:
#   ./scripts/retarget_amass_to_robot.sh \
#       <proto_python> <pyroki_python> <input_root_dir> <output_root_dir> <robot_type> \
#       [skip_freq] [apply_motion_filter]
#
# The input frame rate is read from each packaged MotionLib's `motion_dt`.
# PyRoki keeps one output frame per input frame, so Step 4 must use that same
# rate; assigning a different rate would change the motion's playback speed.
#
# Example 1: 默认关闭 motion filter
#   ./scripts/retarget_amass_to_robot.sh \
#       ~/miniconda3/envs/protomotion/bin/python \
#       ~/miniconda3/envs/jax311/bin/python \
#       /data/BysanRL/data_goal/proto_motion_lafan1_pt/fall \
#       /data/BysanRL/data_goal/lafan1_pyroki/fall \
#       g1 \
#       1
#
# Example 2: 显式开启 motion filter
#   ./scripts/retarget_amass_to_robot.sh \
#       ~/miniconda3/envs/protomotion/bin/python \
#       ~/miniconda3/envs/jax311/bin/python \
#       /data/BysanRL/data_goal/proto_motion_lafan1_pt/fall \
#       /data/BysanRL/data_goal/lafan1_pyroki/fall \
#       g1 \
#       1 \
#       1

set -euo pipefail

if [ $# -lt 5 ]; then
    echo "Usage: $0 <proto_python> <pyroki_python> <input_root_dir> <output_root_dir> <robot_type> [skip_freq] [apply_motion_filter]"
    echo ""
    echo "Arguments:"
    echo "  proto_python         Path to Python interpreter with ProtoMotions installed"
    echo "  pyroki_python        Path to Python interpreter with PyRoki/JAX installed"
    echo "  input_root_dir       Root directory containing packaged MotionLib .pt files"
    echo "  output_root_dir      Root directory to save final *_pyroki.pt files"
    echo "  robot_type           Target robot: 'g1' or 'h1_2'"
    echo "  skip_freq            Optional. Skip every N motions inside each input pt (default: 1)"
    echo "  apply_motion_filter  Optional. 0=disable (default), 1=enable"
    exit 1
fi

PROTO_PYTHON="$1"
PYROKI_PYTHON="$2"
INPUT_ROOT_DIR="$3"
OUTPUT_ROOT_DIR="$4"
ROBOT_TYPE="$5"
SKIP_FREQ="${6:-1}"
APPLY_MOTION_FILTER="${7:-0}"

if [ "$ROBOT_TYPE" != "g1" ] && [ "$ROBOT_TYPE" != "h1_2" ]; then
    echo "Error: robot_type must be 'g1' or 'h1_2'"
    exit 1
fi

if [ "$APPLY_MOTION_FILTER" != "0" ] && [ "$APPLY_MOTION_FILTER" != "1" ]; then
    echo "Error: apply_motion_filter must be 0 or 1"
    exit 1
fi

if [ ! -f "$PROTO_PYTHON" ]; then
    echo "Error: ProtoMotions Python not found: $PROTO_PYTHON"
    exit 1
fi

if [ ! -f "$PYROKI_PYTHON" ]; then
    echo "Error: PyRoki Python not found: $PYROKI_PYTHON"
    exit 1
fi

if [ ! -d "$INPUT_ROOT_DIR" ]; then
    echo "Error: input_root_dir not found or not a directory: $INPUT_ROOT_DIR"
    exit 1
fi

mkdir -p "$OUTPUT_ROOT_DIR"

FAILED_LOG="${OUTPUT_ROOT_DIR}/failed_files.log"
# ======== newADD start======
# 每次运行都重新生成失败日志，避免历史记录混在一起
: > "$FAILED_LOG"
# =========== newADD end ========

echo "=============================================="
echo "Batch retargeting AMASS to ${ROBOT_TYPE^^}"
echo "=============================================="
echo "ProtoMotions Python: $PROTO_PYTHON"
echo "PyRoki Python:       $PYROKI_PYTHON"
echo "Input root:          $INPUT_ROOT_DIR"
echo "Output root:         $OUTPUT_ROOT_DIR"
echo "Robot type:          $ROBOT_TYPE"
echo "Skip freq:           $SKIP_FREQ"
echo "Apply motion filter: $APPLY_MOTION_FILTER"
echo "Failed log:          $FAILED_LOG"
echo "=============================================="

process_one_pt_file() {
    local AMASS_PT_FILE="$1"

    if [ ! -f "$AMASS_PT_FILE" ]; then
        echo "Warning: input file not found, skipping: $AMASS_PT_FILE"
        echo "$AMASS_PT_FILE | missing input file" >> "$FAILED_LOG"
        return 1
    fi

    # 计算相对于输入根目录的路径
    local REL_PATH
    REL_PATH=$(python - "$INPUT_ROOT_DIR" "$AMASS_PT_FILE" <<'PY'
import sys
from pathlib import Path
input_root = Path(sys.argv[1]).resolve()
file_path = Path(sys.argv[2]).resolve()
print(file_path.relative_to(input_root))
PY
)

    local REL_DIR
    REL_DIR=$(dirname "$REL_PATH")

    local INPUT_BASENAME
    INPUT_BASENAME=$(basename "$AMASS_PT_FILE" .pt)

    # 最终输出：保持目录结构，同名改为 *_pyroki.pt
    local FINAL_DIR="${OUTPUT_ROOT_DIR}/${REL_DIR}"
    local FINAL_PT="${FINAL_DIR}/${INPUT_BASENAME}_pyroki.pt"

    # ======== newADD start======
    if [ -f "$FINAL_PT" ]; then
        echo "Skipping already finished file: $FINAL_PT"
        return 0
    fi
    # =========== newADD end ========

    # 中间产物工作目录
    local WORK_DIR="${OUTPUT_ROOT_DIR}/.work/${REL_DIR}/${INPUT_BASENAME}"
    local KEYPOINTS_DIR="${WORK_DIR}/keypoints"
    local RETARGETED_DIR="${WORK_DIR}/retargeted_${ROBOT_TYPE}"
    local CONTACTS_DIR="${WORK_DIR}/contacts"
    local PROTO_DIR="${WORK_DIR}/retargeted_${ROBOT_TYPE}_proto"

    mkdir -p "$FINAL_DIR"
    mkdir -p "$WORK_DIR"

    echo ""
    echo "----------------------------------------------"
    echo "Processing one file"
    echo "----------------------------------------------"
    echo "Input file : $AMASS_PT_FILE"
    echo "Work dir   : $WORK_DIR"
    echo "Final pt   : $FINAL_PT"
    echo "----------------------------------------------"

    # A packaged MotionLib carries the authoritative sampling period.  Do not
    # assume 50 FPS here: GVHMR input videos are commonly 30 FPS and PyRoki
    # does not synthesize intermediate frames during retargeting.
    local MOTION_FPS
    MOTION_FPS=$("$PROTO_PYTHON" - "$AMASS_PT_FILE" <<'PY'
import sys
import torch

data = torch.load(sys.argv[1], map_location="cpu", weights_only=False)
motion_dt = float(data["motion_dt"].flatten()[0])
if motion_dt <= 0:
    raise ValueError(f"invalid motion_dt: {motion_dt}")
fps = round(1.0 / motion_dt)
if abs((1.0 / motion_dt) - fps) > 1e-3:
    raise ValueError(f"non-integral MotionLib FPS: {1.0 / motion_dt}")
print(fps)
PY
)
    echo "Detected MotionLib FPS: $MOTION_FPS"

    # Step 1
    echo ""
    echo "[Step 1/5] Extracting keypoints from packaged MotionLib..."
    "$PROTO_PYTHON" data/scripts/extract_retargeting_input_keypoints_from_packaged_motionlib.py \
        "$AMASS_PT_FILE" \
        --output-path "$KEYPOINTS_DIR" \
        --skeleton-format smpl \
        --start-idx 0 \
        --skip-freq "$SKIP_FREQ" || {
            echo "Error: Step 1 failed for $AMASS_PT_FILE"
            echo "$AMASS_PT_FILE | failed at Step 1: extract keypoints" >> "$FAILED_LOG"
            return 1
        }

    # Step 2
    echo ""
    echo "[Step 2/5] Running PyRoki retargeting to ${ROBOT_TYPE^^}..."
    if [ "$ROBOT_TYPE" == "g1" ]; then
        "$PYROKI_PYTHON" pyroki/batch_retarget_to_g1_from_keypoints.py \
            --subsample-factor 1 \
            --target-raw-frames 20000 \
            --chunk-len 1000 \
            --chunk-overlap 0 \
            --keypoints-folder-path "$KEYPOINTS_DIR" \
            --source-type smpl \
            --output-dir "$RETARGETED_DIR" \
            --no-visualize \
            --skip-existing || {
                echo "Error: Step 2 failed for $AMASS_PT_FILE"
                echo "$AMASS_PT_FILE | failed at Step 2: pyroki retarget" >> "$FAILED_LOG"
                return 1
            }
    else
        "$PYROKI_PYTHON" pyroki/batch_retarget_to_h1_2_from_keypoints.py \
            --subsample-factor 1 \
            --target-raw-frames 20000 \
            --chunk-len 1000 \
            --chunk-overlap 0 \
            --keypoints-folder-path "$KEYPOINTS_DIR" \
            --source-type smpl \
            --output-dir "$RETARGETED_DIR" \
            --no-visualize \
            --skip-existing || {
                echo "Error: Step 2 failed for $AMASS_PT_FILE"
                echo "$AMASS_PT_FILE | failed at Step 2: pyroki retarget" >> "$FAILED_LOG"
                return 1
            }
    fi

    # Step 2 产物检查
    local RETARGETED_COUNT
    RETARGETED_COUNT=$(find "$RETARGETED_DIR" -maxdepth 1 -type f -name "*_retargeted.npz" | wc -l)

    if [ "$RETARGETED_COUNT" -eq 0 ]; then
        echo "Error: Step 2 produced no retargeted .npz files in: $RETARGETED_DIR"
        echo "$AMASS_PT_FILE | failed at Step 2: no *_retargeted.npz produced" >> "$FAILED_LOG"
        return 1
    fi

    # Step 3
    echo ""
    echo "[Step 3/5] Extracting foot contact labels from source SMPL motions..."
    if [ "$ROBOT_TYPE" == "g1" ]; then
        "$PYROKI_PYTHON" pyroki/batch_retarget_to_g1_from_keypoints.py \
            --subsample-factor 1 \
            --target-raw-frames 20000 \
            --chunk-len 1000 \
            --chunk-overlap 0 \
            --keypoints-folder-path "$KEYPOINTS_DIR" \
            --source-type smpl \
            --save-contacts-only \
            --contacts-dir "$CONTACTS_DIR" \
            --skip-existing || {
                echo "Error: Step 3 failed for $AMASS_PT_FILE"
                echo "$AMASS_PT_FILE | failed at Step 3: save contacts" >> "$FAILED_LOG"
                return 1
            }
    else
        "$PYROKI_PYTHON" pyroki/batch_retarget_to_h1_2_from_keypoints.py \
            --subsample-factor 1 \
            --target-raw-frames 20000 \
            --chunk-len 1000 \
            --chunk-overlap 0 \
            --keypoints-folder-path "$KEYPOINTS_DIR" \
            --source-type smpl \
            --save-contacts-only \
            --contacts-dir "$CONTACTS_DIR" \
            --skip-existing || {
                echo "Error: Step 3 failed for $AMASS_PT_FILE"
                echo "$AMASS_PT_FILE | failed at Step 3: save contacts" >> "$FAILED_LOG"
                return 1
            }
    fi

    # Step 4
    echo ""
    echo "[Step 4/5] Converting to ProtoMotions format..."

    # ======== newADD start======
    local STEP4_ARGS=(
        data/scripts/convert_pyroki_retargeted_robot_motions_to_proto.py
        --retargeted-motion-dir "$RETARGETED_DIR"
        --output-dir "$PROTO_DIR"
        --robot-type "$ROBOT_TYPE"
        --contact-labels-dir "$CONTACTS_DIR"
        --input-fps "$MOTION_FPS"
        --output-fps "$MOTION_FPS"
        --force-remake
    )

    if [ "$APPLY_MOTION_FILTER" == "1" ]; then
        STEP4_ARGS+=(--apply-motion-filter)
        echo "[INFO] Step 4 motion filter: ENABLED"
    else
        echo "[INFO] Step 4 motion filter: DISABLED"
    fi

    "$PROTO_PYTHON" "${STEP4_ARGS[@]}" || {
        echo "Error: Step 4 failed for $AMASS_PT_FILE"
        echo "$AMASS_PT_FILE | failed at Step 4: convert to proto" >> "$FAILED_LOG"
        return 1
    }
    # =========== newADD end ========

    # Step 4 产物检查
    local PROTO_MOTION_COUNT
    PROTO_MOTION_COUNT=$(find "$PROTO_DIR" -maxdepth 1 -type f -name "*.motion" | wc -l)

    if [ "$PROTO_MOTION_COUNT" -eq 0 ]; then
        echo "Error: Step 4 produced no .motion files in: $PROTO_DIR"
        echo "$AMASS_PT_FILE | failed at Step 4: no .motion produced" >> "$FAILED_LOG"
        return 1
    fi

    # Step 5
    echo ""
    echo "[Step 5/5] Packaging into MotionLib..."
    "$PROTO_PYTHON" protomotions/components/motion_lib.py \
        --motion-path "$PROTO_DIR" \
        --output-file "$FINAL_PT" || {
            echo "Error: Step 5 failed for $AMASS_PT_FILE"
            echo "$AMASS_PT_FILE | failed at Step 5: package motion lib" >> "$FAILED_LOG"
            return 1
        }

    if [ ! -f "$FINAL_PT" ]; then
        echo "Error: Final output was not created: $FINAL_PT"
        echo "$AMASS_PT_FILE | failed at Step 5: final pt missing" >> "$FAILED_LOG"
        return 1
    fi

    echo ""
    echo "Finished: $FINAL_PT"
    return 0
}

PT_FILES=()
while IFS= read -r -d '' file; do
    PT_FILES+=("$file")
done < <(find "$INPUT_ROOT_DIR" -type f -name "*.pt" -print0 | sort -z)

TOTAL_FILES=${#PT_FILES[@]}

if [ "$TOTAL_FILES" -eq 0 ]; then
    echo "No .pt files found under: $INPUT_ROOT_DIR"
    exit 0
fi

echo "Found $TOTAL_FILES .pt files to process."

PROCESSED_COUNT=0
FAILED_COUNT=0

for AMASS_PT_FILE in "${PT_FILES[@]}"; do
    if process_one_pt_file "$AMASS_PT_FILE"; then
        PROCESSED_COUNT=$((PROCESSED_COUNT + 1))
    else
        FAILED_COUNT=$((FAILED_COUNT + 1))
    fi
done

echo ""
echo "=============================================="
echo "Batch retargeting complete!"
echo "=============================================="
echo "Processed: $PROCESSED_COUNT"
echo "Failed:    $FAILED_COUNT"
echo "Output root: $OUTPUT_ROOT_DIR"
echo "Failed log : $FAILED_LOG"
echo ""

if [ -s "$FAILED_LOG" ]; then
    echo "Some files failed. See:"
    echo "  $FAILED_LOG"
    echo ""
fi

echo "Example verify command:"
echo "  python examples/motion_libs_visualizer.py --motion_files <your_output_pyroki.pt> --robot $ROBOT_TYPE --simulator isaacgym"
echo ""
