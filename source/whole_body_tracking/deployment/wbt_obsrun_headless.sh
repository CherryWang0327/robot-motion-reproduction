#!/usr/bin/env bash
# ROS Noetic's setup.bash reads unset ROS_* variables, so do not use `-u`
# before sourcing it under a clean systemd user environment.
set -eo pipefail

WORKSPACE_DIR=/home/unitree/bysan/trans5_ws
DEPLOY_DIR=/home/unitree/BysanRL/BeyondMimic_nogmr/Beyondmimic_Deploy_G1-main/deploy_multi_policy
PIDS=()

cleanup() {
    for pid in "${PIDS[@]:-}"; do
        kill "${pid}" 2>/dev/null || true
    done
    wait "${PIDS[@]:-}" 2>/dev/null || true
}
trap cleanup EXIT INT TERM

source /opt/ros/noetic/setup.bash
source "${WORKSPACE_DIR}/devel/setup.bash"

# Reuse a running ROS master, otherwise create the one required by FAST-LIO.
if ! rosnode list >/dev/null 2>&1; then
    roscore >/tmp/wbt_obsrun_roscore.log 2>&1 &
    PIDS+=("$!")
    sleep 4
fi

roslaunch livox_ros_driver2 msg_MID360.launch >/tmp/wbt_obsrun_livox.log 2>&1 &
PIDS+=("$!")
sleep 8

roslaunch fast_lio mapping_mid360.launch >/tmp/wbt_obsrun_fastlio.log 2>&1 &
PIDS+=("$!")
sleep 8

python3 "${WORKSPACE_DIR}/py/fastlio_body_to_base_bridge.py" >/tmp/wbt_obsrun_body_to_base.log 2>&1 &
PIDS+=("$!")
sleep 2

(
    source /opt/ros/noetic/setup.bash
    source "${WORKSPACE_DIR}/devel/setup.bash"
    cd "${DEPLOY_DIR}"
    exec python3 tools/ros1_odom_to_ros2_udp_bridge_with_twist.py \
        --mode ros1_tx \
        --ros1_topic /odometry/base_link \
        --udp_host 127.0.0.1 \
        --udp_port 15555 \
        --estimate_twist_from_pose \
        --twist_lpf_alpha 0.35 \
        --print_pose \
        --pose_print_every 50 \
        --pose_print_rpy
) >/tmp/wbt_obsrun_ros1_tx.log 2>&1 &
PIDS+=("$!")

# Fail the service if any required child exits; systemd restarts the whole chain.
while true; do
    for pid in "${PIDS[@]}"; do
        kill -0 "${pid}" 2>/dev/null || exit 1
    done
    sleep 2
done
