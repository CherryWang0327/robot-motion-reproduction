#!/usr/bin/env bash
set -euo pipefail

CONDA_ROOT="/home/unitree/miniconda3"
ENV_NAME="g1_live_dryrun"
ROS2_LOCAL_SETUP="/home/unitree/ros2_humble/install/local_setup.bash"
ROS2_SYSTEM_SETUP="/opt/ros/humble/setup.bash"
DEPLOY_DIR="/home/unitree/projects/whole_body_tracking/deployment"
LOG_DIR="/home/unitree/projects/whole_body_tracking/logs"

if [[ -f "${CONDA_ROOT}/etc/profile.d/conda.sh" ]]; then
    # shellcheck disable=SC1091
    source "${CONDA_ROOT}/etc/profile.d/conda.sh"
    conda activate "${ENV_NAME}"
fi

# Remove inherited ROS1/noetic and custom-domain variables before loading ROS2.
unset ROS_DISTRO ROS_VERSION ROS_PYTHON_VERSION ROS_DOMAIN_ID
unset AMENT_PREFIX_PATH COLCON_PREFIX_PATH ROS_PACKAGE_PATH

if [[ -f "${ROS2_LOCAL_SETUP}" ]]; then
    # shellcheck disable=SC1090
    source "${ROS2_LOCAL_SETUP}"
elif [[ -f "${ROS2_SYSTEM_SETUP}" ]]; then
    # shellcheck disable=SC1091
    source "${ROS2_SYSTEM_SETUP}"
else
    echo "ERROR: cannot find ROS2 Humble local_setup.bash or setup.bash" >&2
    exit 2
fi

unset ROS_DOMAIN_ID
mkdir -p "${LOG_DIR}"
cd "${DEPLOY_DIR}"

# This board's ros2cli does not support --no-daemon.  Clear the graph cache so
# a dead remote publisher cannot be mistaken for a live /Odometry_2 stream.
if command -v ros2 >/dev/null 2>&1; then
    ros2 daemon stop >/dev/null 2>&1 || true
fi

exec python3 -u deploy_real_robotref_160.py eno1 g1_robotref_160_taichi1.yaml \
    2>&1 | tee "${LOG_DIR}/real_single_dryrun_$(date +%Y%m%d_%H%M%S).log"
