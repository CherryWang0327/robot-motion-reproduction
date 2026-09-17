"""
Usage:

# 160D robot-reference版本，沿用177D运行结构。默认 dry-run：
#   - use_ros_state = False：从 Unitree DDS low_state 读取 q/dq/imu
#   - use_ros_cmd   = False：如果 dry_run=False，则用 Unitree DDS send_cmd 发命令
#   - dry_run       = True ：不发送任何控制命令，只读取状态、订阅 /Odometry_2、构造 obs、跑 ONNX、打印信息

cd /home/unitree/projects/whole_body_tracking/deployment

python3 deploy_real_robotref_160.py enp0s31f6 g1_robotref_160.yaml


PYTHONUNBUFFERED=1 python3 -u deploy_real_robotref_160.py enp0s31f6 g1_robotref_160.yaml \
  2>&1 | tee /home/unitree/projects/whole_body_tracking/logs/real_single_dryrun_$(date +%Y%m%d_%H%M%S).log

注意：
1. 当前脚本仍然需要 Unitree low_state，因为遥控器按键也来自 low_state。
2. 当前脚本始终需要 /Odometry_2，因为160维G1FlatEnvCfg观测需要：
   - motion_anchor_pos_b
   - base_lin_vel
3. 如果 use_ros_state=True 或 use_ros_cmd=True，才需要 unitree_rl_msgs。
"""

from typing import Union
import numpy as np
import time
import os
import threading
# ======== newADD start======
from scipy.spatial.transform import Rotation as R
# =========== newADD end ========

from unitree_sdk2py.core.channel import ChannelPublisher, ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_, unitree_hg_msg_dds__LowState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_, unitree_go_msg_dds__LowState_
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as LowCmdHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as LowCmdGo
from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as LowStateHG
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as LowStateGo
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient

import onnxruntime as ort

from common.command_helper import create_damping_cmd, create_zero_cmd, init_cmd_hg, init_cmd_go, MotorMode
from common.rotation_helper import get_gravity_orientation, transform_imu_data, transform_pelvis_to_torso_complete
from common.remote_controller import RemoteController, KeyMap
from config import Config

import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from nav_msgs.msg import Odometry


# ======== newADD start======
use_ros_state = False
use_ros_cmd = False
dry_run = False

if use_ros_state or use_ros_cmd:
    from unitree_rl_msgs.msg import RobotState as RosRobotState
    from unitree_rl_msgs.msg import RobotCommand as RosRobotCommand
# =========== newADD end ========


class RosBridge(Node):
    def __init__(self):
        super().__init__("policy_infer_bridge")

        qos = QoSProfile(
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
            reliability=ReliabilityPolicy.BEST_EFFORT,
        )

        # ======== newADD start======
        # /robot/state 是可选状态源。
        self._state_lock = threading.Lock()
        self._have_state = threading.Event()
        self._last_state = None

        if use_ros_state:
            self._sub = self.create_subscription(
                RosRobotState, "/robot/state", self._on_state, qos
            )

        # /robot/command 是可选命令输出。
        self._pub = None
        if use_ros_cmd and not dry_run:
            self._pub = self.create_publisher(
                RosRobotCommand, "/robot/command", qos
            )
        # =========== newADD end ========

        # /Odometry_2 始终订阅，因为160维obs需要odom pose和base_lin_vel。
        self._odom_lock = threading.Lock()
        self._have_odom = threading.Event()
        self._last_odom: Odometry = None
        self._last_odom_recv_t = 0.0

        self._odom_sub = self.create_subscription(
            Odometry, "/Odometry_2", self._on_odom, qos
        )

    # ======== newADD start======
    def _on_state(self, msg):
        """
        缓存 /robot/state 最新一帧。

        仅在 use_ros_state=True 时会被订阅器调用。
        """
        with self._state_lock:
            self._last_state = msg
        self._have_state.set()

    def wait_state(self, timeout_s: float = 5.0) -> bool:
        """
        等待 /robot/state 至少收到一帧。

        仅 use_ros_state=True 时需要调用。
        """
        return self._have_state.wait(timeout=timeout_s)

    def get_state_copy(self):
        """
        获取 /robot/state 缓存。

        仅 use_ros_state=True 时使用。
        """
        with self._state_lock:
            return self._last_state
    # =========== newADD end ========

    def _on_odom(self, msg: Odometry):
        """
        缓存 /Odometry_2 最新一帧。

        /Odometry_2 使用 nav_msgs/msg/Odometry。
        pose 用于计算 motion_anchor_pos_b；
        twist.linear 用于填 base_lin_vel。
        """
        with self._odom_lock:
            self._last_odom = msg
            self._last_odom_recv_t = time.perf_counter()
        self._have_odom.set()

    def wait_odom(self, timeout_s: float = 5.0) -> bool:
        """
        等待 /Odometry_2 至少收到一帧。
        """
        return self._have_odom.wait(timeout=timeout_s)

    def get_odom_copy(self, max_age_s: float = 0.2):
        """
        返回最新 odom 的位置、姿态和线速度。

        Returns:
            pos_w:
                np.ndarray, shape=(3,)，odom/world 系下的位置。
            quat_wxyz:
                np.ndarray, shape=(4,)，odom/world 系下的姿态，wxyz 顺序。
            lin_vel_b:
                np.ndarray, shape=(3,)，base 坐标系下的线速度。
                这里默认 /Odometry_2.twist.twist.linear 已经是 base frame。
            ok:
                bool，True 表示 odom 数据存在且没有超时。
        """
        with self._odom_lock:
            msg = self._last_odom
            recv_t = self._last_odom_recv_t

        if msg is None:
            return (
                np.zeros(3, dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
            )

        age = time.perf_counter() - recv_t
        if age > max_age_s:
            return (
                np.zeros(3, dtype=np.float32),
                np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
                np.zeros(3, dtype=np.float32),
                False,
            )

        p = msg.pose.pose.position
        q = msg.pose.pose.orientation
        v = msg.twist.twist.linear

        pos_w = np.array([p.x, p.y, p.z], dtype=np.float32)

        # ROS quaternion 是 xyzw，这里转成当前代码使用的 wxyz。
        quat_wxyz = np.array([q.w, q.x, q.y, q.z], dtype=np.float32)
        quat_wxyz = quat_wxyz / (np.linalg.norm(quat_wxyz) + 1e-8)

        # 假设你的桥接脚本已经把 twist.linear 写成 base frame 速度。
        lin_vel_b = np.array([v.x, v.y, v.z], dtype=np.float32)

        return pos_w, quat_wxyz, lin_vel_b, True

    def publish_cmd(self, q_des, kp, kd, tau_ff=None):
        """
        通过 ROS /robot/command 发布命令。

        仅 use_ros_cmd=True 且 dry_run=False 时由 rosbridge_send_cmd 调用。
        """
        # ======== newADD start======
        if not use_ros_cmd:
            return

        if self._pub is None:
            return
        # =========== newADD end ========

        m = RosRobotCommand()
        m.q_des = [float(x) for x in q_des]
        m.kp = [float(x) for x in kp]
        m.kd = [float(x) for x in kd]

        if tau_ff is None:
            m.tau_ff = [0.0] * len(m.q_des)
        else:
            m.tau_ff = [float(x) for x in tau_ff]

        self._pub.publish(m)


# ONNX Runtime setup
so = ort.SessionOptions()
so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
so.intra_op_num_threads = 1
so.inter_op_num_threads = 1
so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL


def axis_angle_to_quat_wxyz_np(aa: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    """
    axis-angle 转 wxyz 四元数。

    Args:
        aa: shape=(3,) 的 axis-angle。
        eps: 小角度阈值。

    Returns:
        shape=(4,) 的 wxyz 四元数。
    """
    aa = np.asarray(aa, dtype=np.float64)
    angle = np.linalg.norm(aa)

    if angle < eps:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    axis = aa / angle
    half = 0.5 * angle
    s = np.sin(half)

    return np.array(
        [np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s],
        dtype=np.float64,
    )

class RobotReferenceNpzLoader:
    """读取154D observation实际使用的机器人参考字段。"""

    def __init__(self, motion_file: str, torso_body_index: int = 9):
        assert os.path.isfile(motion_file), f"Invalid robot-reference NPZ path: {motion_file}"
        motion = np.load(motion_file)
        self.joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
        self.joint_vel = np.asarray(motion["joint_vel"], dtype=np.float32)
        self.body_pos_w = np.asarray(motion["body_pos_w"], dtype=np.float32)
        self.body_quat_w = np.asarray(motion["body_quat_w"], dtype=np.float32)
        motion.close()

        self.time_step_total = int(self.joint_pos.shape[0])
        self.torso_body_index = int(torso_body_index)
        self.torso_pos_w = self.body_pos_w[:, self.torso_body_index, :]
        self.torso_quat_w = self.body_quat_w[:, self.torso_body_index, :]

        print("[RobotReferenceNpzLoader]")
        print(f"  motion_file       = {motion_file}")
        print(f"  frames            = {self.time_step_total}")
        print(f"  joint_pos.shape   = {self.joint_pos.shape}")
        print(f"  joint_vel.shape   = {self.joint_vel.shape}")
        print(f"  torso_body_index  = {self.torso_body_index}")
        print(f"  torso_pos_w[0]    = {self.torso_pos_w[0]}")
        print(f"  torso_quat_w[0]   = {self.torso_quat_w[0]}")
joint_seq = [
    'left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint',
    'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint',
    'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint',
    'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint',
    'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint',
    'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint',
    'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint',
    'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint',
    'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint',
    'left_wrist_yaw_joint', 'right_wrist_yaw_joint'
]

joint_xml = [
    "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
    "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
    "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
    "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
    "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
    "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
    "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint",
    "left_wrist_yaw_joint", "right_shoulder_pitch_joint", "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint", "right_elbow_joint", "right_wrist_roll_joint",
    "right_wrist_pitch_joint", "right_wrist_yaw_joint"
]


def quaternion_conjugate(q):
    """
    四元数共轭。

    Args:
        q: shape=(4,) 的 wxyz 四元数。

    Returns:
        shape=(4,) 的共轭四元数。
    """
    return np.array([q[0], -q[1], -q[2], -q[3]])


def quaternion_multiply(q1, q2):
    """
    四元数乘法 q1 ⊗ q2。

    Args:
        q1: shape=(4,) 的 wxyz 四元数。
        q2: shape=(4,) 的 wxyz 四元数。

    Returns:
        shape=(4,) 的 wxyz 四元数。
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return np.array([w, x, y, z])


def quaternion_to_rotation_matrix(q):
    """
    将 wxyz 四元数转换为旋转矩阵。

    Args:
        q: shape=(4,) 的四元数 [w, x, y, z]。

    Returns:
        shape=(3,3) 的旋转矩阵。
    """
    q = np.array(q, dtype=np.float64)
    q = q / (np.linalg.norm(q) + 1e-12)

    w, x, y, z = q

    r00 = 1 - 2 * y**2 - 2 * z**2
    r01 = 2 * x * y - 2 * z * w
    r02 = 2 * x * z + 2 * y * w

    r10 = 2 * x * y + 2 * z * w
    r11 = 1 - 2 * x**2 - 2 * z**2
    r12 = 2 * y * z - 2 * x * w

    r20 = 2 * x * z - 2 * y * w
    r21 = 2 * y * z + 2 * x * w
    r22 = 1 - 2 * x**2 - 2 * y**2

    return np.array(
        [
            [r00, r01, r02],
            [r10, r11, r12],
            [r20, r21, r22],
        ],
        dtype=np.float64,
    )


def yaw_only_rotation_matrix_from_quat_wxyz(q: np.ndarray) -> np.ndarray:
    """
    从 wxyz 四元数中提取 yaw-only 旋转矩阵。

    Args:
        q: shape=(4,) 的 wxyz 四元数。

    Returns:
        shape=(3,3) 的 yaw-only 旋转矩阵。
    """
    R = quaternion_to_rotation_matrix(q)
    yaw = np.arctan2(R[1, 0], R[0, 0])
    cy = np.cos(yaw)
    sy = np.sin(yaw)

    return np.array(
        [
            [cy, -sy, 0.0],
            [sy, cy, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
# ======== newADD start======
def quat_wxyz_to_yaw_deg(q_wxyz: np.ndarray) -> float:
    """
    从 wxyz 四元数中提取 yaw，单位为 degree。

    Args:
        q_wxyz:
            shape=(4,) 的四元数 [w, x, y, z]。

    Returns:
        yaw_deg:
            绕 z 轴 yaw 角，单位 degree，范围大致为 [-180, 180]。
    """
    R_mat = quaternion_to_rotation_matrix(q_wxyz)
    yaw = np.arctan2(R_mat[1, 0], R_mat[0, 0])
    return float(np.rad2deg(yaw))


def wrap_angle_deg(angle_deg: float) -> float:
    """
    把角度 wrap 到 [-180, 180)。

    Args:
        angle_deg:
            任意角度，单位 degree。

    Returns:
        wrapped:
            wrap 后的角度。
    """
    return float((angle_deg + 180.0) % 360.0 - 180.0)
# =========== newADD end ==￿];󛨑鬶��q�^￿�tor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i] * 15
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i] * 5
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            for i in range(len(self.config.arm_waist_joint2motor_idx)):
                motor_idx = self.config.arm_waist_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i + 12]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i + 12] * 15
                self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i + 12] * 5
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.maybe_send_low_cmd()
            # ======== newADD start======
            self.print_pre_a_yaw_alignment()
            # =========== newADD end ========

            time.sleep(self.config.control_dt)

    def read_robot_state_for_obs(self):
        """
        读取 policy observation 所需的 q/dq/imu。

        Returns:
            ok:
                bool，状态是否可用。
            quat:
                shape=(4,) 的 wxyz IMU/root 四元数。
            ang_vel:
                shape=(3,) 的角速度。
        """
        if use_ros_state:
            ros_st = self.ros.get_state_copy()
            if ros_st is None:
                return False, None, None

            self.qj[:] = np.asarray(ros_st.q, dtype=np.float32)
            self.dqj[:] = np.asarray(ros_st.dq, dtype=np.float32)
            quat = np.asarray(ros_st.quat, dtype=np.float64)
            ang_vel = np.asarray(ros_st.gyro, dtype=np.float32).reshape(3)

            return True, quat, ang_vel

        # 原始 BYDMMC 部署方式：从 Unitree DDS low_state 读取。
        for i in range(len(self.dof_idx)):
            motor_idx = self.dof_idx[i]
            self.qj[i] = self.low_state.motor_state[motor_idx].q
            self.dqj[i] = self.low_state.motor_state[motor_idx].dq

        quat = np.asarray(self.low_state.imu_state.quaternion, dtype=np.float64)
        ang_vel = np.asarray(self.low_state.imu_state.gyroscope, dtype=np.float32).reshape(3)

        return True, quat, ang_vel

    def run(self):
        self.counter += 1

        ok_state, quat, ang_vel = self.read_robot_state_for_obs()
        if not ok_state:
            return

        robot_pos_w, robot_quat_odom_wxyz, base_lin_vel, odom_ok = self.ros.get_odom_copy(max_age_s=0.2)
        if not odom_ok:
            print("[WARN] /Odometry_2 missing or stale, skip this control step.")
            return

        if not self._anchor_pos_offset_ready:
            self.setup_motionpos_anchor_offset(
                robot_pos_w=robot_pos_w,
                xy_only=True,
            )

        # 默认先认为当前 quat 就是 torso/root 观测姿态。
        quat_torso = quat.copy()

        if self.config.imu_type == "torso":
            waist_yaw = self.qj[self.config.arm_waist_joint2motor_idx[0]]
            waist_yaw_omega = self.dqj[self.config.arm_waist_joint2motor_idx[0]]
            quat_torso, ang_vel = transform_imu_data(
                waist_yaw=waist_yaw,
                waist_yaw_omega=waist_yaw_omega,
                imu_quat=quat,
                imu_omega=ang_vel,
            )

        elif self.config.imu_type == "pelvis":
            waist_yaw = self.qj[self.config.arm_waist_joint2motor_idx[0]]
            waist_roll = self.qj[self.config.arm_waist_joint2motor_idx[1]]
            waist_pitch = self.qj[self.config.arm_waist_joint2motor_idx[2]]
            quat_torso = transform_pelvis_to_torso_complete(
                waist_yaw,
                waist_roll,
                waist_pitch,
                quat,
            )

        qj_obs = self.qj
        dqj_obs = self.dqj

        quat_torso = quaternion_multiply(q_pitch, quat_torso)
        quat_torso = quat_torso / (np.linalg.norm(quat_torso) + 1e-12)

        # obs前58维robot-reference command：不加位置offset。
        motioninput = self.motioninput_pos[self.timestep, :]

        # 只用于 motion_anchor_pos_b：加过 offset。
        motionposcurrent_for_anchor = self.motionpos_for_anchor[self.timestep, :]

        motionquatcurrent = self.motionquat[self.timestep, :]

        # # ======== newADD start======
        # if not self._yaw_debug_printed:
        #     motion_yaw_deg = quat_wxyz_to_yaw_deg(motionquatcurrent)
        #     robot_yaw_odom_deg = quat_wxyz_to_yaw_deg(robot_quat_odom_wxyz)
        #     robot_yaw_torso_deg = quat_wxyz_to_yaw_deg(quat_torso)

        #     yaw_err_odom_deg = wrap_angle_deg(robot_yaw_odom_deg - motion_yaw_deg)
        #     yaw_err_torso_deg = wrap_angle_deg(robot_yaw_torso_deg - motion_yaw_deg)

        #     print("[Yaw Debug At Policy Start]")
        #     print(f"  motion_yaw_deg_after_zup = {motion_yaw_deg:.3f}")
        #     print(f"  robot_yaw_odom_deg       = {robot_yaw_odom_deg:.3f}")
        #     print(f"  robot_yaw_torso_deg      = {robot_yaw_torso_deg:.3f}")
        #     print(f"  yaw_err_odom_deg         = {yaw_err_odom_deg:.3f}  # robot_odom - motion")
        #     print(f"  yaw_err_torso_deg        = {yaw_err_torso_deg:.3f} # robot_torso - motion")
        #     print("  Expected physical heading:")
        #     print("    yaw_err close to 0 deg means robot heading matches sim yaw initialization.")
        #     print("    yaw_err around 90/180 deg means robot is physically facing the wrong direction.")

        #     self._yaw_debug_printed = True
        # # =========== newADD end ========

        motion_anchor_pos_b = self.compute_motion_anchor_pos_b_from_offset_motion(
            robot_pos_w=robot_pos_w,
            robot_quat_wxyz=robot_quat_odom_wxyz,
            motion_pos_for_anchor=motionposcurrent_for_anchor,
        )

        relquat = quaternion_multiply(quaternion_conjugate(quat_torso), motionquatcurrent)
        relquat = relquat / (np.linalg.norm(relquat) + 1e-12)
        relmatrix = quaternion_to_rotation_matrix(relquat)[:, :2].reshape(-1,)

        offset = 0

        # 1. robot-reference command: 58D = joint_pos(29) + joint_vel(29)
        self.obs[0][offset:offset + 58] = motioninput
        offset += 58

        # 2. motion_anchor_pos_b: 3D
        self.obs[0][offset:offset + 3] = motion_anchor_pos_b
        offset += 3

        # 3. motion_anchor_ori_b: 6D
        self.obs[0][offset:offset + 6] = relmatrix
        offset += 6

        # 4. base_lin_vel: 3D，来自 /Odometry_2。
        self.obs[0][offset:offset + 3] = base_lin_vel
        offset += 3

        # 5. base_ang_vel: 3D，来自 IMU gyro 或 ROS state gyro。
        self.obs[0][offset:offset + 3] = ang_vel
        offset += 3

        # 6. joint_pos_rel: 29D
        qpos_urdf = qj_obs
        qj_obs_seq = qpos_urdf[self.seq_idx_in_xml]
        self.obs[0][offset:offset + 29] = qj_obs_seq - self.config.default_angles_seq
        offset += 29

        # 7. joint_vel: 29D
        qvel_urdf = dqj_obs
        dqj_obs_seq = qvel_urdf[self.seq_idx_in_xml]
        self.obs[0][offset:offset + 29] = dqj_obs_seq
        offset += 29

        # 8. last_action: 29D
        self.obs[0][offset:offset + 29] = self.action_buffer
        offset += 29

        if offset != self.config.num_obs:
            raise RuntimeError(
                f"Observation dim mismatch: packed offset={offset}, "
                f"config.num_obs={self.config.num_obs}"
            )

        # if self.counter % 50 == 0:
        #     print(
        #         f"[OBS] dim={offset}, "
        #         f"motioninput_transl={motioninput[:3]}, "
        #         f"motionpos_for_anchor={motionposcurrent_for_anchor}, "
        #         f"robot_pos_w={robot_pos_w}, "
        #         f"motion_anchor_pos_b={motion_anchor_pos_b}, "
        #         f"base_lin_vel={base_lin_vel}, "
        #         f"base_ang_vel={ang_vel}, "
        #         # f"dry_run={dry_run}"
        #     )

        # self.time_in[0, 0] = float(self.timestep)
        # action = self.policy.run(
        #     None,
        #     {"obs": self.obs, "time_step": self.time_in},
        # )[0]
        # action = action.reshape(-1).astype(np.float32, copy=False)
        # ======== newADD start======
        self.time_in[0, 0] = float(self.timestep)

        onnx_outputs = self.policy.run(
            None,
            {"obs": self.obs, "time_step": self.time_in},
        )

        output_dict = {
            name: value
            for name, value in zip(self.policy_output_names, onnx_outputs)
        }
        
        action = output_dict["actions"]
        action = action.reshape(-1).astype(np.float32, copy=False)

        ref_q_seq = None
        ref_dq_seq = None

        if "joint_pos" in output_dict:
            ref_q_seq = output_dict["joint_pos"].reshape(-1).astype(np.float32, copy=False)

        if "joint_vel" in output_dict:
            ref_dq_seq = output_dict["joint_vel"].reshape(-1).astype(np.float32, copy=False)
        # =========== newADD end ========

        self.action = action.copy()
        self.action_buffer = action.copy()

        target_dof_pos = self.config.default_angles_seq + self.action * self.config.action_scale_seq
        target_dof_pos = target_dof_pos.reshape(-1)
        target_dof_pos = target_dof_pos[self.xml_idx_in_seq]

        lo = self.joint_lower + self.joint_limit_eps
        hi = self.joint_upper - self.joint_limit_eps
        target_dof_pos = np.clip(target_dof_pos.astype(np.float32, copy=False), lo, hi)

        # ======== newADD start======
        self.print_real_ref_debug(
            motioninput=motioninput,
            motionposcurrent_for_anchor=motionposcurrent_for_anchor,
            robot_pos_w=robot_pos_w,
            motion_anchor_pos_b=motion_anchor_pos_b,
            base_lin_vel=base_lin_vel,
            ang_vel=ang_vel,
            real_q_seq=qj_obs_seq,
            real_dq_seq=dqj_obs_seq,
            ref_q_seq=ref_q_seq,
            ref_dq_seq=ref_dq_seq,
            action=action,
            target_q_xml=target_dof_pos,
        )
        # =========== newADD end ========

        self.timestep += 1

        for i in range(len(self.config.leg_joint2motor_idx)):
            motor_idx = self.config.leg_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        for i in range(len(self.config.arm_waist_joint2motor_idx)):
            motor_idx = self.config.arm_waist_joint2motor_idx[i]
            self.low_cmd.motor_cmd[motor_idx].q = target_dof_pos[i + 12]
            self.low_cmd.motor_cmd[motor_idx].dq = 0
            self.low_cmd.motor_cmd[motor_idx].kp = self.config.stiffness[i + 12]
            self.low_cmd.motor_cmd[motor_idx].kd = self.config.damping[i + 12]
            self.low_cmd.motor_cmd[motor_idx].tau = 0

        self.maybe_send_low_cmd()

        self._next_t += self._dt
        now = time.perf_counter()
        remain = self._next_t - now

        # if self.counter % 50 == 0:
        #     print(
        #         f"[loop] next_t={self._next_t:.6f}, "
        #         f"now={now:.6f}, "
        #         f"remain={remain:.6f}"
        #     )
        # ======== newADD start======
        if remain < -0.005:
            print(
                f"[LOOP_LATE] step={self.timestep}, "
                f"late_ms={-remain * 1000.0:.2f}, "
                f"next_t={self._next_t:.6f}, "
                f"now={now:.6f}"
            )
        # =========== newADD end ========

        if remain > 0.002:
            time.sleep(remain - 0.001)

        if self.timestep >= self.motion.time_step_total:
            self.mainloop_flag = False


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("net", type=str, help="network interface")
    parser.add_argument(
        "config",
        type=str,
        help="config file name in the configs folder",
        default="g1_for_bydmimic.yaml",
    )
    args = parser.parse_args()

    config_path = f"configs/{args.config}"
    config = Config(config_path)

    ChannelFactoryInitialize(0, args.net)

    controller = None
    control_session_started = False
    try:
        controller = Controller(config)

        controller.zero_torque_state()
        # zero_torque_state() returns only after Start is received.  From this
        # point this process has begun its low-level control session.
        control_session_started = True
        controller.move_to_default_pos()
        controller.wait_for_obsrun_ready()
        controller.default_pos_state()

        controller._next_t = time.perf_counter()

        while controller.mainloop_flag:
            controller.run()
            if controller.remote_controller.button[KeyMap.select] == 1:
                break
    except KeyboardInterrupt:
        print("KeyboardInterrupt: stopping control session.")
    finally:
        if controller is not None and control_session_started:
            try:
                create_damping_cmd(controller.low_cmd)
                controller.maybe_send_low_cmd()
                print("[CLEANUP] Damping command sent.")
            except Exception as exc:
                print(f"[CLEANUP] Failed to send damping command: {exc}")

            # Release the currently active motion mode instead of re-selecting
            # AI.  This prevents this deployment from leaving a stale mode
            # owner after normal exit, Ctrl+C, or an exception.
            try:
                motion_switcher = MotionSwitcherClient()
                motion_switcher.SetTimeout(3.0)
                motion_switcher.Init()
                status, before = motion_switcher.CheckMode()
                print(f"[CLEANUP] MotionSwitcher before release: status={status}, mode={before}")
                if status == 0 and before and before.get("name"):
                    release_status, _ = motion_switcher.ReleaseMode()
                    print(f"[CLEANUP] ReleaseMode status={release_status}")
                status, after = motion_switcher.CheckMode()
                print(f"[CLEANUP] MotionSwitcher after release: status={status}, mode={after}")
            except Exception as exc:
                print(f"[CLEANUP] MotionSwitcher cleanup failed: {exc}")

        print("Exit")

