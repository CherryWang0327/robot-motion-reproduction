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
# =========== newADD end ========


PITCH_BIAS_DEG = 0.0
pitch_bias = PITCH_BIAS_DEG * np.pi / 180.0
q_pitch = axis_angle_to_quat_wxyz_np(
    np.array([0.0, pitch_bias, 0.0], dtype=np.float64)
)


class Controller:
    def __init__(self, config: Config) -> None:
        self.config = config
        self.remote_controller = RemoteController()

        self.onnx = config.policy_path
        self.policy = ort.InferenceSession(
            self.onnx,
            sess_options=so,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        print("inferenceSession:", self.policy.get_providers())

        self.qj = np.zeros(config.num_actions, dtype=np.float32)
        self.dqj = np.zeros(config.num_actions, dtype=np.float32)
        self.action = np.zeros(config.num_actions, dtype=np.float32)
        self.target_dof_pos = config.default_angles.copy()
        self.obs = np.zeros((1, self.config.num_obs), dtype=np.float32)
        self.time_in = np.zeros((1, 1), dtype=np.float32)

        self.cmd = np.array([0.0, 0, 0])
        self.counter = 0
        self.timestep = 0
        self.mainloop_flag = True

        self._next_t = time.perf_counter()
        self._dt = float(self.config.control_dt)

        self.seq_idx_in_xml = np.array(
            [joint_xml.index(j) for j in joint_seq],
            dtype=np.int64,
        )
        self.xml_idx_in_seq = np.array(
            [joint_seq.index(j) for j in joint_xml],
            dtype=np.int64,
        )

        self.joint_lower = np.array([
            -2.5, -0.5, -2.7, -0.087, -0.87, -0.26,
            -2.5, -2.9, -2.7, -0.087, -0.87, -0.26,
            -2.6, -0.52, -0.52,
            -3.08, -1.58, -2.6, -1.04, -1.97, -1.614, -1.614,
            -3.08, -2.25, -2.6, -1.04, -1.97, -1.614, -1.614,
        ], dtype=np.float32)

        self.joint_upper = np.array([
            2.87, 2.96, 2.75, 2.87, 0.5236, 0.2618,
            2.87, 0.52, 2.75, 2.87, 0.5236, 0.2619,
            2.618, 0.52, 0.52,
            2.6, 2.25, 2.6, 2.09, 1.97, 1.614, 1.614,
            2.6, 1.58, 2.6, 2.09, 1.97, 1.614, 1.614,
        ], dtype=np.float32)

        assert self.joint_lower.shape[0] == self.config.num_actions
        assert self.joint_upper.shape[0] == self.config.num_actions

        self.joint_limit_eps = 1e-3
        self.late_max = 0.0
        self.late_cnt = 0
        self.loop_cnt = 0

        self.motion_npz = config.motion_npz
        self.motion = RobotReferenceNpzLoader(self.motion_npz)

        print("[INIT] Load onnx:", self.onnx)
        print("[INIT] Load robot-reference motion:", self.motion_npz)

        self.motionpos = self.motion.torso_pos_w
        self.motionquat = self.motion.torso_quat_w
        # ======== newADD start======
        # 参考机器人NPZ中torso第0帧yaw，用于沿用177D的人工朝向对齐。
        self.motionquat0_wxyz = self.motionquat[0, :]
        self.motion_yaw0_deg = quat_wxyz_to_yaw_deg(self.motionquat0_wxyz)

        # 等待 A 阶段的 yaw 对齐打印节流参数。
        self._pre_a_yaw_print_t = 0.0
        self._pre_a_yaw_print_period_s = 0.5

        print("[Motion Heading]")
        print(f"  motion_torso_yaw0 = {self.motion_yaw0_deg:.3f} deg")
        print("  During default_pos_state, rotate the robot until yaw_err_odom_deg is close to 0 deg.")
        # =========== newADD end ========

        # 160D robot-reference版本：command = joint_pos(29) + joint_vel(29) = 58。
        self.motioninput_pos = np.concatenate(
            [self.motion.joint_pos, self.motion.joint_vel],
            axis=-1,
        ).astype(np.float32, copy=False)

        assert self.motioninput_pos.shape[1] == 58, (
            f"Expected robot-reference motioninput dim 58, got {self.motioninput_pos.shape[1]}"
        )

        # motionpos_for_anchor 只用于 motion_anchor_pos_b，不改 motioninput_pos。
        self.motionpos_for_anchor = self.motionpos.copy()
        self._anchor_pos_offset_ready = False
        self._motionpos_anchor_offset = np.zeros(3, dtype=np.float64)
        # ======== newADD start======
        self._yaw_debug_printed = False
        # =========== newADD end ========

        self.action_buffer = np.zeros((self.config.num_actions,), dtype=np.float32)

        self.dof_idx = [
            0, 1, 2, 3, 4, 5,
            6, 7, 8, 9, 10, 11,
            12, 13, 14,
            15, 16, 17, 18, 19, 20, 21,
            22, 23, 24, 25, 26, 27, 28,
        ]

        self.lowcmd_publisher_ = None

        if config.msg_type == "hg":
            self.low_cmd = unitree_hg_msg_dds__LowCmd_()
            self.low_state = unitree_hg_msg_dds__LowState_()
            self.mode_pr_ = MotorMode.PR
            self.mode_machine_ = 0

            if not dry_run:
                self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdHG)
                self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateHG)
            self.lowstate_subscriber.Init(self.LowStateHgHandler, 10)

        elif config.msg_type == "go":
            self.low_cmd = unitree_go_msg_dds__LowCmd_()
            self.low_state = unitree_go_msg_dds__LowState_()

            if not dry_run:
                self.lowcmd_publisher_ = ChannelPublisher(config.lowcmd_topic, LowCmdGo)
                self.lowcmd_publisher_.Init()

            self.lowstate_subscriber = ChannelSubscriber(config.lowstate_topic, LowStateGo)
            self.lowstate_subscriber.Init(self.LowStateGoHandler, 10)

        else:
            raise ValueError("Invalid msg_type")

        self.wait_for_low_state()

        # ROS 只强制用于 /Odometry_2。
        if not rclpy.ok():
            rclpy.init(args=None)

        self.ros = RosBridge()
        self._ros_exec = rclpy.executors.SingleThreadedExecutor()
        self._ros_exec.add_node(self.ros)

        self._ros_thread = threading.Thread(
            target=self._ros_exec.spin,
            daemon=True,
        )
        self._ros_thread.start()

        # OBSRun 需要在机器人按 Start 并完成直立后才启动；因此这里仅启动
        # ROS 接收线程，不在构造阶段等待 /robot/state 或 /Odometry_2。
        # 具体就绪检查由 wait_for_obsrun_ready() 在直立后完成。
        print("ROS bridge started; OBSRun readiness will be checked after the robot stands.")

        if config.msg_type == "hg":
            init_cmd_hg(self.low_cmd, self.mode_machine_, self.mode_pr_)
        elif config.msg_type == "go":
            init_cmd_go(self.low_cmd, weak_motor=self.config.weak_motor)

        print("ONNX inputs:")
        for inp in self.policy.get_inputs():
            print(f"  name={inp.name}, shape={inp.shape}, type={inp.type}")

        print("ONNX outputs:")
        for out in self.policy.get_outputs():
            print(f"  name={out.name}, shape={out.shape}, type={out.type}")

        # ======== newADD start======
        # 保存 ONNX 输出名，后面 policy.run(None, ...) 后可以按名字取 actions / joint_pos / joint_vel。
        self.policy_output_names = [out.name for out in self.policy.get_outputs()]
        print(f"ONNX output names = {self.policy_output_names}")
        # =========== newADD end ========
        # ======== newADD start======
        self.compare_onnx_metadata_with_config()
        # =========== newADD end ========

        print("[Runtime Switches]")
        print(f"  use_ros_state = {use_ros_state}")
        print(f"  use_ros_cmd   = {use_ros_cmd}")
        print(f"  dry_run       = {dry_run}")

    def setup_motionpos_anchor_offset(self, robot_pos_w: np.ndarray, xy_only: bool = True):
        """
        根据 policy 启动时的机器人位置，为 motion_anchor_pos_b 构造专用 transl offset。

        这个函数只改 self.motionpos_for_anchor，不改 self.motionpos，也不改 self.motioninput_pos。
        因此 obs 前58维robot-reference command保持原NPZ数值，不加入位置offset。

        Args:
            robot_pos_w: shape=(3,) 的机器人 odom/world 位置。
            xy_only: True 表示只对齐 x/y，不改 z。
        """
        robot_pos_w = np.asarray(robot_pos_w, dtype=np.float64).reshape(3)
        motion0 = np.asarray(self.motionpos[0], dtype=np.float64).reshape(3)

        offset = np.zeros(3, dtype=np.float64)
        if xy_only:
            offset[:2] = robot_pos_w[:2] - motion0[:2]
            offset[2] = 0.0
        else:
            offset = robot_pos_w - motion0

        self._motionpos_anchor_offset = offset
        self.motionpos_for_anchor = self.motionpos + offset[None, :]
        self._anchor_pos_offset_ready = True

        print("[Anchor Offset Setup]")
        print(f"  robot_pos_w             = {robot_pos_w}")
        print(f"  motionpos[0]            = {motion0}")
        print(f"  offset_for_anchor_only  = {offset}")
        print(f"  motionpos_for_anchor[0] = {self.motionpos_for_anchor[0]}")
        print(f"  motioninput_pos[0,:3]   = {self.motioninput_pos[0, :3]}")

    def compute_motion_anchor_pos_b_from_offset_motion(
        self,
        robot_pos_w: np.ndarray,
        robot_quat_wxyz: np.ndarray,
        motion_pos_for_anchor: np.ndarray,
    ):
        """
        用已经 offset 过的 motion_pos_for_anchor 计算 motion_anchor_pos_b。

        Args:
            robot_pos_w: shape=(3,) 的真实机器人 odom/world 位置。
            robot_quat_wxyz: shape=(4,) 的真实机器人 odom/world 姿态，wxyz。
            motion_pos_for_anchor: shape=(3,) 的 offset 后参考 anchor 位置。

        Returns:
            shape=(3,) 的参考 anchor 在机器人当前 body/root frame 下的位置。
        """
        robot_pos_w = np.asarray(robot_pos_w, dtype=np.float64).reshape(3)
        robot_quat_wxyz = np.asarray(robot_quat_wxyz, dtype=np.float64).reshape(4)
        motion_pos_for_anchor = np.asarray(motion_pos_for_anchor, dtype=np.float64).reshape(3)

        R_robot_w = quaternion_to_rotation_matrix(robot_quat_wxyz)
        motion_anchor_pos_b = R_robot_w.T @ (motion_pos_for_anchor - robot_pos_w)

        return motion_anchor_pos_b.astype(np.float32)

# ======== newADD start======
    def _format_vec(self, arr, precision: int = 3, max_items=None) -> str:
        """
        将向量格式化成紧凑字符串，便于终端观察。

        Args:
            arr: 任意可转成 np.ndarray 的数组。
            precision: 小数位数。
            max_items: 如果不为 None，只打印前 max_items 个元素。

        Returns:
            格式化后的字符串。
        """
        arr = np.asarray(arr, dtype=np.float64).reshape(-1)

        if max_items is not None:
            arr = arr[:max_items]

        return np.array2string(
            arr,
            precision=precision,
            suppress_small=True,
            separator=", ",
            max_line_width=200,
        )


    def _print_topk_q_error(
        self,
        real_q_seq: np.ndarray,
        ref_q_seq: np.ndarray,
        target_q_seq: np.ndarray = None,
        k: int = 8,
    ) -> None:
        """
        打印 real/ref q 误差最大的 top-k 关节。

        Args:
            real_q_seq: shape=(29,)，policy joint_seq 顺序下的真实关节角。
            ref_q_seq: shape=(29,)，policy joint_seq 顺序下的参考关节角。
            target_q_seq: shape=(29,)，policy joint_seq 顺序下的 PD target，可选。
            k: 打印误差最大的前 k 个关节。

        Returns:
            None.
        """
        real_q_seq = np.asarray(real_q_seq, dtype=np.float64).reshape(-1)
        ref_q_seq = np.asarray(ref_q_seq, dtype=np.float64).reshape(-1)
        q_err = real_q_seq - ref_q_seq

        idxs = np.argsort(np.abs(q_err))[::-1][:k]

        print("  q_err_top{}:".format(k))
        for rank, idx in enumerate(idxs, start=1):
            joint_name = joint_seq[idx]
            if target_q_seq is None:
                print(
                    f"    {rank:02d}. {joint_name:<28s} "
                    f"real={real_q_seq[idx]: .4f}, "
                    f"ref={ref_q_seq[idx]: .4f}, "
                    f"err={q_err[idx]: .4f}"
                )
            else:
                target_q_seq = np.asarray(target_q_seq, dtype=np.float64).reshape(-1)
                print(
                    f"    {rank:02d}. {joint_name:<28s} "
                    f"real={real_q_seq[idx]: .4f}, "
                    f"ref={ref_q_seq[idx]: .4f}, "
                    f"err={q_err[idx]: .4f}, "
                    f"target={target_q_seq[idx]: .4f}"
                )


    def print_real_ref_debug(
        self,
        motioninput: np.ndarray,
        motionposcurrent_for_anchor: np.ndarray,
        robot_pos_w: np.ndarray,
        motion_anchor_pos_b: np.ndarray,
        base_lin_vel: np.ndarray,
        ang_vel: np.ndarray,
        real_q_seq: np.ndarray,
        real_dq_seq: np.ndarray,
        ref_q_seq: np.ndarray = None,
        ref_dq_seq: np.ndarray = None,
        action: np.ndarray = None,
        target_q_xml: np.ndarray = None,
    ) -> None:
        """
        统一打印 real/ref 调试信息。

        打印策略：
            1. 前 20 个 policy step 每帧打印，方便看第一帧是否异常；
            2. 之后每 50 个 policy step 打印一次摘要；
            3. 重点显示新增 6 维和 29 维 q 的 real/ref 关系。

        Args:
            motioninput:
                当前输入policy的58D robot-reference command：29D q_ref + 29D dq_ref。
            motionposcurrent_for_anchor:
                加 offset 后、只用于计算 motion_anchor_pos_b 的 ref anchor position。
            robot_pos_w:
                /Odometry_2 给出的 real anchor/base position。
            motion_anchor_pos_b:
                输入 policy 的新增 3D 相对 anchor position。
            base_lin_vel:
                输入 policy 的新增 3D real base linear velocity。
            ang_vel:
                输入 policy 的 base angular velocity。
            real_q_seq:
                shape=(29,)，policy joint_seq 顺序下的真实关节角。
            real_dq_seq:
                shape=(29,)，policy joint_seq 顺序下的真实关节速度。
            ref_q_seq:
                shape=(29,)，ONNX 输出的参考 joint_pos，可选。
            ref_dq_seq:
                shape=(29,)，ONNX 输出的参考 joint_vel，可选。
            action:
                shape=(29,)，policy action，可选。
            target_q_xml:
                shape=(29,)，XML joint order 下的 PD target，可选。

        Returns:
            None.
        """
        step = int(self.timestep)
        full_print = step < 20
        periodic_print = (self.counter % 50 == 0)

        if not (full_print or periodic_print):
            return

        real_q_seq = np.asarray(real_q_seq, dtype=np.float64).reshape(-1)
        real_dq_seq = np.asarray(real_dq_seq, dtype=np.float64).reshape(-1)

        print("\n" + "=" * 90)
        print(f"[REAL/REF DEBUG] step={step}, counter={self.counter}, full_print={full_print}")

        # ------------------------------------------------------------
        # 新增 6 维相关：motion_anchor_pos_b(3) + base_lin_vel(3)
        # ------------------------------------------------------------
        print("[add6 / anchor / velocity]")
        print(f"  real_anchor_pos_w        = {self._format_vec(robot_pos_w, precision=4)}")
        print(f"  ref_anchor_pos_w(offset) = {self._format_vec(motionposcurrent_for_anchor, precision=4)}")
        print(f"  obs_motion_anchor_pos_b  = {self._format_vec(motion_anchor_pos_b, precision=4)} "
              f"|norm|={np.linalg.norm(motion_anchor_pos_b):.4f}")
        print(f"  obs_base_lin_vel_b       = {self._format_vec(base_lin_vel, precision=4)} "
              f"|norm|={np.linalg.norm(base_lin_vel):.4f}")
        print(f"  obs_base_ang_vel_b       = {self._format_vec(ang_vel, precision=4)} "
              f"|norm|={np.linalg.norm(ang_vel):.4f}")

        print("[motion command]")
        print(f"  reference_q_head         = {self._format_vec(motioninput[:6], precision=4)}")
        print(f"  reference_dq_head        = {self._format_vec(motioninput[29:35], precision=4)}")
        print(f"  anchor_offset            = {self._format_vec(self._motionpos_anchor_offset, precision=4)}")

        # ------------------------------------------------------------
        # 29 维 q 相关
        # ------------------------------------------------------------
        print("[q / dq summary]")
        print(
            f"  real_q_seq:  min={np.min(real_q_seq): .4f}, "
            f"max={np.max(real_q_seq): .4f}, "
            f"mean={np.mean(real_q_seq): .4f}, "
            f"std={np.std(real_q_seq): .4f}"
        )
        print(
            f"  real_dq_seq: min={np.min(real_dq_seq): .4f}, "
            f"max={np.max(real_dq_seq): .4f}, "
            f"mean={np.mean(real_dq_seq): .4f}, "
            f"std={np.std(real_dq_seq): .4f}"
        )

        target_q_seq = None
        if target_q_xml is not None:
            # target_q_xml 是 XML joint order，转成 policy joint_seq 顺序。
            target_q_xml = np.asarray(target_q_xml, dtype=np.float64).reshape(-1)
            target_q_seq = target_q_xml[self.seq_idx_in_xml]

        if ref_q_seq is not None:
            ref_q_seq = np.asarray(ref_q_seq, dtype=np.float64).reshape(-1)
            q_err = real_q_seq - ref_q_seq

            print(
                f"  ref_q_seq:   min={np.min(ref_q_seq): .4f}, "
                f"max={np.max(ref_q_seq): .4f}, "
                f"mean={np.mean(ref_q_seq): .4f}, "
                f"std={np.std(ref_q_seq): .4f}"
            )
            print(
                f"  q_err:       norm={np.linalg.norm(q_err): .4f}, "
                f"mean_abs={np.mean(np.abs(q_err)): .4f}, "
                f"max_abs={np.max(np.abs(q_err)): .4f}"
            )

            self._print_topk_q_error(
                real_q_seq=real_q_seq,
                ref_q_seq=ref_q_seq,
                target_q_seq=target_q_seq,
                k=8,
            )
        else:
            print("  ref_q_seq:   None. ONNX output 'joint_pos' not found.")

        if action is not None:
            action = np.asarray(action, dtype=np.float64).reshape(-1)
            print(
                f"[action] norm={np.linalg.norm(action):.4f}, "
                f"min={np.min(action): .4f}, "
                f"max={np.max(action): .4f}"
            )

        if target_q_seq is not None:
            print(
                f"[target_q_seq] min={np.min(target_q_seq): .4f}, "
                f"max={np.max(target_q_seq): .4f}, "
                f"mean={np.mean(target_q_seq): .4f}"
            )

        # 前 20 帧打印完整 29 维数组，后面只打印摘要和 top-k。
        if full_print:
            print("[full arrays, policy joint_seq order]")
            print(f"  real_q_seq = {self._format_vec(real_q_seq, precision=4)}")
            if ref_q_seq is not None:
                print(f"  ref_q_seq  = {self._format_vec(ref_q_seq, precision=4)}")
                print(f"  q_err_seq  = {self._format_vec(real_q_seq - ref_q_seq, precision=4)}")
            print(f"  real_dq_seq= {self._format_vec(real_dq_seq, precision=4)}")
            if ref_dq_seq is not None:
                ref_dq_seq = np.asarray(ref_dq_seq, dtype=np.float64).reshape(-1)
                print(f"  ref_dq_seq = {self._format_vec(ref_dq_seq, precision=4)}")
                print(f"  dq_err_seq = {self._format_vec(real_dq_seq - ref_dq_seq, precision=4)}")

        print("=" * 90 + "\n")
# =========== newADD end ========

# ======== newADD start======
    def compare_onnx_metadata_with_config(self):
        """
        对比 ONNX metadata 和实机 config 中的 default/action_scale。

        这个函数只打印，不改变控制逻辑。

        重点检查：
            1. ONNX default_joint_pos vs config.default_angles_seq；
            2. ONNX action_scale vs config.action_scale_seq；
            3. joint_names 是否和当前 joint_seq 一致。

        Returns:
            None.
        """
        try:
            import onnx
        except Exception as e:
            print(f"[ONNX Metadata Compare] skip because import onnx failed: {e}")
            return

        if not os.path.isfile(self.onnx):
            print(f"[ONNX Metadata Compare] skip because onnx file not found: {self.onnx}")
            return

        model = onnx.load(self.onnx)
        meta = {prop.key: prop.value for prop in model.metadata_props}

        print("[ONNX Metadata Compare]")

        if "joint_names" in meta:
            onnx_joint_names = meta["joint_names"].split(",")
            same_joint_names = list(onnx_joint_names) == list(joint_seq)
            print(f"  joint_names same as script joint_seq: {same_joint_names}")
            if not same_joint_names:
                print(f"  onnx_joint_names[:5] = {onnx_joint_names[:5]}")
                print(f"  script_joint_seq[:5] = {joint_seq[:5]}")
        else:
            onnx_joint_names = None
            print("  joint_names: not found in ONNX metadata")

        def _parse_float_array(key: str):
            if key not in meta:
                print(f"  {key}: not found in ONNX metadata")
                return None
            arr = np.array([float(x) for x in meta[key].split(",")], dtype=np.float64)
            print(f"  {key}: shape={arr.shape}, head={arr[:6]}")
            return arr

        onnx_default_seq = _parse_float_array("default_joint_pos")
        onnx_action_scale_seq = _parse_float_array("action_scale")
        _ = _parse_float_array("joint_stiffness")
        _ = _parse_float_array("joint_damping")

        def _print_top_diff(name: str, a: np.ndarray, b: np.ndarray, k: int = 8):
            a = np.asarray(a, dtype=np.float64).reshape(-1)
            b = np.asarray(b, dtype=np.float64).reshape(-1)
            diff = a - b

            print(f"  [{name}]")
            print(
                f"    diff_norm={np.linalg.norm(diff):.6f}, "
                f"mean_abs={np.mean(np.abs(diff)):.6f}, "
                f"max_abs={np.max(np.abs(diff)):.6f}"
            )

            idxs = np.argsort(np.abs(diff))[::-1][:k]
            for rank, idx in enumerate(idxs, start=1):
                joint_name = joint_seq[idx] if idx < len(joint_seq) else f"idx_{idx}"
                print(
                    f"    {rank:02d}. {joint_name:<28s} "
                    f"onnx={a[idx]: .6f}, "
                    f"config={b[idx]: .6f}, "
                    f"diff={diff[idx]: .6f}"
                )

        if onnx_default_seq is not None and hasattr(self.config, "default_angles_seq"):
            _print_top_diff(
                name="default_joint_pos_seq vs config.default_angles_seq",
                a=onnx_default_seq,
                b=self.config.default_angles_seq,
                k=8,
            )

        if onnx_action_scale_seq is not None and hasattr(self.config, "action_scale_seq"):
            _print_top_diff(
                name="action_scale_seq vs config.action_scale_seq",
                a=onnx_action_scale_seq,
                b=self.config.action_scale_seq,
                k=8,
            )
# =========== newADD end ========

    def LowStateHgHandler(self, msg: LowStateHG):
        self.low_state = msg
        self.mode_machine_ = self.low_state.mode_machine
        self.remote_controller.set(self.low_state.wireless_remote)

    def LowStateGoHandler(self, msg: LowStateGo):
        self.low_state = msg
        self.remote_controller.set(self.low_state.wireless_remote)

    def send_cmd(self, cmd: Union[LowCmdGo, LowCmdHG]):
        """
        通过 Unitree DDS 发送 low_cmd。

        注意：
            这个函数本身不判断 dry_run。
            所有流程里应通过 maybe_send_low_cmd() 发送，避免漏保护。
        """
        if dry_run or self.lowcmd_publisher_ is None:
            return
        cmd.crc = CRC().Crc(cmd)
        self.lowcmd_publisher_.Write(cmd)

    def rosbridge_send_cmd(self, cmd):
        """
        将 low_cmd 转换为 ROS RobotCommand 并通过 /robot/command 发送。

        仅 use_ros_cmd=True 且 dry_run=False 时使用。
        """
        q_des = [0.0] * self.config.num_actions
        kp = [0.0] * self.config.num_actions
        kd = [0.0] * self.config.num_actions
        tau = [0.0] * self.config.num_actions

        for i in range(self.config.num_actions):
            q_des[i] = float(cmd.motor_cmd[i].q)
            kp[i] = float(cmd.motor_cmd[i].kp)
            kd[i] = float(cmd.motor_cmd[i].kd)
            tau[i] = float(cmd.motor_cmd[i].tau)

        self.ros.publish_cmd(q_des=q_des, kp=kp, kd=kd, tau_ff=tau)

    # ======== newADD start======
    def maybe_send_low_cmd(self):
        """
        根据 dry_run 和 use_ros_cmd 决定是否发送 low_cmd。

        dry_run=True:
            完全不发控制命令。
        use_ros_cmd=True:
            通过 /robot/command 发。
        use_ros_cmd=False:
            通过 Unitree DDS self.send_cmd 发。
        """
        if dry_run:
            return

        if use_ros_cmd:
            self.rosbridge_send_cmd(self.low_cmd)
        else:
            self.send_cmd(self.low_cmd)
    # =========== newADD end ========

    def wait_for_low_state(self):
        while self.low_state.tick == 0:
            time.sleep(self.config.control_dt)
        print("Successfully connected to the robot.")

    def zero_torque_state(self):
        print("Enter zero torque state.")
        print("Waiting for the start signal...")
        while self.remote_controller.button[KeyMap.start] != 1:
            create_zero_cmd(self.low_cmd)
            self.maybe_send_low_cmd()
            time.sleep(self.config.control_dt)

    def wait_for_obsrun_ready(self, timeout_s: float = 300.0):
        """等待 OBSRun 时持续保持默认姿态，避免 LowCmd 超时后变软。"""
        print("Robot is standing. Start OBSRun now; holding default pose while waiting for /Odometry_2...")

        if use_ros_state:
            state_deadline = time.monotonic() + 20.0
            while not self.ros.wait_state(timeout_s=0.0):
                self.hold_default_pos()
                if time.monotonic() >= state_deadline:
                    raise RuntimeError(
                        "Timeout waiting /robot/state. Is C++ unitree_io_node running?"
                    )
                time.sleep(self.config.control_dt)
            print("ROS bridge ready: /robot/state rx ok.")
        else:
            print("use_ros_state=False: using Unitree DDS low_state for q/dq/imu.")

        odom_deadline = time.monotonic() + timeout_s
        next_status_t = time.monotonic()
        while not self.ros.wait_odom(timeout_s=0.0):
            self.hold_default_pos()
            now = time.monotonic()
            if now >= odom_deadline:
                raise RuntimeError(
                    "Timeout waiting /Odometry_2. Please check OBSRun and: ros2 topic hz /Odometry_2"
                )
            if now >= next_status_t:
                print(
                    "[WAIT /Odometry_2] Holding default pose; "
                    f"{odom_deadline - now:.0f} s remaining."
                )
                next_status_t = now + 5.0
            time.sleep(self.config.control_dt)
        print("OBSRun ready: /Odometry_2 rx ok; default pose hold complete.")

    def hold_default_pos(self):
        """用 YAML 原有 default_angles/stiffness/damping 发送一帧保持命令。"""
        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        for j, motor_idx in enumerate(dof_idx):
            motor_cmd = self.low_cmd.motor_cmd[motor_idx]
            motor_cmd.q = self.config.default_angles[j]
            motor_cmd.dq = 0
            motor_cmd.kp = self.config.stiffness[j]
            motor_cmd.kd = self.config.damping[j]
            motor_cmd.tau = 0
        self.maybe_send_low_cmd()

    def move_to_default_pos(self):
        print("Moving to default pos.")

        total_time = 6
        num_step = int(total_time / self.config.control_dt)

        dof_idx = self.config.leg_joint2motor_idx + self.config.arm_waist_joint2motor_idx
        kps = self.config.stiffness
        kds = self.config.damping
        default_pos = self.config.default_angles.copy()
        dof_size = len(dof_idx)

        init_dof_pos = np.zeros(dof_size, dtype=np.float32)
        for i in range(dof_size):
            init_dof_pos[i] = self.low_state.motor_state[dof_idx[i]].q

        for i in range(num_step):
            alpha = i / num_step
            for j in range(dof_size):
                motor_idx = dof_idx[j]
                target_pos = default_pos[j]
                self.low_cmd.motor_cmd[motor_idx].q = (
                    init_dof_pos[j] * (1 - alpha) + target_pos * alpha
                )
                self.low_cmd.motor_cmd[motor_idx].dq = 0
                self.low_cmd.motor_cmd[motor_idx].kp = kps[j]
                self.low_cmd.motor_cmd[motor_idx].kd = kds[j]
                self.low_cmd.motor_cmd[motor_idx].tau = 0

            self.maybe_send_low_cmd()
            time.sleep(self.config.control_dt)

# ======== newADD start======
    def print_pre_a_yaw_alignment(self):
        """
        在 default_pos_state 等待 A 的阶段，低频打印机器人当前 odom yaw 和参考 motion yaw 的误差。

        这个函数只用于人工摆放机器人朝向，不改变控制、不改变 obs、不改变 policy。
        目前只使用 /Odometry_2 的 yaw，不使用 IMU yaw。
        """
        now = time.perf_counter()
        if now - self._pre_a_yaw_print_t < self._pre_a_yaw_print_period_s:
            return

        self._pre_a_yaw_print_t = now

        robot_pos_w, robot_quat_odom_wxyz, base_lin_vel, odom_ok = self.ros.get_odom_copy(
            max_age_s=0.5
        )

        if not odom_ok:
            print("[PRE-A YAW ALIGN] /Odometry_2 missing or stale, cannot compute yaw alignment.")
            return

        robot_yaw_odom_deg = quat_wxyz_to_yaw_deg(robot_quat_odom_wxyz)
        yaw_err_odom_deg = wrap_angle_deg(robot_yaw_odom_deg - self.motion_yaw0_deg)

        print(
            "[PRE-A YAW ALIGN] "
            f"motion_yaw0={self.motion_yaw0_deg:8.3f} deg | "
            f"robot_yaw_odom={robot_yaw_odom_deg:8.3f} deg | "
            f"yaw_err_odom={yaw_err_odom_deg:8.3f} deg | "
            f"odom_pos={np.array2string(robot_pos_w, precision=3, suppress_small=True)}"
        )

        if abs(yaw_err_odom_deg) < 5.0:
            print("  -> heading OK: yaw_err_odom_deg is within 5 deg.")
        elif abs(abs(yaw_err_odom_deg) - 90.0) < 15.0:
            print("  -> warning: robot is roughly sideways relative to the reference heading.")
        elif abs(abs(yaw_err_odom_deg) - 180.0) < 15.0:
            print("  -> warning: robot is roughly facing the opposite direction.")
# =========== newADD end ========

    def default_pos_state(self):
        print("Enter default pos state.")
        print("Waiting for the Button A signal...")

        while self.remote_controller.button[KeyMap.A] != 1:
            for i in range(len(self.config.leg_joint2motor_idx)):
                motor_idx = self.config.leg_joint2motor_idx[i]
                self.low_cmd.motor_cmd[motor_idx].q = self.config.default_angles[i]
                self.low_cmd.motor_cmd[motor_idx].dq = 0
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
