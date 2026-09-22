"""
Usage:
cd ~/BysanRL/scripts
conda activate bydmmc_nogmr
   
python play_npz_csv_smpl_mujoco_fk_isaaclab.py \
    --motion_file /data/BysanRL/data_goal/npz/GVHMR_gmr/demo/happy/happy_gmr_50fps.npz \
    --smpl_csv /data/BysanRL/data_goal/csv/GVHMR/demo/happy/happy_smpl_50fps.csv \
    --smpl_mjcf /data/BysanRL/data_goal/xml/smpl_humanoid_zup.xml \
    --start 0 \
    --speed 1.0 \
    --smpl_xyz_offset 0 0 0.05 \
    --root_frame_length 0.35 \
    --root_frame_line_width 4.0 \
    --convert_csv_rot_to_zup True \
    --convert_csv_transl_to_zup False \
    --root_rot_mode asset_basis \
    --smpl_world_alignment gvhmr_gmr_root0 \
    --smpl_frame_length 0.07 \
    --smpl_frame_line_width 5.0 \
    --draw_smpl_all_frames True \
    
    --print_theory_odom True \
    --print_theory_odom_every 25 \
    --print_theory_odom_compare_smpl True

Purpose:
    在同一个 IsaacLab world 里播放 G1 NPZ，并用 MuJoCo 直接计算 SMPL z_up MJCF + SMPL CSV 的 FK。

    重要区别：
        1. G1 仍然是 IsaacLab Articulation；
        2. SMPL 不再通过 IsaacLab MjcfConverter 转 USD；
        3. SMPL 不再是 IsaacLab/PhysX Articulation；
        4. SMPL FK 由 MuJoCo 原生 mj_forward() 计算；
        5. IsaacLab viewer 只用 debug_draw 绘制 SMPL skeleton 和 body/root frame。

Why:
    你已经验证过：
        MuJoCo-only 播放 smpl_humanoid_zup.xml + SMPL CSV 是正常的；
        但 MJCF -> USD -> IsaacLab Articulation 后 body frame 不正常。

    因此本脚本绕开 IsaacLab 的 SMPL articulation，只把 MuJoCo 正确 FK 结果画到 IsaacLab viewer 中，
    用于观察 NPZ 机器人和 CSV-SMPL 是否处于同一个世界坐标系。
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import mujoco
import numpy as np
import pandas as pd
import torch
from scipy.spatial.transform import Rotation as R

from isaaclab.app import AppLauncher


# ============================================================
# 0. 命令行参数：必须放在 AppLauncher 启动前
# ============================================================

def str2bool(v):
    """
    解析命令行布尔值。

    详细说明：
        argparse 如果直接使用 bool，会把几乎所有非空字符串都解析为 True。
        因此这里显式支持 true/false、1/0、yes/no 等写法。

    Args:
        v:
            命令行输入值。

    Returns:
        bool:
            解析后的布尔值。

    Raises:
        argparse.ArgumentTypeError:
            当输入不是合法布尔值时抛出。
    """
    if isinstance(v, bool):
        return v

    value = str(v).strip().lower()

    if value in ("true", "1", "yes", "y", "t"):
        return True
    if value in ("false", "0", "no", "n", "f"):
        return False

    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")


parser = argparse.ArgumentParser(
    description="Play G1 NPZ in IsaacLab and draw SMPL CSV using MuJoCo FK in the same world."
)

parser.add_argument(
    "--motion_file",
    "--npz",
    "--npz_file",
    dest="motion_file",
    type=str,
    required=True,
    help="G1 reference NPZ file.",
)
parser.add_argument(
    "--smpl_csv",
    type=str,
    required=True,
    help="SMPL CSV file.",
)
parser.add_argument(
    "--smpl_mjcf",
    type=str,
    required=True,
    help="Z-up SMPL humanoid MJCF XML file. This file is loaded by MuJoCo directly.",
)
parser.add_argument(
    "--start",
    type=int,
    default=0,
    help="Start frame index.",
)
parser.add_argument(
    "--end",
    type=int,
    default=-1,
    help="End frame index. -1 means min(npz_len, csv_len).",
)
parser.add_argument(
    "--speed",
    type=float,
    default=1.0,
    help="Frame increment per render loop. 1.0 means one frame per loop.",
)
parser.add_argument(
    "--num_envs",
    type=int,
    default=1,
    help="Number of IsaacLab environments. This script is intended for 1 env.",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=2.0,
    help="IsaacLab env spacing.",
)
parser.add_argument(
    "--npz_root_body_idx",
    type=int,
    default=0,
    help="Root body index in NPZ body_pos_w/body_quat_w.",
)
parser.add_argument(
    "--root_frame_length",
    type=float,
    default=0.35,
    help="G1 root coordinate frame axis length.",
)
parser.add_argument(
    "--root_frame_line_width",
    type=float,
    default=4.0,
    help="G1 root coordinate frame line width.",
)
parser.add_argument(
    "--smpl_frame_length",
    type=float,
    default=0.16,
    help="SMPL body coordinate frame axis length for debug_draw.",
)
parser.add_argument(
    "--smpl_frame_line_width",
    type=float,
    default=2.0,
    help="SMPL body coordinate frame line width for debug_draw.",
)
parser.add_argument(
    "--smpl_skeleton_line_width",
    type=float,
    default=3.0,
    help="SMPL skeleton line width for debug_draw.",
)
parser.add_argument(
    "--smpl_xyz_offset",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    help="Extra world-frame xyz offset added to SMPL root position after CSV transl processing.",
)
parser.add_argument(
    "--convert_csv_rot_to_zup",
    type=str2bool,
    default=True,
    help=(
        "Whether to convert SMPL CSV rotations from old SMPL local convention "
        "(X=left, Y=up, Z=forward) to robot/Z-up convention "
        "(X=forward, Y=left, Z=up)."
    ),
)
parser.add_argument(
    "--convert_csv_transl_to_zup",
    type=str2bool,
    default=False,
    help=(
        "Whether to convert CSV transl by new=[old_z, old_x, old_y]. "
        "Usually keep False if raw CSV transl already played correctly in MuJoCo z_up MJCF."
    ),
)
parser.add_argument(
    "--root_rot_mode",
    type=str,
    default="asset_basis",
    choices=["asset_basis", "similarity", "none"],
    help=(
        "How to convert CSV global_orient for z_up SMPL asset. "
        "asset_basis: R_new = R_old @ P.T; "
        "similarity: R_new = P @ R_old @ P.T; "
        "none: R_new = R_old."
    ),
)
parser.add_argument(
    "--smpl_world_alignment",
    type=str,
    default="gvhmr_gmr_root0",
    choices=["none", "gvhmr_gmr_root0"],
    help=(
        "Optional world alignment applied after SMPL FK. gvhmr_gmr_root0 rotates the "
        "project Z-up CSV world into the dedicated GVHMR-GMR world and aligns the "
        "first SMPL pelvis XY with the first robot root XY."
    ),
)
parser.add_argument(
    "--max_render_loops",
    type=int,
    default=-1,
    help="Stop after N render loops for automated checks. -1 keeps playing.",
)
parser.add_argument(
    "--draw_smpl_frames",
    type=str2bool,
    default=True,
    help="Whether to draw SMPL body frames from MuJoCo FK.",
)
parser.add_argument(
    "--draw_smpl_skeleton",
    type=str2bool,
    default=True,
    help="Whether to draw SMPL skeleton segments from MuJoCo FK.",
)
parser.add_argument(
    "--draw_smpl_all_frames",
    type=str2bool,
    default=False,
    help="If True, draw frames for all SMPL_DRAW_BODIES; otherwise only root and lower-body frames.",
)

# Compatibility-only arguments: accepted but ignored, so old commands do not crash.
parser.add_argument(
    "--smpl_usd_dir",
    type=str,
    default="",
    help="Ignored in this MuJoCo-FK script. Kept only for old command compatibility.",
)
parser.add_argument(
    "--force_smpl_usd_conversion",
    action="store_true",
    default=False,
    help="Ignored in this MuJoCo-FK script. Kept only for old command compatibility.",
)
parser.add_argument(
    "--smpl_converter_fix_base",
    type=str2bool,
    default=False,
    help="Ignored in this MuJoCo-FK script. Kept only for old command compatibility.",
)

# ======== newADD start======
parser.add_argument(
    "--print_theory_odom",
    type=str2bool,
    default=False,
    help="Whether to print theoretical real odometry derived from the current NPZ root state.",
)
parser.add_argument(
    "--print_theory_odom_every",
    type=int,
    default=25,
    help="Print theoretical odometry every N playback loops. <=0 disables printing.",
)
parser.add_argument(
    "--print_theory_odom_compare_smpl",
    type=str2bool,
    default=True,
    help="Whether to also print SMPL Pelvis FK for reference comparison.",
)
# =========== newADD end ========

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.num_envs != 1:
    print("[WARN] This script is intended for --num_envs 1. SMPL MuJoCo FK is drawn only once.")

if args_cli.speed <= 0.0:
    raise ValueError(f"--speed must be positive, got {args_cli.speed}")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


# ============================================================
# 1. IsaacLab imports：必须放在 AppLauncher 启动后
# ============================================================

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader

try:
    from isaacsim.util.debug_draw import _debug_draw
except Exception:
    try:
        from omni.isaac.debug_draw import _debug_draw
    except Exception:
        _debug_draw = None


# ============================================================
# 2. 坐标和 SMPL/MJCF 映射定义
# ============================================================

# old SMPL convention:
#   old X = left
#   old Y = up
#   old Z = forward
#
# new robot/Z-up convention:
#   new X = forward
#   new Y = left
#   new Z = up
#
# Therefore:
#   v_new = P @ v_old = [old_z, old_x, old_y]
P_OLD_SMPL_TO_NEW_ZUP = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)

# Public GVHMR CSV uses P_OLD_SMPL_TO_NEW_ZUP, while the dedicated GMR
# preprocessor uses Q=[x,-z,y]. Therefore A=Q@P.T is a -90 degree yaw.
R_PROJECT_ZUP_TO_GVHMR_GMR = np.array(
    [
        [0.0, 1.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
    ],
    dtype=np.float64,
)

SMPL_23_JOINTS = [
    "L_Hip", "R_Hip", "Spine1",
    "L_Knee", "R_Knee", "Spine2",
    "L_Ankle", "R_Ankle", "Spine3",
    "L_Foot", "R_Foot",
    "Neck", "L_Collar", "R_Collar", "Head",
    "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
    "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
]

SMPL_TO_MJCF = {
    "L_Hip": "L_Hip",
    "R_Hip": "R_Hip",
    "Spine1": "Torso",
    "L_Knee": "L_Knee",
    "R_Knee": "R_Knee",
    "Spine2": "Spine",
    "L_Ankle": "L_Ankle",
    "R_Ankle": "R_Ankle",
    "Spine3": "Chest",
    "L_Foot": "L_Toe",
    "R_Foot": "R_Toe",
    "Neck": "Neck",
    "L_Collar": "L_Thorax",
    "R_Collar": "R_Thorax",
    "Head": "Head",
    "L_Shoulder": "L_Shoulder",
    "R_Shoulder": "R_Shoulder",
    "L_Elbow": "L_Elbow",
    "R_Elbow": "R_Elbow",
    "L_Wrist": "L_Wrist",
    "R_Wrist": "R_Wrist",
    "L_Hand": None,
    "R_Hand": None,
}

SMPL_DRAW_BODIES = [
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
    "Torso", "Spine", "Chest", "Neck", "Head",
    "L_Thorax", "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Thorax", "R_Shoulder", "R_Elbow", "R_Wrist",
]

SMPL_FRAME_BODIES_DEFAULT = [
    "Pelvis",
    "L_Hip", "L_Knee", "L_Ankle", "L_Toe",
    "R_Hip", "R_Knee", "R_Ankle", "R_Toe",
]

SMPL_DRAW_SEGMENTS = [
    ("Pelvis", "L_Hip"),
    ("L_Hip", "L_Knee"),
    ("L_Knee", "L_Ankle"),
    ("L_Ankle", "L_Toe"),
    ("Pelvis", "R_Hip"),
    ("R_Hip", "R_Knee"),
    ("R_Knee", "R_Ankle"),
    ("R_Ankle", "R_Toe"),
    ("Pelvis", "Torso"),
    ("Torso", "Spine"),
    ("Spine", "Chest"),
    ("Chest", "Neck"),
    ("Neck", "Head"),
    ("Chest", "L_Thorax"),
    ("L_Thorax", "L_Shoulder"),
    ("L_Shoulder", "L_Elbow"),
    ("L_Elbow", "L_Wrist"),
    ("Chest", "R_Thorax"),
    ("R_Thorax", "R_Shoulder"),
    ("R_Shoulder", "R_Elbow"),
    ("R_Elbow", "R_Wrist"),
]


# ============================================================
# 3. 旋转工具
# ============================================================

def quat_wxyz_to_rotmat(q_wxyz: np.ndarray) -> np.ndarray:
    """
    将 wxyz quaternion 转成旋转矩阵。

    Args:
        q_wxyz:
            shape=(4,)，wxyz quaternion。

    Returns:
        np.ndarray:
            shape=(3, 3)，旋转矩阵，每一列分别是 local X/Y/Z 在 world 中的方向。
    """
    q = np.asarray(q_wxyz, dtype=np.float64).reshape(4)
    norm = np.linalg.norm(q)

    if norm < 1e-12:
        return np.eye(3, dtype=np.float64)

    q = q / norm
    q_xyzw = np.array([q[1], q[2], q[3], q[0]], dtype=np.float64)
    return R.from_quat(q_xyzw).as_matrix()


def rotmat_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为 wxyz quaternion。

    Args:
        rot_mat:
            shape=(3, 3)，旋转矩阵。

    Returns:
        np.ndarray:
            shape=(4,)，wxyz quaternion。
    """
    quat_xyzw = R.from_matrix(np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array(
        [quat_xyzw[3], quat_xyzw[0], quat_xyzw[1], quat_xyzw[2]],
        dtype=np.float64,
    )


def transform_smpl_fk_state(
    state: Dict[str, Tuple[np.ndarray, np.ndarray]],
    world_rotation: np.ndarray,
    world_translation: np.ndarray,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Apply one rigid world transform to all SMPL FK bodies."""
    world_rotation = np.asarray(world_rotation, dtype=np.float64).reshape(3, 3)
    world_translation = np.asarray(world_translation, dtype=np.float64).reshape(3)
    transformed: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for body_name, (position, quat_wxyz) in state.items():
        position_new = world_rotation @ np.asarray(position, dtype=np.float64) + world_translation
        rotation_new = world_rotation @ quat_wxyz_to_rotmat(quat_wxyz)
        transformed[body_name] = (position_new, rotmat_to_quat_wxyz(rotation_new))
    return transformed


def convert_root_rotvec_by_mode_to_quat_wxyz(rotvec: np.ndarray, mode: str) -> np.ndarray:
    """
    按模式转换 CSV global_orient，并输出 wxyz quaternion。

    Args:
        rotvec:
            shape=(3,)，CSV global_orient rotvec。
        mode:
            root rotation conversion mode:
                asset_basis:
                    R_new = R_old @ P.T
                similarity:
                    R_new = P @ R_old @ P.T
                none:
                    R_new = R_old

    Returns:
        np.ndarray:
            shape=(4,)，wxyz quaternion。
    """
    p = P_OLD_SMPL_TO_NEW_ZUP
    r_old = R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()

    if mode == "asset_basis":
        r_new = r_old @ p.T
    elif mode == "similarity":
        r_new = p @ r_old @ p.T
    elif mode == "none":
        r_new = r_old
    else:
        raise ValueError(f"Unsupported root_rot_mode: {mode}")

    return rotmat_to_quat_wxyz(r_new)


def convert_body_rotvec_old_smpl_to_zup_xyz_euler(rotvec: np.ndarray) -> np.ndarray:
    """
    将 CSV body_pose 局部 rotvec 从 old SMPL 局部基转换到 z_up 资产局部基。

    公式：
        R_joint_new = P @ R_joint_old @ P.T

    Args:
        rotvec:
            shape=(3,)，CSV 原始 body_pose rotvec。

    Returns:
        np.ndarray:
            shape=(3,)，转换后的 XYZ Euler，单位 rad。
    """
    p = P_OLD_SMPL_TO_NEW_ZUP
    r_old = R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()
    r_new = p @ r_old @ p.T
    return R.from_matrix(r_new).as_euler("XYZ", degrees=False)


def convert_body_rotvec_to_xyz_euler_raw(rotvec: np.ndarray) -> np.ndarray:
    """
    将 body_pose 原始 rotvec 直接转成 XYZ Euler。

    Args:
        rotvec:
            shape=(3,)，axis-angle / rotation vector。

    Returns:
        np.ndarray:
            shape=(3,)，XYZ Euler，单位 rad。
    """
    return R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_euler(
        "XYZ",
        degrees=False,
    )


def convert_transl_old_smpl_to_zup(transl: np.ndarray) -> np.ndarray:
    """
    可选：将 CSV transl 从 old SMPL 坐标排列转换到 new z_up 坐标排列。

    Args:
        transl:
            shape=(3,)，CSV 原始 transl。

    Returns:
        np.ndarray:
            shape=(3,)，new=[old_z, old_x, old_y] 后的 transl。
    """
    return P_OLD_SMPL_TO_NEW_ZUP @ np.asarray(transl, dtype=np.float64).reshape(3)


# ============================================================
# 4. SMPL CSV 数据结构与读取
# ============================================================

@dataclass
class SmplCsvMotion:
    """
    SMPL CSV motion 容器。

    Attributes:
        transl:
            shape=(T, 3)，CSV 原始 root translation。
        global_orient:
            shape=(T, 3)，CSV 原始 root orientation rotvec。
        body_pose:
            shape=(T, 23, 3)，CSV 原始 body pose rotvec。
    """
    transl: np.ndarray
    global_orient: np.ndarray
    body_pose: np.ndarray

    @property
    def num_frames(self) -> int:
        """
        返回 SMPL CSV 帧数。
        """
        return int(self.transl.shape[0])


def load_smpl_csv(csv_path: str) -> SmplCsvMotion:
    """
    读取 SMPL CSV，只提取 transl/global_orient/body_pose。

    Args:
        csv_path:
            SMPL CSV 路径。

    Returns:
        SmplCsvMotion:
            读取后的 SMPL motion。

    Raises:
        FileNotFoundError:
            CSV 不存在时抛出。
        ValueError:
            CSV 为空或 body_pose 维度不合法时抛出。
    """
    path = Path(csv_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"SMPL CSV does not exist: {path}")

    df = pd.read_csv(path)

    if len(df) == 0:
        raise ValueError(f"SMPL CSV is empty: {path}")

    transl = df[
        [
            "smpl_params_global_transl_0",
            "smpl_params_global_transl_1",
            "smpl_params_global_transl_2",
        ]
    ].to_numpy(dtype=np.float64)

    global_orient = df[
        [
            "smpl_params_global_global_orient_0",
            "smpl_params_global_global_orient_1",
            "smpl_params_global_global_orient_2",
        ]
    ].to_numpy(dtype=np.float64)

    body_cols = [
        col for col in df.columns
        if col.startswith("smpl_params_global_body_pose_")
    ]
    body_indices = sorted(
        int(col.replace("smpl_params_global_body_pose_", ""))
        for col in body_cols
    )

    if len(body_indices) == 0:
        raise ValueError("No SMPL body_pose columns found.")

    body_dim = body_indices[-1] + 1

    if body_indices != list(range(body_dim)):
        raise ValueError("body_pose columns must be contiguous from 0 to N-1.")

    if body_dim not in (63, 69):
        raise ValueError(f"Unsupported SMPL body_pose dim: {body_dim}. Expected 63 or 69.")

    body_pose_flat = df[
        [f"smpl_params_global_body_pose_{i}" for i in range(body_dim)]
    ].to_numpy(dtype=np.float64)

    if body_dim == 63:
        body_pose_flat = np.concatenate(
            [body_pose_flat, np.zeros((len(df), 6), dtype=np.float64)],
            axis=1,
        )

    body_pose = body_pose_flat.reshape(-1, 23, 3)

    print("=" * 80)
    print("[SMPL CSV]")
    print(f"  path      = {path}")
    print(f"  frames    = {len(df)}")
    print(f"  body_dim  = {body_dim} -> {body_pose.shape[1] * 3}")
    print("=" * 80)

    return SmplCsvMotion(
        transl=transl,
        global_orient=global_orient,
        body_pose=body_pose,
    )


# ============================================================
# 5. MuJoCo SMPL FK
# ============================================================

class MuJoCoSmplFK:
    """
    使用 MuJoCo 计算 SMPL z_up XML 的正运动学。

    详细说明：
        该类只负责：
            1. 加载 smpl_humanoid_zup.xml；
            2. 每帧写入 SMPL CSV root/body_pose；
            3. 调用 mujoco.mj_forward()；
            4. 输出 body 的 world position / quaternion。

        它不依赖 IsaacLab 的 SMPL Articulation，也不依赖 MJCF -> USD converter。
    """

    def __init__(
        self,
        mjcf_path: str,
        smpl_motion: SmplCsvMotion,
        smpl_xyz_offset=(0.0, 0.0, 0.0),
        convert_csv_rot_to_zup: bool = True,
        convert_csv_transl_to_zup: bool = False,
        root_rot_mode: str = "asset_basis",
    ):
        """
        初始化 MuJoCo SMPL FK 计算器。

        Args:
            mjcf_path:
                z_up SMPL MJCF XML 路径。
            smpl_motion:
                SMPL CSV motion。
            smpl_xyz_offset:
                额外世界坐标平移偏置。
            convert_csv_rot_to_zup:
                是否把 CSV rotation 从 old SMPL convention 转到 z_up convention。
            convert_csv_transl_to_zup:
                是否把 CSV transl 从 old 坐标排列转到 z_up 坐标排列。
            root_rot_mode:
                root global_orient 转换模式。
        """
        self.mjcf_path = str(Path(mjcf_path).expanduser().resolve())
        self.model = mujoco.MjModel.from_xml_path(self.mjcf_path)
        self.data = mujoco.MjData(self.model)
        self.smpl_motion = smpl_motion
        self.smpl_xyz_offset = np.asarray(smpl_xyz_offset, dtype=np.float64).reshape(3)
        self.convert_csv_rot_to_zup = bool(convert_csv_rot_to_zup)
        self.convert_csv_transl_to_zup = bool(convert_csv_transl_to_zup)
        self.root_rot_mode = str(root_rot_mode)

        self.body_name_to_id: Dict[str, int] = {}
        for name in SMPL_DRAW_BODIES:
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                self.body_name_to_id[name] = int(bid)

        self.joint_name_to_qadr: Dict[str, int] = {}
        for smpl_name, base_name in SMPL_TO_MJCF.items():
            if base_name is None:
                continue

            for suffix in ["x", "y", "z"]:
                joint_name = f"{base_name}_{suffix}"
                jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
                if jid >= 0:
                    self.joint_name_to_qadr[joint_name] = int(self.model.jnt_qposadr[jid])

        pelvis_jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, "Pelvis")
        self.pelvis_qadr = None
        if pelvis_jid >= 0 and self.model.jnt_type[pelvis_jid] == mujoco.mjtJoint.mjJNT_FREE:
            self.pelvis_qadr = int(self.model.jnt_qposadr[pelvis_jid])

        print("=" * 80)
        print("[MuJoCo SMPL FK]")
        print(f"  mjcf_path                = {self.mjcf_path}")
        print(f"  nq                       = {self.model.nq}")
        print(f"  nv                       = {self.model.nv}")
        print(f"  tracked bodies            = {len(self.body_name_to_id)} / {len(SMPL_DRAW_BODIES)}")
        print(f"  tracked scalar joints     = {len(self.joint_name_to_qadr)}")
        print(f"  pelvis_qadr              = {self.pelvis_qadr}")
        print(f"  convert_csv_rot_to_zup    = {self.convert_csv_rot_to_zup}")
        print(f"  convert_csv_transl_to_zup = {self.convert_csv_transl_to_zup}")
        print(f"  root_rot_mode             = {self.root_rot_mode}")
        print(f"  smpl_xyz_offset           = {self.smpl_xyz_offset}")
        print("=" * 80)

    def _set_root(self, frame_idx: int):
        """
        写入 Pelvis freejoint root。

        Args:
            frame_idx:
                当前帧编号。
        """
        if self.pelvis_qadr is None:
            return

        if self.convert_csv_transl_to_zup:
            pos = convert_transl_old_smpl_to_zup(self.smpl_motion.transl[frame_idx])
        else:
            pos = self.smpl_motion.transl[frame_idx]

        # ======== newADD start======
        # 这里不要加 smpl_xyz_offset。
        # root qpos 只保留 CSV transl 经过可选坐标转换后的结果；
        # smpl_xyz_offset 作为最终显示空间偏移，在 forward() 输出 body xpos 后再加。
        pos = np.asarray(pos, dtype=np.float64).reshape(3)
        # =========== newADD end ========

        if self.convert_csv_rot_to_zup:
            quat_wxyz = convert_root_rotvec_by_mode_to_quat_wxyz(
                self.smpl_motion.global_orient[frame_idx],
                mode=self.root_rot_mode,
            )
        else:
            r_raw = R.from_rotvec(
                np.asarray(self.smpl_motion.global_orient[frame_idx], dtype=np.float64).reshape(3)
            ).as_matrix()
            quat_wxyz = rotmat_to_quat_wxyz(r_raw)

        qadr = self.pelvis_qadr
        self.data.qpos[qadr:qadr + 3] = pos
        self.data.qpos[qadr + 3:qadr + 7] = quat_wxyz

    def _set_body_pose(self, frame_idx: int):
        """
        写入所有 SMPL body_pose joint。

        Args:
            frame_idx:
                当前帧编号。
        """
        body_pose = self.smpl_motion.body_pose[frame_idx]

        for smpl_idx, smpl_name in enumerate(SMPL_23_JOINTS):
            base_name = SMPL_TO_MJCF.get(smpl_name)
            if base_name is None:
                continue

            if self.convert_csv_rot_to_zup:
                angles_xyz = convert_body_rotvec_old_smpl_to_zup_xyz_euler(
                    body_pose[smpl_idx]
                )
            else:
                angles_xyz = convert_body_rotvec_to_xyz_euler_raw(
                    body_pose[smpl_idx]
                )

            for axis_i, suffix in enumerate(["x", "y", "z"]):
                joint_name = f"{base_name}_{suffix}"
                qadr = self.joint_name_to_qadr.get(joint_name)
                if qadr is None:
                    continue
                self.data.qpos[qadr] = float(angles_xyz[axis_i])

    def forward(self, frame_idx: int) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
        """
        计算某一帧的 SMPL FK。

        Args:
            frame_idx:
                当前帧编号。

        Returns:
            Dict[str, Tuple[np.ndarray, np.ndarray]]:
                body_name -> (position, quaternion_wxyz)
        """
        frame_idx = int(frame_idx) % self.smpl_motion.num_frames

        self.data.qpos[:] = 0.0
        self.data.qvel[:] = 0.0

        self._set_root(frame_idx)
        self._set_body_pose(frame_idx)

        mujoco.mj_forward(self.model, self.data)

        out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}

        # ======== newADD start======
        for body_name, bid in self.body_name_to_id.items():
            # MuJoCo FK 得到的是未显示偏移的 world position。
            # smpl_xyz_offset 在这里作为最终显示空间偏移统一加到所有 body 上。
            pos = (
                np.asarray(self.data.xpos[bid], dtype=np.float64).copy()
                + self.smpl_xyz_offset
            )

            rot = np.asarray(
                self.data.xmat[bid],
                dtype=np.float64,
            ).reshape(3, 3).copy()

            quat = rotmat_to_quat_wxyz(rot)
            out[body_name] = (pos, quat)
        # =========== newADD end ========

        return out


# ============================================================
# 6. IsaacLab 场景：只保留 G1，不加载 SMPL USD
# ============================================================

@configclass
class MinimalSceneCfg(InteractiveSceneCfg):
    """最小场景：地面、光照、G1。SMPL 由 MuJoCo FK + debug_draw 显示。"""

    ground = AssetBaseCfg(
        prim_path="/World/defaultGroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(
        prim_path="{ENV_REGEX_NS}/Robot"
    )


# ============================================================
# 7. 写入 G1 状态
# ============================================================

def write_g1_frame(
    robot: Articulation,
    motion,
    frame_idx: int,
    scene: InteractiveScene,
) -> torch.Tensor:
    """
    将 NPZ 中的一帧 G1 状态写入 IsaacLab。

    Args:
        robot:
            G1 articulation。
        motion:
            MotionLoader 读取出的 NPZ motion。
        frame_idx:
            当前帧。
        scene:
            IsaacLab scene。

    Returns:
        torch.Tensor:
            写入后的 G1 root_states，shape=(num_envs, 13)。
    """
    device = robot.device
    frame_ids = torch.full(
        (scene.num_envs,),
        int(frame_idx),
        dtype=torch.long,
        device=device,
    )

    root_states = robot.data.default_root_state.clone()

    root_states[:, :3] = motion.body_pos_w[frame_ids][:, args_cli.npz_root_body_idx] + scene.env_origins
    root_states[:, 3:7] = motion.body_quat_w[frame_ids][:, args_cli.npz_root_body_idx]
    root_states[:, 7:10] = motion.body_lin_vel_w[frame_ids][:, args_cli.npz_root_body_idx]
    root_states[:, 10:13] = motion.body_ang_vel_w[frame_ids][:, args_cli.npz_root_body_idx]

    robot.write_root_state_to_sim(root_states)
    robot.write_joint_state_to_sim(
        motion.joint_pos[frame_ids],
        motion.joint_vel[frame_ids],
    )

    return root_states


# ============================================================
# 8. 绘制工具
# ============================================================

def draw_frame_torch(debug_draw, root_state: torch.Tensor, length: float, width: float, alpha: float):
    """
    绘制 torch root_state 表示的坐标系。

    Args:
        debug_draw:
            Isaac Sim debug draw interface。
        root_state:
            shape=(13,)，root state。
        length:
            坐标轴长度。
        width:
            线宽。
        alpha:
            透明度。
    """
    state = root_state.detach().cpu().numpy()
    pos = state[:3]
    quat = state[3:7]
    draw_frame_np(debug_draw, pos, quat, length, width, alpha)


def draw_frame_np(
    debug_draw,
    pos: np.ndarray,
    quat_wxyz: np.ndarray,
    length: float,
    width: float,
    alpha: float,
):
    """
    绘制 numpy position + quaternion 表示的坐标系。

    Args:
        debug_draw:
            Isaac Sim debug draw interface。
        pos:
            shape=(3,)，坐标系原点。
        quat_wxyz:
            shape=(4,)，wxyz quaternion。
        length:
            坐标轴长度。
        width:
            线宽。
        alpha:
            透明度。
    """
    if debug_draw is None:
        return

    pos = np.asarray(pos, dtype=np.float64).reshape(3)
    rot_mat = quat_wxyz_to_rotmat(np.asarray(quat_wxyz, dtype=np.float64).reshape(4))

    x_end = pos + length * rot_mat[:, 0]
    y_end = pos + length * rot_mat[:, 1]
    z_end = pos + length * rot_mat[:, 2]

    debug_draw.draw_lines(
        [
            tuple(pos.tolist()),
            tuple(pos.tolist()),
            tuple(pos.tolist()),
        ],
        [
            tuple(x_end.tolist()),
            tuple(y_end.tolist()),
            tuple(z_end.tolist()),
        ],
        [
            (1.0, 0.0, 0.0, alpha),
            (0.0, 1.0, 0.0, alpha),
            (0.0, 0.25, 1.0, alpha),
        ],
        [
            float(width),
            float(width),
            float(width),
        ],
    )


def draw_smpl_mujoco_fk(
    debug_draw,
    smpl_fk_state: Dict[str, Tuple[np.ndarray, np.ndarray]],
    draw_frames: bool = True,
    draw_skeleton: bool = True,
):
    """
    在 IsaacLab viewer 里绘制 MuJoCo 计算出来的 SMPL FK。

    Args:
        debug_draw:
            Isaac Sim debug draw interface。
        smpl_fk_state:
            body_name -> (position, quaternion_wxyz)。
        draw_frames:
            是否绘制 body frame。
        draw_skeleton:
            是否绘制 skeleton segment。
    """
    if debug_draw is None:
        return

    if draw_skeleton:
        starts = []
        ends = []
        colors = []
        widths = []

        for a, b in SMPL_DRAW_SEGMENTS:
            if a not in smpl_fk_state or b not in smpl_fk_state:
                continue

            pa = smpl_fk_state[a][0]
            pb = smpl_fk_state[b][0]

            starts.append(tuple(pa.tolist()))
            ends.append(tuple(pb.tolist()))
            colors.append((0.9, 0.9, 0.9, 1.0))
            widths.append(float(args_cli.smpl_skeleton_line_width))

        if len(starts) > 0:
            debug_draw.draw_lines(starts, ends, colors, widths)

    if draw_frames:
        frame_bodies = SMPL_DRAW_BODIES if bool(args_cli.draw_smpl_all_frames) else SMPL_FRAME_BODIES_DEFAULT

        for body_name in frame_bodies:
            if body_name not in smpl_fk_state:
                continue

            pos, quat = smpl_fk_state[body_name]
            draw_frame_np(
                debug_draw=debug_draw,
                pos=pos,
                quat_wxyz=quat,
                length=float(args_cli.smpl_frame_length),
                width=float(args_cli.smpl_frame_line_width),
                alpha=1.0,
            )

# ======== newADD start======
def print_theoretical_odom_from_playback(
    frame_idx: int,
    g1_root_state: torch.Tensor,
    smpl_fk_state: Dict[str, Tuple[np.ndarray, np.ndarray]],
):
    """
    打印当前播放帧对应的“理论真实里程计”。

    详细说明：
        这里不订阅真实 ROS /Odometry。
        这个函数只根据当前 NPZ 写入 IsaacLab 的 G1 root_state 推导理论 odom。

        理论含义：
            如果真实机器人完美跟踪当前 NPZ reference，
            那么真实里程计在这一帧应当接近这些值。

        root_state 格式：
            root_state[0:3]    = root position in world
            root_state[3:7]    = root quaternion wxyz
            root_state[7:10]   = root linear velocity in world
            root_state[10:13]  = root angular velocity in world

    Args:
        frame_idx:
            当前播放帧编号。
        g1_root_state:
            当前 G1 root state，shape=(13,)。
        smpl_fk_state:
            当前 SMPL MuJoCo FK 结果。

    Returns:
        None。

    Raises:
        None。
    """
    root_state = g1_root_state.detach().cpu().numpy().astype(np.float64).reshape(-1)

    base_pos_w = root_state[0:3].copy()
    base_quat_wxyz = root_state[3:7].copy()
    base_lin_vel_w = root_state[7:10].copy()
    base_ang_vel_w = root_state[10:13].copy()

    rot_wb = quat_wxyz_to_rotmat(base_quat_wxyz)
    base_lin_vel_b = rot_wb.T @ base_lin_vel_w
    base_ang_vel_b = rot_wb.T @ base_ang_vel_w

    rpy_deg = R.from_matrix(rot_wb).as_euler("XYZ", degrees=True)
    speed_xy_w = float(np.linalg.norm(base_lin_vel_w[:2]))
    speed_xy_b = float(np.linalg.norm(base_lin_vel_b[:2]))

    print("[Theory Real Odometry From NPZ]")
    print(f"  frame_idx              = {frame_idx}")
    print("  --- nav_msgs/Odometry-like pose ---")
    print(f"  frame_id               = world / odom / camera_init")
    print(f"  child_frame_id          = body / base / root")
    print(f"  base_pos_w              = {base_pos_w}")
    print(f"  base_quat_wxyz          = {base_quat_wxyz}")
    print(f"  base_rpy_deg            = {rpy_deg}")
    print("  --- velocity from NPZ root state ---")
    print(f"  base_lin_vel_w          = {base_lin_vel_w}")
    print(f"  base_ang_vel_w          = {base_ang_vel_w}")
    print(f"  base_lin_vel_b          = {base_lin_vel_b}")
    print(f"  base_ang_vel_b          = {base_ang_vel_b}")
    print(f"  speed_xy_w              = {speed_xy_w:.6f}")
    print(f"  speed_xy_b              = {speed_xy_b:.6f}")

    if bool(args_cli.print_theory_odom_compare_smpl):
        print("  --- SMPL reference root / Pelvis for comparison ---")
        if "Pelvis" in smpl_fk_state:
            smpl_pelvis_pos_w = np.asarray(smpl_fk_state["Pelvis"][0], dtype=np.float64)
            smpl_pelvis_quat_wxyz = np.asarray(smpl_fk_state["Pelvis"][1], dtype=np.float64)
            smpl_pelvis_rpy_deg = R.from_matrix(
                quat_wxyz_to_rotmat(smpl_pelvis_quat_wxyz)
            ).as_euler("XYZ", degrees=True)

            print(f"  smpl_pelvis_pos_w       = {smpl_pelvis_pos_w}")
            print(f"  smpl_pelvis_quat_wxyz   = {smpl_pelvis_quat_wxyz}")
            print(f"  smpl_pelvis_rpy_deg     = {smpl_pelvis_rpy_deg}")
            print(f"  smpl_minus_g1_pos_w     = {smpl_pelvis_pos_w - base_pos_w}")
        else:
            print("  smpl_pelvis             = unavailable")
# =========== newADD end ========



# ============================================================
# 9. 主循环
# ============================================================

def get_motion_num_frames(motion) -> int:
    """
    获取 NPZ motion 帧数。

    Args:
        motion:
            MotionLoader 对象。

    Returns:
        int:
            motion 帧数。
    """
    total = motion.time_step_total

    if isinstance(total, torch.Tensor):
        total = int(total.reshape(-1)[0].item())
    else:
        total = int(total)

    if total <= 0:
        raise RuntimeError(f"Invalid motion frame count: {total}")

    return total


def run(sim: SimulationContext, scene: InteractiveScene):
    """
    运行播放器。

    Args:
        sim:
            IsaacLab SimulationContext。
        scene:
            IsaacLab InteractiveScene。
    """
    robot: Articulation = scene["robot"]

    env_ids = torch.arange(scene.num_envs, dtype=torch.long, device=sim.device)

    motion_file = str(Path(args_cli.motion_file).expanduser().resolve())
    smpl_csv = str(Path(args_cli.smpl_csv).expanduser().resolve())
    smpl_mjcf = str(Path(args_cli.smpl_mjcf).expanduser().resolve())

    motion = MotionLoader(motion_file, env_ids, sim.device)
    smpl_motion = load_smpl_csv(smpl_csv)
    smpl_fk = MuJoCoSmplFK(
        mjcf_path=smpl_mjcf,
        smpl_motion=smpl_motion,
        smpl_xyz_offset=args_cli.smpl_xyz_offset,
        convert_csv_rot_to_zup=bool(args_cli.convert_csv_rot_to_zup),
        convert_csv_transl_to_zup=bool(args_cli.convert_csv_transl_to_zup),
        root_rot_mode=str(args_cli.root_rot_mode),
    )

    npz_frames = get_motion_num_frames(motion)
    csv_frames = smpl_motion.num_frames

    start = max(int(args_cli.start), 0)

    if int(args_cli.end) < 0:
        end = min(npz_frames, csv_frames)
    else:
        end = min(int(args_cli.end), npz_frames, csv_frames)

    if start >= end:
        raise ValueError(f"Invalid frame range: start={start}, end={end}")

    print("=" * 80)
    print("[Playback]")
    print(f"  motion_file              = {motion_file}")
    print(f"  smpl_csv                 = {smpl_csv}")
    print(f"  smpl_mjcf                = {smpl_mjcf}")
    print(f"  npz_frames               = {npz_frames}")
    print(f"  csv_frames               = {csv_frames}")
    print(f"  range                    = [{start}, {end})")
    print(f"  mode                     = same_index only")
    print(f"  SMPL visualization        = MuJoCo FK + IsaacLab debug_draw")
    print(f"  convert_csv_rot_to_zup    = {bool(args_cli.convert_csv_rot_to_zup)}")
    print(f"  convert_csv_transl_to_zup = {bool(args_cli.convert_csv_transl_to_zup)}")
    print(f"  root_rot_mode             = {args_cli.root_rot_mode}")
    print(f"  smpl_world_alignment      = {args_cli.smpl_world_alignment}")
    print(f"  smpl_xyz_offset           = {np.asarray(args_cli.smpl_xyz_offset, dtype=np.float64)}")
    print(f"  draw_smpl_frames          = {bool(args_cli.draw_smpl_frames)}")
    print(f"  draw_smpl_skeleton        = {bool(args_cli.draw_smpl_skeleton)}")
    print("=" * 80)

    first_root = motion.body_pos_w[
        torch.tensor([start], dtype=torch.long, device=sim.device)
    ][:, args_cli.npz_root_body_idx][0].detach().cpu().numpy()

    smpl_world_rotation = np.eye(3, dtype=np.float64)
    smpl_world_translation = np.zeros(3, dtype=np.float64)
    if args_cli.smpl_world_alignment == "gvhmr_gmr_root0":
        smpl_world_rotation = R_PROJECT_ZUP_TO_GVHMR_GMR.copy()
        first_smpl_state = smpl_fk.forward(start)
        if "Pelvis" not in first_smpl_state:
            raise KeyError("SMPL FK state has no Pelvis body for root alignment")
        first_pelvis_rotated = smpl_world_rotation @ first_smpl_state["Pelvis"][0]
        smpl_world_translation[:2] = first_root[:2] - first_pelvis_rotated[:2]
        print("[SMPL world alignment]")
        print(f"  rotation_project_to_gmr =\n{smpl_world_rotation}")
        print(f"  translation_xy          = {smpl_world_translation}")
        print("  z_alignment             = ground-preserving (no pelvis-height match)")

    sim.set_camera_view(
        eye=first_root + np.array([2.0, 2.0, 1.0]),
        target=first_root,
    )

    debug_draw = None
    if _debug_draw is not None:
        debug_draw = _debug_draw.acquire_debug_draw_interface()
    else:
        print("[WARN] debug_draw is not available. SMPL skeleton/frames will not be drawn.")

    sim_dt = sim.get_physics_dt()
    cursor = float(start)

    # ======== newADD start======
    playback_loop_idx = 0
    # =========== newADD end ========

    while simulation_app.is_running():
        frame_idx = int(math.floor(cursor))
        frame_idx = int(np.clip(frame_idx, start, end - 1))

        g1_root_states = write_g1_frame(
            robot=robot,
            motion=motion,
            frame_idx=frame_idx,
            scene=scene,
        )

        smpl_fk_state = transform_smpl_fk_state(
            smpl_fk.forward(frame_idx),
            smpl_world_rotation,
            smpl_world_translation,
        )


        # ======== newADD start======
        if (
            bool(args_cli.print_theory_odom)
            and int(args_cli.print_theory_odom_every) > 0
            and playback_loop_idx % int(args_cli.print_theory_odom_every) == 0
        ):
            print_theoretical_odom_from_playback(
                frame_idx=frame_idx,
                g1_root_state=g1_root_states[0],
                smpl_fk_state=smpl_fk_state,
            )
        # =========== newADD end ========

        scene.write_data_to_sim()

        if hasattr(sim, "forward"):
            sim.forward()

        scene.update(sim_dt)

        if debug_draw is not None:
            debug_draw.clear_lines()

            draw_frame_torch(
                debug_draw=debug_draw,
                root_state=g1_root_states[0],
                length=float(args_cli.root_frame_length),
                width=float(args_cli.root_frame_line_width),
                alpha=1.0,
            )

            draw_smpl_mujoco_fk(
                debug_draw=debug_draw,
                smpl_fk_state=smpl_fk_state,
                draw_frames=bool(args_cli.draw_smpl_frames),
                draw_skeleton=bool(args_cli.draw_smpl_skeleton),
            )

        sim.render()

        cursor += float(args_cli.speed)

        # ======== newADD start======
        playback_loop_idx += 1
        # =========== newADD end ========

        if int(args_cli.max_render_loops) > 0 and playback_loop_idx >= int(args_cli.max_render_loops):
            break

        if cursor >= float(end):
            cursor = float(start)

    print(
        f"[Playback] completed {playback_loop_idx} render loop(s); "
        f"last_frame={frame_idx}",
        flush=True,
    )


# ============================================================
# 10. 程序入口
# ============================================================

def main():
    """
    程序入口。
    """
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02

    sim = SimulationContext(sim_cfg)

    scene_cfg = MinimalSceneCfg(
        num_envs=args_cli.num_envs,
        env_spacing=args_cli.env_spacing,
    )
    scene = InteractiveScene(scene_cfg)

    sim.reset()

    run(sim, scene)


if __name__ == "__main__":
    main()
    if int(args_cli.max_render_loops) > 0:
        # Isaac Sim 4.5 can hang during headless shutdown after a finite smoke test.
        # All requested frames have already rendered and stdout is flushed above.
        import os

        os._exit(0)
    simulation_app.close()
