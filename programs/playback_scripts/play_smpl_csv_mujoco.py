"""
Usage:
    cd ~/BysanRL/scripts

# 1) 单个 CSV 离线播放（推荐新写法）
cd ~/BysanRL/scripts
conda activate bydmmc_nogmr
python play_smpl_csv_mujoco.py \
    --mjcf /data/BysanRL/data_goal/xml/smpl_humanoid_zup.xml \
    --motion_path /data/BysanRL/data_goal/csv/lafan_smpl_csv/example.csv \
    --fps 50


# 2) 一个目录离线播放（会递归查找所有 .csv）
cd ~/BysanRL/scripts
conda activate bydmmc_nogmr
python play_smpl_csv_mujoco.py \
    --mjcf /data/BysanRL/data_goal/xml/smpl_humanoid_zup.xml \
    --motion_path /data/BysanRL/data_goal/csv/GVHMR/demo \
    --fps 50

# 3) 单个 CSV 实时跟随模式（文件持续增长）
cd ~/BysanRL/scripts
conda activate bydmmc_nogmr
python play_smpl_csv_mujoco.py \
    --mjcf /data/BysanRL/data_goal/xml/smpl_humanoid_zup.xml \
    --motion_path /data/BysanRL/data_goal/csv/rebocap/rebocap_realtime.csv \
    --fps 50 \
    --follow_csv True \
    --follow_start_mode latest

Controls (offline mode):
    Space : pause / resume
    1     : playback speed -0.5x
    2     : playback speed +0.5x
    5     : previous csv
    6     : next csv

Features:
- Offline mode:
  - supports one CSV file
  - supports one directory, recursively searching all .csv files
  - supports switching previous / next CSV during playback
  - supports changing playback speed during playback
- Realtime mode:
  - follows a growing CSV file
  - supports waiting for file creation
  - supports reopening automatically when the target CSV is truncated/recreated
  - only supports a single CSV file, not a directory

FPS priority:
- --force_fps > csv column `fps` > --fps

Frame range:
- --start and --end are applied to each CSV independently.
- --end = -1 means play to the last frame.

CSV columns expected:
- fps
- smpl_params_global_transl_0..2
- smpl_params_global_global_orient_0..2          (axis-angle / rotvec, radians)
- smpl_params_global_body_pose_0..68             (23 joints * 3, axis-angle / rotvec)
- smpl_params_global_betas_0..9                  (ignored for MuJoCo playback)

Notes:
- This script maps SMPL joints -> MJCF joints (x/y/z hinge stacks).
- `--z_trans True` applies an extra standing correction after SMPL->MuJoCo world conversion.
- `--csv` is kept as a backward-compatible alias of `--motion_path`.
"""

import argparse
import time
import csv
import pathlib
import os
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import mujoco
import mujoco.viewer
from scipy.spatial.transform import Rotation as R


# ======== newADD start======
def str2bool(v):
    """
    Parse common string forms into bool for argparse.

    详细说明：
        argparse 默认对 bool 类型参数处理不直观，因此这里手动支持
        true/false、1/0、yes/no 等常见写法。

    Args:
        v:
            输入的命令行参数值。

    Returns:
        bool:
            解析后的布尔值。

    Raises:
        argparse.ArgumentTypeError:
            当输入字符串不属于支持的布尔形式时抛出。
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("true", "1", "yes", "y", "t"):
        return True
    if v.lower() in ("false", "0", "no", "n", "f"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got: {v}")
# =========== newADD end ========

# ======== newADD start======
# old SMPL convention:
#   old X = left
#   old Y = up
#   old Z = forward
#
# new z-up SMPL MJCF convention:
#   new X = forward
#   new Y = left
#   new Z = up
#
# v_new = P @ v_old = [old_z, old_x, old_y]
P_OLD_SMPL_TO_NEW_ZUP = np.array(
    [
        [0.0, 0.0, 1.0],
        [1.0, 0.0, 0.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)


def rotmat_to_quat_wxyz(rot_mat: np.ndarray) -> np.ndarray:
    """
    将旋转矩阵转换为 MuJoCo freejoint 使用的 wxyz 四元数。

    详细说明：
        scipy 的 Rotation.as_quat() 返回 xyzw；
        MuJoCo freejoint 的 qpos 需要 wxyz。
        因此这里显式重排四元数顺序。

    Args:
        rot_mat:
            shape (3, 3) 的旋转矩阵。

    Returns:
        np.ndarray:
            shape (4,) 的 wxyz 四元数。
    """
    q_xyzw = R.from_matrix(np.asarray(rot_mat, dtype=np.float64).reshape(3, 3)).as_quat()
    return np.array(
        [q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]],
        dtype=np.float64,
    )


def root_rotvec_to_zup_quat_asset_basis(rotvec: np.ndarray) -> np.ndarray:
    """
    将 SMPL root global_orient 转成 z-up SMPL MJCF 的 root quaternion。

    详细说明：
        这里完全对齐 play_smpl_zup_mujoco_only.py 的处理：
            R_root_new = R_root_old @ P.T

        这个处理方式对应你前面验证过的 asset_basis root 模式。

    Args:
        rotvec:
            shape (3,) 的 SMPL root axis-angle / rotvec。

    Returns:
        np.ndarray:
            shape (4,) 的 MuJoCo wxyz quaternion。
    """
    r_old = R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()
    r_new = r_old @ P_OLD_SMPL_TO_NEW_ZUP.T
    return rotmat_to_quat_wxyz(r_new)


def body_rotvec_to_zup_xyz(rotvec: np.ndarray) -> np.ndarray:
    """
    将 SMPL body_pose rotvec 转成 z-up SMPL MJCF 的 XYZ hinge 角。

    详细说明：
        这里完全对齐 play_smpl_zup_mujoco_only.py 的处理：
            R_joint_new = P @ R_joint_old @ P.T

        然后再分解成 intrinsic XYZ 欧拉角，写入：
            joint_x, joint_y, joint_z

    Args:
        rotvec:
            shape (3,) 的 SMPL body joint rotvec。

    Returns:
        np.ndarray:
            shape (3,) 的 XYZ 欧拉角，单位为弧度。
    """
    r_old = R.from_rotvec(np.asarray(rotvec, dtype=np.float64).reshape(3)).as_matrix()
    p = P_OLD_SMPL_TO_NEW_ZUP
    r_new = p @ r_old @ p.T
    return R.from_matrix(r_new).as_euler("XYZ", degrees=False)
# =========== newADD end ========


# -----------------------------
# 1) Joint naming + mapping
# -----------------------------

def build_smpl_joint_names_23() -> List[str]:
    """
    One common SMPL(24) layout is: root + 23 body joints.
    Here body_pose is 23 joints (69 dims), root is in global_orient.

    详细说明：
        返回 SMPL 中除 root/pelvis 外的 23 个 body joints 名称列表。
        每个关节对应 body_pose 中连续的 3 个 rotvec 参数。

    Args:
        无

    Returns:
        List[str]:
            长度为 23 的关节名称列表。
    """
    return [
        "L_Hip", "R_Hip", "Spine1",
        "L_Knee", "R_Knee", "Spine2",
        "L_Ankle", "R_Ankle", "Spine3",
        "L_Foot", "R_Foot",
        "Neck", "L_Collar", "R_Collar", "Head",
        "L_Shoulder", "R_Shoulder", "L_Elbow", "R_Elbow",
        "L_Wrist", "R_Wrist", "L_Hand", "R_Hand",
    ]


def build_smpl_to_mjcf_mapping() -> Dict[str, Optional[str]]:
    """
    Map SMPL joint name -> your MJCF base joint name (without _x/_y/_z).

    详细说明：
        建立从 SMPL 关节名到 MJCF 中基础关节名的映射。
        若值为 None，则表示该 SMPL 关节在当前 MJCF 中忽略不使用。

    Args:
        无

    Returns:
        Dict[str, Optional[str]]:
            键为 SMPL 关节名，值为 MJCF 基础关节名或 None。
    """
    return {
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


# -----------------------------
# 2) MuJoCo helpers
# -----------------------------

def get_joint_qposadr(model: mujoco.MjModel, joint_name: str) -> int:
    """
    Get qpos address of a joint by name.

    详细说明：
        在 MuJoCo 中，每个 joint 对应 `data.qpos` 中某个起始地址。
        这个函数根据 joint 名字查询其 qpos 起始索引。

    Args:
        model:
            MuJoCo 模型对象。
        joint_name:
            MJCF 中 joint 的名字。

    Returns:
        int:
            该 joint 在 `data.qpos` 中的起始索引。

    Raises:
        ValueError:
            当 joint 名称不存在时抛出。
    """
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise ValueError(f"Joint not found in MJCF: {joint_name}")
    return int(model.jnt_qposadr[jid])


def set_freejoint_from_smpl(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    freejoint_name: str,
    transl_xyz: np.ndarray,
    global_orient_rotvec: np.ndarray,
) -> None:
    """
    用 z-up SMPL MJCF 逻辑写入 MuJoCo freejoint。

    详细说明：
        这里不再使用旧的 SMPL_TO_MJ，也不再使用 z_trans。
        处理方式与 play_smpl_zup_mujoco_only.py 保持一致：

        - root 平移 transl 直接写入 freejoint position；
        - root 旋转使用 asset_basis：
              R_root_new = R_root_old @ P.T

        MuJoCo freejoint qpos 格式为：
            [x, y, z, qw, qx, qy, qz]

    Args:
        model:
            MuJoCo 模型对象。
        data:
            MuJoCo 数据对象。
        freejoint_name:
            freejoint 名称，通常是 "Pelvis"。
        transl_xyz:
            shape (3,) 的 SMPL root transl。
        global_orient_rotvec:
            shape (3,) 的 SMPL root global_orient rotvec。

    Returns:
        None。
    """
    adr = get_joint_qposadr(model, freejoint_name)

    data.qpos[adr:adr + 3] = np.asarray(transl_xyz, dtype=np.float64).reshape(3)

    quat_wxyz = root_rotvec_to_zup_quat_asset_basis(global_orient_rotvec)
    data.qpos[adr + 3:adr + 7] = quat_wxyz

def set_xyz_hinges(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    base_name: str,
    angles_xyz: np.ndarray,
) -> None:
    """
    Set three hinge joints (base_name + _x/_y/_z) to angles_xyz.

    详细说明：
        将一个 3 维欧拉角向量依次写入 `base_name_x`、`base_name_y`、`base_name_z`
        这三个 MuJoCo hinge joint。

    Args:
        model:
            MuJoCo 模型对象。
        data:
            MuJoCo 数据对象。
        base_name:
            关节基础名，例如 "L_Hip"。
        angles_xyz:
            shape (3,) 的欧拉角向量。

    Returns:
        None

    Raises:
        ValueError:
            当某个对应 joint 不存在时抛出。
    """
    for ax_i, ax in enumerate(["x", "y", "z"]):
        jname = f"{base_name}_{ax}"
        adr = get_joint_qposadr(model, jname)
        data.qpos[adr] = float(angles_xyz[ax_i])


# -----------------------------
# 3) CSV reading
# -----------------------------

def infer_body_pose_dim_from_df(df: pd.DataFrame) -> int:
    """
    Infer how many body_pose columns exist in the CSV.

    详细说明：
        自动统计 CSV 中以 `smpl_params_global_body_pose_` 开头的列，
        判断 body_pose 是：
        - 63 维（21*3）
        - 69 维（23*3）

        要求这些列必须从 0 开始连续编号。

    Args:
        df:
            读入的 pandas DataFrame。

    Returns:
        int:
            body_pose 的标量维度，只能是 63 或 69。

    Raises:
        ValueError:
            当列不存在、编号不连续、或者维度不是 63/69 时抛出。
    """
    prefix = "smpl_params_global_body_pose_"
    idxs: List[int] = []
    for c in df.columns:
        if c.startswith(prefix):
            suf = c[len(prefix):]
            if suf.isdigit():
                idxs.append(int(suf))

    if len(idxs) == 0:
        raise ValueError(f"No body_pose columns found with prefix: {prefix}")

    idxs_sorted = sorted(set(idxs))
    expected = list(range(idxs_sorted[0], idxs_sorted[-1] + 1))
    if idxs_sorted != expected or idxs_sorted[0] != 0:
        raise ValueError(
            f"body_pose columns must be contiguous from 0..N-1. "
            f"Found min={idxs_sorted[0]}, max={idxs_sorted[-1]}, count={len(idxs_sorted)}"
        )

    dim = idxs_sorted[-1] + 1
    if dim not in (63, 69):
        raise ValueError(f"Unexpected body_pose dim={dim}. Only 63 or 69 are supported.")
    return dim


def infer_csv_fps(df: pd.DataFrame) -> Optional[float]:
    """
    Infer fps from CSV column `fps` if it contains a valid positive number.

    详细说明：
        该函数只负责解析 CSV 内部的 fps，不负责和命令行参数比较优先级。
        解析规则：
        - 若没有 `fps` 列，返回 None
        - 若 DataFrame 为空，返回 None
        - 若第一行值为 "none" / 空字符串 / NaN，返回 None
        - 若第一行值可转成正浮点数，返回该值
        - 其他非法情况返回 None

        这里默认整个 CSV 文件的 fps 是固定的，因此只读取第一行。

    Args:
        df:
            pandas DataFrame。

    Returns:
        Optional[float]:
            若存在合法 csv fps，则返回该值；否则返回 None。
    """
    if "fps" not in df.columns:
        return None

    if len(df) == 0:
        return None

    value = df.iloc[0]["fps"]

    if pd.isna(value):
        return None

    if isinstance(value, str):
        value_strip = value.strip().lower()
        if value_strip == "" or value_strip == "none":
            return None
        try:
            fps_from_csv = float(value_strip)
            if fps_from_csv > 0:
                return fps_from_csv
            return None
        except ValueError:
            return None

    try:
        fps_from_csv = float(value)
        if fps_from_csv > 0:
            return fps_from_csv
    except (TypeError, ValueError):
        pass

    return None


def infer_effective_fps(
    df: pd.DataFrame,
    fallback_fps: float,
    force_fps: Optional[float] = None,
) -> float:
    """
    Infer final playback fps with priority:
        force_fps > csv_fps > fallback_fps

    详细说明：
        最终播放频率优先级如下：
        1. 若命令行传入 `--force_fps` 且为正数，则无条件使用它
        2. 否则若 CSV 中 `fps` 列存在合法正数，则使用 csv fps
        3. 否则使用普通命令行 `--fps`

    Args:
        df:
            pandas DataFrame。
        fallback_fps:
            普通命令行参数 `--fps`。
        force_fps:
            高优先级强制播放频率 `--force_fps`，可为 None。

    Returns:
        float:
            实际用于播放的 fps。

    Raises:
        ValueError:
            当 force_fps 传入但不是正数时抛出。
    """
    if force_fps is not None:
        if force_fps <= 0:
            raise ValueError(f"force_fps must be positive, got {force_fps}")
        return float(force_fps)

    csv_fps = infer_csv_fps(df)
    if csv_fps is not None:
        return float(csv_fps)

    return float(fallback_fps)


def extract_smpl_frame(
    row: pd.Series,
    n_body_pose: int = 69,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract one frame's SMPL params from CSV row.

    详细说明：
        从一行 CSV 中读取：
        - transl
        - global_orient
        - body_pose

        若 body_pose 只有 63 维，则自动补 6 个 0，扩展成 69 维，
        以适配后续按 23 个关节处理的逻辑。

    Args:
        row:
            pandas 的一行数据。
        n_body_pose:
            body_pose 标量维度，支持 63 或 69。

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray]:
            - transl: shape (3,)
            - global_orient: shape (3,)
            - body_pose: shape (23, 3)

    Raises:
        ValueError:
            当 n_body_pose 不是 63 或 69 时抛出。
    """
    transl = np.array(
        [row["smpl_params_global_transl_0"], row["smpl_params_global_transl_1"], row["smpl_params_global_transl_2"]],
        dtype=np.float64,
    )
    global_orient = np.array(
        [row["smpl_params_global_global_orient_0"], row["smpl_params_global_global_orient_1"], row["smpl_params_global_global_orient_2"]],
        dtype=np.float64,
    )

    body_pose_flat = np.array([row[f"smpl_params_global_body_pose_{i}"] for i in range(n_body_pose)], dtype=np.float64)

    if n_body_pose == 63:
        body_pose_flat = np.concatenate([body_pose_flat, np.zeros((6,), dtype=np.float64)], axis=0)
    elif n_body_pose == 69:
        pass
    else:
        raise ValueError(f"n_body_pose must be 63 or 69, got {n_body_pose}")

    body_pose = body_pose_flat.reshape(-1, 3)
    return transl, global_orient, body_pose


# ======== newADD start======
def infer_body_pose_dim_from_columns(columns: List[str]) -> int:
    """
    从 CSV 表头推断 body_pose 维度。
    支持 63 或 69。
    """
    prefix = "smpl_params_global_body_pose_"
    idxs: List[int] = []

    for c in columns:
        if c.startswith(prefix):
            suf = c[len(prefix):]
            if suf.isdigit():
                idxs.append(int(suf))

    if len(idxs) == 0:
        raise ValueError(f"No body_pose columns found with prefix: {prefix}")

    idxs_sorted = sorted(set(idxs))
    expected = list(range(idxs_sorted[0], idxs_sorted[-1] + 1))
    if idxs_sorted != expected or idxs_sorted[0] != 0:
        raise ValueError(
            f"body_pose columns must be contiguous from 0..N-1. "
            f"Found min={idxs_sorted[0]}, max={idxs_sorted[-1]}, count={len(idxs_sorted)}"
        )

    dim = idxs_sorted[-1] + 1
    if dim not in (63, 69):
        raise ValueError(f"Unexpected body_pose dim={dim}. Only 63 or 69 are supported.")

    return dim


def infer_csv_fps_from_row_dict(row_dict: Dict[str, str]) -> Optional[float]:
    """
    从单行字典中解析 fps。
    若没有 fps 列，或者值非法，则返回 None。
    """
    if "fps" not in row_dict:
        return None

    value = row_dict["fps"]
    if value is None:
        return None

    if isinstance(value, str):
        value_strip = value.strip().lower()
        if value_strip == "" or value_strip == "none":
            return None
        try:
            fps_from_csv = float(value_strip)
            return fps_from_csv if fps_from_csv > 0 else None
        except ValueError:
            return None

    try:
        fps_from_csv = float(value)
        return fps_from_csv if fps_from_csv > 0 else None
    except (TypeError, ValueError):
        return None


def infer_effective_fps_from_optional_csv_value(
    csv_fps: Optional[float],
    fallback_fps: float,
    force_fps: Optional[float] = None,
) -> float:
    """
    实时模式下的 fps 选择逻辑：
        force_fps > csv_fps > fallback_fps
    """
    if force_fps is not None:
        if force_fps <= 0:
            raise ValueError(f"force_fps must be positive, got {force_fps}")
        return float(force_fps)

    if csv_fps is not None:
        return float(csv_fps)

    return float(fallback_fps)


def extract_smpl_frame_from_row_dict(
    row_dict: Dict[str, str],
    n_body_pose: int = 69,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    从一行 CSV 字典中提取：
        transl (3,)
        global_orient (3,)
        body_pose (23, 3)
    """
    transl = np.array(
        [
            float(row_dict["smpl_params_global_transl_0"]),
            float(row_dict["smpl_params_global_transl_1"]),
            float(row_dict["smpl_params_global_transl_2"]),
        ],
        dtype=np.float64,
    )

    global_orient = np.array(
        [
            float(row_dict["smpl_params_global_global_orient_0"]),
            float(row_dict["smpl_params_global_global_orient_1"]),
            float(row_dict["smpl_params_global_global_orient_2"]),
        ],
        dtype=np.float64,
    )

    body_pose_flat = np.array(
        [float(row_dict[f"smpl_params_global_body_pose_{i}"]) for i in range(n_body_pose)],
        dtype=np.float64,
    )

    if n_body_pose == 63:
        body_pose_flat = np.concatenate(
            [body_pose_flat, np.zeros((6,), dtype=np.float64)],
            axis=0,
        )
    elif n_body_pose != 69:
        raise ValueError(f"n_body_pose must be 63 or 69, got {n_body_pose}")

    body_pose = body_pose_flat.reshape(-1, 3)
    return transl, global_orient, body_pose


def apply_smpl_frame_to_mujoco(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    smpl_joint_names: List[str],
    mapping: Dict[str, Optional[str]],
    transl: np.ndarray,
    global_orient: np.ndarray,
    body_pose: np.ndarray,
) -> None:
    """
    把一帧 SMPL 参数写入 z-up SMPL MuJoCo MJCF。

    详细说明：
        该函数对齐 play_smpl_zup_mujoco_only.py 的 write_frame 逻辑：

        - qpos/qvel 清零，避免上一帧残留；
        - root 使用 asset_basis：
              R_root_new = R_root_old @ P.T
        - body joint 使用：
              R_joint_new = P @ R_joint_old @ P.T
        - body joint 最后分解成 intrinsic XYZ，写入 base_x/base_y/base_z。

    Args:
        model:
            MuJoCo 模型对象。
        data:
            MuJoCo 数据对象。
        smpl_joint_names:
            SMPL 23 个 body joints 名称。
        mapping:
            SMPL joint name 到 MJCF base joint name 的映射。
        transl:
            shape (3,) 的 root 平移。
        global_orient:
            shape (3,) 的 root global_orient。
        body_pose:
            shape (23, 3) 的 body_pose。

    Returns:
        None。
    """
    data.qpos[:] = 0.0
    data.qvel[:] = 0.0

    set_freejoint_from_smpl(
        model=model,
        data=data,
        freejoint_name="Pelvis",
        transl_xyz=transl,
        global_orient_rotvec=global_orient,
    )

    for j, smpl_name in enumerate(smpl_joint_names):
        mjcf_base = mapping.get(smpl_name, None)
        if mjcf_base is None:
            continue

        rotvec = body_pose[j]
        angles_xyz = body_rotvec_to_zup_xyz(rotvec)
        set_xyz_hinges(model, data, mjcf_base, angles_xyz)

    mujoco.mj_forward(model, data)

def open_follow_csv_and_prepare(
    csv_path: str,
    follow_start_mode: str,
    start_frame: int,
    follow_poll_dt: float,
    follow_wait_timeout: float,
) -> Tuple[object, List[str], int, Optional[int]]:
    """
    打开 follow 模式的 CSV，并完成：
    1) 等待文件出现
    2) 等待 header 可读
    3) 解析 body_pose 维度
    4) 根据 begin/latest 决定初始读位置

    Returns:
        fp, header, body_pose_dim, file_inode
    """
    wait_start_time = time.time()

    while True:
        try:
            fp = open(csv_path, "r", newline="")
            header_line = fp.readline()

            if header_line.strip() != "":
                break

            fp.close()
        except FileNotFoundError:
            pass

        if follow_wait_timeout > 0 and (time.time() - wait_start_time) > follow_wait_timeout:
            raise TimeoutError(
                f"Waited {follow_wait_timeout} seconds, but CSV is still not ready: {csv_path}"
            )

        time.sleep(follow_poll_dt)

    header = next(csv.reader([header_line]))
    body_pose_dim = infer_body_pose_dim_from_columns(header)

    if follow_start_mode == "latest":
        fp.seek(0, os.SEEK_END)
    else:
        skipped = 0
        while skipped < start_frame:
            pos = fp.tell()
            line = fp.readline()
            if not line:
                fp.seek(pos)
                break
            if not line.endswith("\n") and not line.endswith("\r\n"):
                fp.seek(pos)
                break
            skipped += 1

    try:
        st = os.stat(csv_path)
        file_inode = getattr(st, "st_ino", None)
    except FileNotFoundError:
        file_inode = None

    return fp, header, body_pose_dim, file_inode


def maybe_reopen_follow_csv_if_rotated_or_truncated(
    csv_path: str,
    fp,
    old_inode: Optional[int],
    args,
) -> Tuple[object, Optional[List[str]], Optional[int], Optional[int], bool]:
    """
    检查 follow 的目标文件是否被：
    1) 截断（size < 当前读取位置）
    2) 替换/重建（inode 变化）

    若发生，则自动重开文件。

    Returns:
        fp, header_or_none, body_pose_dim_or_none, inode, reopened
    """
    try:
        st = os.stat(csv_path)
    except FileNotFoundError:
        return fp, None, None, old_inode, False

    current_pos = fp.tell()
    current_inode = getattr(st, "st_ino", None)
    current_size = st.st_size

    need_reopen = False

    if current_size < current_pos:
        need_reopen = True

    if old_inode is not None and current_inode is not None and current_inode != old_inode:
        need_reopen = True

    if not need_reopen:
        return fp, None, None, old_inode, False

    print("[Info] follow target file was truncated/recreated, reopen it.")

    try:
        fp.close()
    except Exception:
        pass

    new_fp, new_header, new_body_pose_dim, new_inode = open_follow_csv_and_prepare(
        csv_path=csv_path,
        follow_start_mode=args.follow_start_mode,
        start_frame=args.start,
        follow_poll_dt=args.follow_poll_dt,
        follow_wait_timeout=args.follow_wait_timeout,
    )

    return new_fp, new_header, new_body_pose_dim, new_inode, True


def run_follow_csv_playback(args) -> None:
    """
    实时跟随一个持续增长的 CSV 文件。
    适合一个进程录制 rebocap CSV，另一个进程实时播放。

    额外增强：
    - 支持启动时文件还不存在
    - 支持录制端后续用 "w" 截断/重建文件后自动重开
    """
    fp, header, body_pose_dim, file_inode = open_follow_csv_and_prepare(
        csv_path=args.csv,
        follow_start_mode=args.follow_start_mode,
        start_frame=args.start,
        follow_poll_dt=args.follow_poll_dt,
        follow_wait_timeout=args.follow_wait_timeout,
    )

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    smpl_joint_names = build_smpl_joint_names_23()
    mapping = build_smpl_to_mjcf_mapping()

    effective_fps = infer_effective_fps_from_optional_csv_value(
        csv_fps=None,
        fallback_fps=args.fps,
        force_fps=args.force_fps,
    )
    dt = (1.0 / effective_fps) / max(args.speed, 1e-6)

    print(f"[Info] follow_csv enabled.")
    print(f"[Info] root mode         = asset_basis")
    print(f"[Info] body mode         = P @ R_old @ P.T + XYZ")
    print(f"[Info] follow_start_mode = {args.follow_start_mode}")
    print(f"[Info] follow_poll_dt    = {args.follow_poll_dt}")
    print(f"[Info] initial_fps       = {effective_fps:.6f}")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        mujoco.mj_resetData(model, data)
        next_frame_wall_time = time.perf_counter()

        while viewer.is_running():
            # ======== newADD start======
            fp, maybe_header, maybe_body_pose_dim, maybe_inode, reopened = (
                maybe_reopen_follow_csv_if_rotated_or_truncated(
                    csv_path=args.csv,
                    fp=fp,
                    old_inode=file_inode,
                    args=args,
                )
            )
            if reopened:
                header = maybe_header
                body_pose_dim = maybe_body_pose_dim
                file_inode = maybe_inode
                next_frame_wall_time = time.perf_counter()
                viewer.sync()
                time.sleep(args.follow_poll_dt)
                continue
            # =========== newADD end ========

            now = time.perf_counter()

            if now < next_frame_wall_time:
                viewer.sync()
                time.sleep(min(args.follow_poll_dt, max(next_frame_wall_time - now, 0.001)))
                continue

            pos = fp.tell()
            line = fp.readline()

            if not line:
                viewer.sync()
                time.sleep(args.follow_poll_dt)
                continue

            if not line.endswith("\n") and not line.endswith("\r\n"):
                fp.seek(pos)
                viewer.sync()
                time.sleep(args.follow_poll_dt)
                continue

            row_dict = next(csv.DictReader([line], fieldnames=header))

            csv_fps = infer_csv_fps_from_row_dict(row_dict)
            effective_fps = infer_effective_fps_from_optional_csv_value(
                csv_fps=csv_fps,
                fallback_fps=args.fps,
                force_fps=args.force_fps,
            )
            dt = (1.0 / effective_fps) / max(args.speed, 1e-6)

            transl, global_orient, body_pose = extract_smpl_frame_from_row_dict(
                row_dict,
                n_body_pose=body_pose_dim,
            )

            apply_smpl_frame_to_mujoco(
                model=model,
                data=data,
                smpl_joint_names=smpl_joint_names,
                mapping=mapping,
                transl=transl,
                global_orient=global_orient,
                body_pose=body_pose,
            )

            viewer.sync()

            next_frame_wall_time = max(
                next_frame_wall_time + dt,
                time.perf_counter(),
            )

    try:
        fp.close()
    except Exception:
        pass
# =========== newADD end ========


# ======== newADD start======
def resolve_input_motion_path(args) -> str:
    """
    Resolve input path from `--motion_path` or legacy `--csv`.

    详细说明：
        为了兼容旧版本，这里允许用户使用：
        - --motion_path
        - --csv

        但两者不能同时传。最终返回统一的输入路径字符串。

    Args:
        args:
            argparse 解析后的参数对象。

    Returns:
        str:
            统一后的输入路径。

    Raises:
        ValueError:
            当两个参数同时给出，或者两个都没给出时抛出。
    """
    if args.motion_path is not None and args.csv is not None:
        raise ValueError("Please use only one of --motion_path and --csv.")

    motion_path = args.motion_path if args.motion_path is not None else args.csv
    if motion_path is None:
        raise ValueError("One of --motion_path or --csv must be provided.")

    return str(pathlib.Path(motion_path).expanduser())


def collect_csv_files(motion_path: str) -> List[str]:
    """
    Collect offline CSV files from a file path or a directory.

    详细说明：
        - 如果 `motion_path` 是单个 `.csv` 文件，则返回长度为 1 的列表
        - 如果 `motion_path` 是目录，则递归查找其下所有 `.csv`
        - 返回结果按字典序排序，保证播放顺序稳定

    Args:
        motion_path:
            单个 CSV 文件路径，或目录路径。

    Returns:
        List[str]:
            所有待播放 CSV 的绝对路径列表。

    Raises:
        FileNotFoundError:
            输入路径不存在，或目录中没有任何 `.csv` 文件时抛出。
        ValueError:
            当输入是文件但后缀不是 `.csv` 时抛出。
    """
    path_obj = pathlib.Path(motion_path).expanduser()

    if not path_obj.exists():
        raise FileNotFoundError(f"Input path does not exist: {path_obj}")

    if path_obj.is_file():
        if path_obj.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file, got: {path_obj}")
        return [str(path_obj.resolve())]

    csv_files = sorted(
        str(p.resolve())
        for p in path_obj.rglob("*.csv")
        if p.is_file()
    )

    if len(csv_files) == 0:
        raise FileNotFoundError(f"No .csv files found recursively under: {path_obj}")

    return csv_files


def resolve_single_csv_for_follow(args) -> str:
    """
    Resolve a single CSV path for realtime follow mode.

    详细说明：
        `follow_csv=True` 时，不允许目录模式，
        但允许目标 CSV 在程序启动时还不存在，
        因为它可能稍后由另一个录制进程创建并持续追加内容。

    Args:
        args:
            argparse 参数对象。

    Returns:
        str:
            单个 CSV 文件的规范化路径字符串。

    Raises:
        ValueError:
            当传入目录，或者路径后缀不是 `.csv` 时抛出。
    """
    motion_path = resolve_input_motion_path(args)
    path_obj = pathlib.Path(motion_path).expanduser()

    if path_obj.exists():
        if not path_obj.is_file():
            raise ValueError("--follow_csv=True only supports a single CSV file, not a directory.")
        if path_obj.suffix.lower() != ".csv":
            raise ValueError(f"Expected a .csv file for follow mode, got: {path_obj}")
        return str(path_obj.resolve())

    if path_obj.suffix.lower() != ".csv":
        raise ValueError(
            "--follow_csv=True expects a target .csv file path. "
            f"Got a non-existing path without .csv suffix: {path_obj}"
        )

    return str(path_obj)


def describe_effective_fps_source(
    csv_fps: Optional[float],
    effective_fps: float,
    args,
) -> str:
    """
    Describe where the final playback FPS comes from.

    详细说明：
        只是为了打印信息更清楚，让用户知道当前播放频率
        到底来自：
        - force_fps
        - CSV 内部 fps
        - 普通 --fps

    Args:
        csv_fps:
            CSV 中解析出的 fps，可为 None。
        effective_fps:
            最终实际使用的播放 fps。
        args:
            argparse 参数对象。

    Returns:
        str:
            可直接打印的描述字符串。
    """
    if args.force_fps is not None:
        return (
            f"force_fps is set: {args.force_fps:.6f}, "
            f"override csv_fps={csv_fps} and --fps={args.fps}"
        )

    if csv_fps is not None:
        return f"CSV fps detected: {csv_fps:.6f}, override --fps {args.fps}"

    return f"Use playback fps from --fps: {effective_fps:.6f}"


def load_csv_motion_bundle(csv_path: str, args) -> Dict[str, object]:
    """
    Load one offline CSV and package all metadata needed for playback.

    详细说明：
        每次切换到一个新 CSV 时，统一在这里完成：
        - 读取 DataFrame
        - 推断 body_pose 维度
        - 推断最终 fps
        - 计算 start / end 帧范围

    Args:
        csv_path:
            当前 CSV 文件路径。
        args:
            argparse 参数对象。

    Returns:
        Dict[str, object]:
            一个包含播放所需所有信息的字典。

    Raises:
        ValueError:
            CSV 为空，或者 start/end 范围非法时抛出。
    """
    df = pd.read_csv(csv_path)

    if len(df) == 0:
        raise ValueError(f"CSV has no frame rows: {csv_path}")

    body_pose_dim = infer_body_pose_dim_from_df(df)
    csv_fps = infer_csv_fps(df)
    effective_fps = infer_effective_fps(df, args.fps, args.force_fps)

    start = max(int(args.start), 0)
    end = len(df) if int(args.end) < 0 else min(int(args.end), len(df))

    if start >= end:
        raise ValueError(
            f"Invalid frame range for {csv_path}: "
            f"start={start}, end={end}, total_frames={len(df)}"
        )

    return {
        "csv_path": csv_path,
        "df": df,
        "body_pose_dim": int(body_pose_dim),
        "csv_fps": csv_fps,
        "effective_fps": float(effective_fps),
        "start": start,
        "end": end,
        "num_frames": end - start,
    }


def safe_keycode_to_char(keycode: int) -> str:
    """
    Convert MuJoCo keycode to a Python character safely.

    详细说明：
        某些特殊键可能无法直接安全转换，这里做一个兜底。
        对于本脚本关心的按键：
        - Space
        - 1 / 2
        - 5 / 6

        这个函数已经够用。

    Args:
        keycode:
            MuJoCo 回调传入的整数键值。

    Returns:
        str:
            转换后的单字符；失败则返回空字符串。
    """
    try:
        return chr(int(keycode))
    except (TypeError, ValueError, OverflowError):
        return ""


def run_offline_csv_playlist_playback(args) -> None:
    """
    Offline playback for one CSV or a recursively collected CSV playlist.

    详细说明：
        这是这次新增的核心离线播放入口。支持：
        - 单个 CSV 播放
        - 目录递归播放
        - 播放中按 1/2 动态调速
        - 播放中按 5/6 切换上一条/下一条 CSV
        - Space 暂停/继续

    Args:
        args:
            argparse 参数对象。

    Returns:
        None
    """
    motion_path = resolve_input_motion_path(args)
    csv_files = collect_csv_files(motion_path)

    model = mujoco.MjModel.from_xml_path(args.mjcf)
    data = mujoco.MjData(model)

    smpl_joint_names = build_smpl_joint_names_23()
    mapping = build_smpl_to_mjcf_mapping()

    state = {
        "motion_idx": 0,
        "frame_idx": 0,
        "speed": float(args.speed),
        "paused": False,
        "pending_switch": 0,
        "reload_motion": True,
    }

    current_motion: Dict[str, object] = {}

    def key_callback(keycode):
        key = safe_keycode_to_char(keycode)

        if key == " ":
            state["paused"] = not bool(state["paused"])
            print(f"[Key] paused = {state['paused']}")
            return

        if key == "1":
            state["speed"] = max(0.5, float(state["speed"]) - 0.5)
            print(f"[Key] speed = {state['speed']:.2f}x")
            return

        if key == "2":
            state["speed"] = float(state["speed"]) + 0.5
            print(f"[Key] speed = {state['speed']:.2f}x")
            return

        if key == "5":
            if len(csv_files) > 1:
                state["pending_switch"] = -1
                state["reload_motion"] = True
                print("[Key] switch to previous csv")
            else:
                print("[Key] only one csv, ignore previous")
            return

        if key == "6":
            if len(csv_files) > 1:
                state["pending_switch"] = 1
                state["reload_motion"] = True
                print("[Key] switch to next csv")
            else:
                print("[Key] only one csv, ignore next")
            return

    print("=" * 80)
    print(f"[Info] offline csv playback")
    print(f"[Info] root mode      = asset_basis")
    print(f"[Info] body mode      = P @ R_old @ P.T + XYZ")
    print(f"[Info] motion_path     = {motion_path}")
    print(f"[Info] csv_count       = {len(csv_files)}")
    print(f"[Info] initial_speed   = {float(args.speed):.2f}x")
    print("[Info] controls:")
    print("       Space : pause / resume")
    print("       1     : playback speed -0.5x")
    print("       2     : playback speed +0.5x")
    print("       5     : previous csv")
    print("       6     : next csv")
    print("=" * 80)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        next_frame_wall_time = time.perf_counter()

        while viewer.is_running():
            if int(state["pending_switch"]) != 0:
                delta = int(state["pending_switch"])
                state["motion_idx"] = (int(state["motion_idx"]) + delta) % len(csv_files)
                state["pending_switch"] = 0

            if bool(state["reload_motion"]):
                load_ok = False
                tried = 0

                while tried < len(csv_files):
                    candidate = csv_files[int(state["motion_idx"])]
                    try:
                        current_motion = load_csv_motion_bundle(candidate, args)
                        load_ok = True
                        break
                    except Exception as e:
                        print(f"[Warn] failed to load csv: {candidate}")
                        print(f"       reason: {e}")
                        tried += 1
                        if len(csv_files) == 1:
                            raise
                        state["motion_idx"] = (int(state["motion_idx"]) + 1) % len(csv_files)

                if not load_ok:
                    raise RuntimeError("No valid CSV could be loaded from motion_path.")

                mujoco.mj_resetData(model, data)
                state["reload_motion"] = False

                start_frame = int(current_motion["start"])
                first_row = current_motion["df"].iloc[start_frame]
                transl, global_orient, body_pose = extract_smpl_frame(
                    first_row,
                    n_body_pose=int(current_motion["body_pose_dim"]),
                )

                apply_smpl_frame_to_mujoco(
                    model=model,
                    data=data,
                    smpl_joint_names=smpl_joint_names,
                    mapping=mapping,
                    transl=transl,
                    global_orient=global_orient,
                    body_pose=body_pose,
                )

                state["frame_idx"] = start_frame + 1
                next_frame_wall_time = time.perf_counter()
                viewer.sync()

                print("-" * 80)
                print(f"[Info] motion {int(state['motion_idx']) + 1}/{len(csv_files)}")
                print(f"[Info] csv_path      = {current_motion['csv_path']}")
                print(f"[Info] frame_range   = [{current_motion['start']}, {current_motion['end']})")
                print(
                    f"[Info] "
                    f"{describe_effective_fps_source(current_motion['csv_fps'], current_motion['effective_fps'], args)}"
                )
                print(f"[Info] speed        = {float(state['speed']):.2f}x")
                print("-" * 80)
                continue

            if bool(state["paused"]):
                viewer.sync()
                time.sleep(0.01)
                continue

            effective_fps = float(current_motion["effective_fps"])
            dt = (1.0 / effective_fps) / max(float(state["speed"]), 1e-6)

            now = time.perf_counter()
            if now < next_frame_wall_time:
                viewer.sync()
                time.sleep(min(0.005, max(next_frame_wall_time - now, 0.001)))
                continue

            frame_idx = int(state["frame_idx"])
            if frame_idx >= int(current_motion["end"]):
                state["paused"] = True
                state["frame_idx"] = int(current_motion["start"])
                print("[Info] reached end of current csv. Paused at the end; no auto switch.")
                print("[Info] Press Space to replay current csv from the start, or press 5/6 to switch manually.")
                viewer.sync()
                time.sleep(0.01)
                continue

            row = current_motion["df"].iloc[frame_idx]
            transl, global_orient, body_pose = extract_smpl_frame(
                row,
                n_body_pose=int(current_motion["body_pose_dim"]),
            )

            apply_smpl_frame_to_mujoco(
                model=model,
                data=data,
                smpl_joint_names=smpl_joint_names,
                mapping=mapping,
                transl=transl,
                global_orient=global_orient,
                body_pose=body_pose,
            )

            viewer.sync()

            state["frame_idx"] = frame_idx + 1
            next_frame_wall_time = max(
                next_frame_wall_time + dt,
                time.perf_counter(),
            )
# =========== newADD end ========


# -----------------------------
# 4) Main playback
# -----------------------------

def main():
    """
    Main entry of the MuJoCo playback script.

    详细说明：
        该函数完成以下流程：
        1. 解析命令行参数
        2. 对离线模式：
           - 支持单个 CSV
           - 支持目录递归查找所有 CSV
           - 支持播放中用按键切换上一条/下一条 CSV
           - 支持播放中动态调速
        3. 对实时模式：
           - 跟随一个持续增长的 CSV 文件

    Args:
        无

    Returns:
        None
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--mjcf", type=str, required=True, help="Path to humanoid MJCF xml")

    parser.add_argument(
        "--motion_path",
        type=str,
        default=None,
        help="Offline input path: either a single .csv file or a directory to recursively search all .csv files.",
    )
    parser.add_argument(
        "--csv",
        type=str,
        default=None,
        help="Legacy alias of --motion_path. Still supported for backward compatibility.",
    )

    parser.add_argument("--fps", type=float, default=30.0, help="Fallback playback FPS")
    parser.add_argument(
        "--force_fps",
        type=float,
        default=None,
        help="Highest-priority playback FPS. If set, override both CSV fps and --fps",
    )
    parser.add_argument("--speed", type=float, default=1.0, help="Initial speed multiplier (1.0 = realtime)")
    parser.add_argument("--start", type=int, default=0, help="Start frame index for each CSV")
    parser.add_argument("--end", type=int, default=-1, help="End frame index (exclusive), -1 means to end")

    parser.add_argument(
        "--follow_csv",
        type=str2bool,
        default=False,
        help="Whether to follow a growing CSV file in realtime. This mode only supports a single CSV file.",
    )
    parser.add_argument(
        "--follow_poll_dt",
        type=float,
        default=0.002,
        help="Polling interval in seconds when waiting for new CSV rows.",
    )
    parser.add_argument(
        "--follow_start_mode",
        type=str,
        choices=["begin", "latest"],
        default="begin",
        help="begin: play from file start; latest: jump to current EOF and only play newly appended rows.",
    )
    parser.add_argument(
        "--follow_wait_timeout",
        type=float,
        default=-1.0,
        help="Seconds to wait for CSV file/header before raising error. Negative means wait forever.",
    )

    args = parser.parse_args()

    if args.speed <= 0:
        raise ValueError(f"--speed must be positive, got {args.speed}")

    if args.follow_csv:
        args.csv = resolve_single_csv_for_follow(args)
        run_follow_csv_playback(args)
        return

    run_offline_csv_playlist_playback(args)


if __name__ == "__main__":
    main()