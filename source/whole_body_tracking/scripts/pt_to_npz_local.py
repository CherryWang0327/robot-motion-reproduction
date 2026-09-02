"""Usage
conda activate bydmmc
cd /home/pcbysan/BysanRL/BeyondMimic/whole_body_tracking/scripts/
python pt_to_npz_local.py \
    --input_path /data/BysanRL/data_goal/takiguchi_accident_11_fall_down_takiguchi_stageii_pyroki.pt \
    --output_path /data/BysanRL/data_goal/npz/rebocap_proto \
    --output_fps 50 \
    --device cpu \
    --headless \
    --skip_existing

    --input_fps 50 \
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import numpy as np

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay motion from csv file and output to npz file.")
parser.add_argument(
    "--input_path",
    type=str,
    required=True,
    help="输入 pt 文件路径，或包含多个 pt 的目录路径。",
)

parser.add_argument(
    "--input_fps",
    type=float,
    default=None,
    help="输入 fps。若提供，则优先使用该值；若不提供，则读取 pt 文件中的 fps。",
)

parser.add_argument(
    "--output_path",
    type=str,
    required=True,
    help=(
        "输出路径。"
        "若 input_path 是单文件，则这里可以是输出目录或具体 .npz 文件；"
        "若 input_path 是目录，则这里应为输出根目录。"
    ),
)

parser.add_argument(
    "--output_fps",
    type=int,
    default=60,
    help="输出 fps。若不提供，则默认使用 pkl 内部的 fps。",
)

# parser.add_argument(
#     "--frame_range",
#     nargs=2,
#     type=int,
#     metavar=("START", "END"),
#     help=(
#         "frame range: START END (both inclusive). The frame index starts from 1. If not provided, all frames will be"
#         " loaded."
#     ),
# )
parser.add_argument(
    "--pattern",
    type=str,
    default="*.pt",
    help="当输入是目录时，用于递归匹配文件的 glob 模式，默认 *.pt。",
)

parser.add_argument(
    "--suffix",
    type=str,
    default="_proto",
    help="输出文件名附加后缀，默认 _proto。",
)

parser.add_argument(
    "--skip_existing",
    action="store_true",
    default=False,
    help="若指定，则当输出文件已存在时跳过转换。",
)

parser.add_argument(
    "--input_root_rot_order",
    type=str,
    choices=["xyzw", "wxyz"],
    default="xyzw",
    help="输入 pt 中 root_rot 的四元数顺序，默认 xyzw。",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()


# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.math import axis_angle_from_quat, quat_conjugate, quat_mul, quat_slerp

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    # ground plane
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    # lights
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


# ======== newADD start======
def reorder_quaternion(quat: torch.Tensor, src_order: str, dst_order: str) -> torch.Tensor:
    """
    重排四元数分量顺序。

    该函数只做分量顺序重排，不做归一化、不做坐标系变换。
    例如：
    - xyzw -> wxyz: [x, y, z, w] -> [w, x, y, z]
    - wxyz -> xyzw: [w, x, y, z] -> [x, y, z, w]

    Args:
        quat:
            形状为 (..., 4) 的四元数张量。
        src_order:
            输入顺序，xyzw 或 wxyz。
        dst_order:
            输出顺序，xyzw 或 wxyz。

    Returns:
        torch.Tensor:
            重排后的四元数张量。
    """
    if quat.shape[-1] != 4:
        raise ValueError(f"Quaternion tensor must have last dimension 4, but got shape {tuple(quat.shape)}.")

    if src_order == dst_order:
        return quat.clone()

    if src_order == "xyzw" and dst_order == "wxyz":
        return quat[..., [3, 0, 1, 2]]

    if src_order == "wxyz" and dst_order == "xyzw":
        return quat[..., [1, 2, 3, 0]]

    raise ValueError(f"Unsupported quaternion order conversion: {src_order} -> {dst_order}")
# =========== newADD end ========



# ======== newADD start======
from pathlib import Path


def collect_input_files(input_path: Path, pattern: str) -> list[Path]:
    """
    收集待处理的 pt 文件。

    规则：
    - 若 input_path 是单个文件，则返回 [input_path]
    - 若 input_path 是目录，则递归匹配该目录下所有 pattern 对应文件

    Args:
        input_path:
            输入路径，可以是文件或目录。
        pattern:
            匹配模式，例如 *.pt

    Returns:
        list[Path]:
            待处理文件列表。
    """
    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        if input_path.suffix.lower() != ".pt":
            raise ValueError(f"Input file must be a .pt file, but got: {input_path}")
        return [input_path]

    files = sorted([p for p in input_path.rglob(pattern) if p.is_file()])
    if len(files) == 0:
        raise ValueError(f"No files matched pattern '{pattern}' under directory: {input_path}")

    return files
# =========== newADD end ========

# ======== newADD start======
def build_output_file_path(
    src_file: Path,
    input_root: Path,
    output_path: Path,
    suffix: str,
    input_is_file: bool,
) -> Path:
    """
    根据输入文件路径构造输出 npz 路径。

    规则：
    1. 若输入是单文件：
       - 若 output_path 以 .npz 结尾，则直接使用它
       - 否则视为输出目录，输出为 output_path / (src_stem + suffix + ".npz")
    2. 若输入是目录：
       - 保持相对目录树结构
       - 输出为 output_path / 相对父目录 / (src_stem + suffix + ".npz")

    Args:
        src_file:
            当前输入 pt 文件。
        input_root:
            输入根目录。若是目录批处理，就用它计算相对路径。
        output_path:
            用户传入的输出路径。
        suffix:
            文件名附加后缀。
        input_is_file:
            是否为单文件模式。

    Returns:
        Path:
            输出 npz 路径。
    """
    dst_name = f"{src_file.stem}{suffix}.npz"

    if input_is_file:
        if output_path.suffix.lower() == ".npz":
            return output_path
        return output_path / dst_name

    rel_parent = src_file.relative_to(input_root).parent
    return output_path / rel_parent / dst_name
# =========== newADD end ========


class MotionLoader:
    def __init__(
        self,
        motion_file,
        input_fps,
        output_fps,
        device,
        input_root_rot_order="xyzw",
    ):
        self.motion_file = motion_file
        self.requested_input_fps  = input_fps
        self.requested_output_fps = output_fps
        self.current_idx = 0
        self.device = device
        self.input_root_rot_order = input_root_rot_order
        self._load_motion()
        self._interpolate_motion()
        self._compute_velocities()

    # ======== newADD start======
    def _load_motion(self):
        """
        从 pt 文件中加载动作数据。

        pt 需要至少包含以下键：
        - root_pos: shape = (T, 3)
        - root_rot: shape = (T, 4)
        - dof_pos:  shape = (T, D)

        其中：
        - 输入 root_rot 的顺序由 self.input_root_rot_order 指定
        - 读入后统一转换成内部使用的 wxyz
        """
        motion_data = torch.load(self.motion_file, map_location=self.device, weights_only=False)

        if not isinstance(motion_data, dict):
            raise TypeError(f"Expected top-level object in {self.motion_file} to be dict, but got {type(motion_data)}.")

        required_keys = ["gts", "grs", "dps"]
        for key in required_keys:
            if key not in motion_data:
                raise KeyError(f"{self.motion_file} is missing required key: {key}")

        gts = motion_data["gts"]
        grs = motion_data["grs"]
        dps = motion_data["dps"]
        motion_dt = motion_data["motion_dt"]

        if isinstance(gts, torch.Tensor):
            gts = gts.detach().cpu().numpy()
        if isinstance(grs, torch.Tensor):
            grs = grs.detach().cpu().numpy()
        if isinstance(dps, torch.Tensor):
            dps = dps.detach().cpu().numpy()
        if isinstance(motion_dt, torch.Tensor):
            motion_dt = float(motion_dt.reshape(-1)[0].item())
        else:
            motion_dt = float(np.asarray(motion_dt).reshape(-1)[0])

        root_pos = np.asarray(gts[:, 0, :], dtype=np.float32)
        root_rot = np.asarray(grs[:, 0, :], dtype=np.float32)
        dof_pos = np.asarray(dps, dtype=np.float32)
        pt_fps = int(1.0 / motion_dt)

        if self.requested_input_fps is not None:
            self.input_fps = float(self.requested_input_fps)
        else:
            self.input_fps = pt_fps

        if self.input_fps <= 0:
            raise ValueError(f"input_fps must be positive, but got {self.input_fps}")

        if self.requested_output_fps is not None:
            self.output_fps = int(self.requested_output_fps)
        else:
            self.output_fps = int(round(self.input_fps))

        if self.output_fps <= 0:
            raise ValueError(f"output_fps must be positive, but got {self.output_fps}")

        self.input_dt = 1.0 / float(self.input_fps)
        self.output_dt = 1.0 / float(self.output_fps)

        if root_pos.ndim != 2 or root_pos.shape[1] != 3:
            raise ValueError(f"{self.motion_file}: root_pos must have shape (T, 3), but got {root_pos.shape}.")
        if root_rot.ndim != 2 or root_rot.shape[1] != 4:
            raise ValueError(f"{self.motion_file}: root_rot must have shape (T, 4), but got {root_rot.shape}.")
        if dof_pos.ndim != 2:
            raise ValueError(f"{self.motion_file}: dof_pos must have shape (T, D), but got {dof_pos.shape}.")
        if not (root_pos.shape[0] == root_rot.shape[0] == dof_pos.shape[0]):
            raise ValueError(
                f"{self.motion_file}: frame count mismatch, "
                f"root_pos={root_pos.shape[0]}, root_rot={root_rot.shape[0]}, dof_pos={dof_pos.shape[0]}"
            )

        self.output_fps = int(self.requested_output_fps) if self.requested_output_fps is not None else int(round(self.input_fps))

        if self.output_fps <= 0:
            raise ValueError(f"output_fps must be positive, but got {self.output_fps}")

        self.input_dt = 1.0 / float(self.input_fps)
        self.output_dt = 1.0 / float(self.output_fps)

        self.motion_base_poss_input = torch.from_numpy(root_pos).to(torch.float32).to(self.device)

        root_rot_tensor = torch.from_numpy(root_rot).to(torch.float32).to(self.device)
        self.motion_base_rots_input = reorder_quaternion(
            root_rot_tensor,
            src_order=self.input_root_rot_order,
            dst_order="wxyz",
        )

        self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(torch.float32).to(self.device)

        self.input_frames = self.motion_base_poss_input.shape[0]
        if self.input_frames < 1:
            raise ValueError(f"{self.motion_file}: empty motion.")

        self.duration = (self.input_frames - 1) * self.input_dt

        print(
            f"input_frames={self.input_frames}, input_fps={self.input_fps}, "
            f"output_fps={self.output_fps}, duration={self.duration:.6f}s"
        )
    # =========== newADD end ========

    def _interpolate_motion(self):
        """Interpolates the motion to the output fps."""
        times = torch.arange(0, self.duration, self.output_dt, device=self.device, dtype=torch.float32)
        self.output_frames = times.shape[0]
        index_0, index_1, blend = self._compute_frame_blend(times)
        self.motion_base_poss = self._lerp(
            self.motion_base_poss_input[index_0],
            self.motion_base_poss_input[index_1],
            blend.unsqueeze(1),
        )
        self.motion_base_rots = self._slerp(
            self.motion_base_rots_input[index_0],
            self.motion_base_rots_input[index_1],
            blend,
        )
        self.motion_dof_poss = self._lerp(
            self.motion_dof_poss_input[index_0],
            self.motion_dof_poss_input[index_1],
            blend.unsqueeze(1),
        )
        print(
            f"Motion interpolated, input frames: {self.input_frames}, input fps: {self.input_fps}, output frames:"
            f" {self.output_frames}, output fps: {self.output_fps}"
        )

    def _lerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Linear interpolation between two tensors."""
        return a * (1 - blend) + b * blend

    def _slerp(self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
        """Spherical linear interpolation between two quaternions."""
        slerped_quats = torch.zeros_like(a)
        for i in range(a.shape[0]):
            slerped_quats[i] = quat_slerp(a[i], b[i], blend[i])
        return slerped_quats

    def _compute_frame_blend(self, times: torch.Tensor) -> torch.Tensor:
        """Computes the frame blend for the motion."""
        phase = times / self.duration
        index_0 = (phase * (self.input_frames - 1)).floor().long()
        index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
        blend = phase * (self.input_frames - 1) - index_0
        return index_0, index_1, blend

    def _compute_velocities(self):
        """Computes the velocities of the motion."""
        self.motion_base_lin_vels = torch.gradient(self.motion_base_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_dof_vels = torch.gradient(self.motion_dof_poss, spacing=self.output_dt, dim=0)[0]
        self.motion_base_ang_vels = self._so3_derivative(self.motion_base_rots, self.output_dt)

    def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
        """Computes the derivative of a sequence of SO3 rotations.

        Args:
            rotations: shape (B, 4).
            dt: time step.
        Returns:
            shape (B, 3).
        """
        q_prev, q_next = rotations[:-2], rotations[2:]
        q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

        omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
        omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)  # repeat first and last sample
        return omega

    def get_next_state(
        self,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """Gets the next state of the motion."""
        state = (
            self.motion_base_poss[self.current_idx : self.current_idx + 1],
            self.motion_base_rots[self.current_idx : self.current_idx + 1],
            self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
            self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
            self.motion_dof_poss[self.current_idx : self.current_idx + 1],
            self.motion_dof_vels[self.current_idx : self.current_idx + 1],
        )
        self.current_idx += 1
        reset_flag = False
        if self.current_idx >= self.output_frames:
            self.current_idx = 0
            reset_flag = True
        return state, reset_flag


def run_simulator(
    sim: sim_utils.SimulationContext,
    scene: InteractiveScene, 
    joint_names: list[str],
    motion_file: str,
    save_file: str,
):
    """Runs the simulation loop."""
    # Load motion
    motion = MotionLoader(
        motion_file=motion_file,
        input_fps=args_cli.input_fps,
        output_fps=args_cli.output_fps,
        device=sim.device,
        input_root_rot_order=args_cli.input_root_rot_order,
 
    )

    # Extract scene entities
    robot = scene["robot"]
    robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

    # ------- data logger -------------------------------------------------------
    log = {
        "fps": [args_cli.output_fps],
        "joint_pos": [],
        "joint_vel": [],
        "body_pos_w": [],
        "body_quat_w": [],
        "body_lin_vel_w": [],
        "body_ang_vel_w": [],
    }
    file_saved = False
    # --------------------------------------------------------------------------

    # Simulation loop
    while simulation_app.is_running():
        (
            (
                motion_base_pos,
                motion_base_rot,
                motion_base_lin_vel,
                motion_base_ang_vel,
                motion_dof_pos,
                motion_dof_vel,
            ),
            reset_flag,
        ) = motion.get_next_state()

        # set root state
        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion_base_pos
        root_states[:, :2] += scene.env_origins[:, :2]
        root_states[:, 3:7] = motion_base_rot
        root_states[:, 7:10] = motion_base_lin_vel
        root_states[:, 10:] = motion_base_ang_vel
        robot.write_root_state_to_sim(root_states)

        # set joint state
        joint_pos = robot.data.default_joint_pos.clone()
        joint_vel = robot.data.default_joint_vel.clone()
        joint_pos[:, robot_joint_indexes] = motion_dof_pos
        joint_vel[:, robot_joint_indexes] = motion_dof_vel
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim.get_physics_dt())
        # Isaac Lab 自动执行 Forward Kinematics（FK），基于当前的关节角 & 根状态，计算所有 “link”（rigid bodies）的 位置、旋转、速度

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

        # save_file=args_cli.output_path

        if not file_saved:
            log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
            log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
            log["body_pos_w"].append(robot.data.body_pos_w[0, :].cpu().numpy().copy())
            log["body_quat_w"].append(robot.data.body_quat_w[0, :].cpu().numpy().copy())
            log["body_lin_vel_w"].append(robot.data.body_lin_vel_w[0, :].cpu().numpy().copy())
            log["body_ang_vel_w"].append(robot.data.body_ang_vel_w[0, :].cpu().numpy().copy())

        if reset_flag and not file_saved:
            file_saved = True
            for k in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                log[k] = np.stack(log[k], axis=0)

            np.savez(save_file, **log)
           
            print(f"[INFO]:Motion saved to",save_file)
            print(f"[INFO]:Completed one motion, continue...")
            # print(f"Ctrl+C to exit")
            return



def main():
    """Main function."""
    # Load kit helper
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 1.0 / args_cli.output_fps
    sim = SimulationContext(sim_cfg)
    # Design scene
    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    # Play the simulator
    sim.reset()

    input_path = Path(args_cli.input_path).expanduser().resolve()
    output_path = Path(args_cli.output_path).expanduser().resolve()

    input_is_file = input_path.is_file()
    input_root = input_path.parent if input_is_file else input_path

    # files = collect_input_files(input_path, args_cli.pattern)
    # print(f"[INFO] Found {len(files)} file(s) to process.")
    # ======== newADD start======
    files = collect_input_files(input_path, args_cli.pattern)
    total_n = len(files)
    skip_n = 0
    print(f"[INFO] Found {total_n} file(s) to process.")
    # =========== newADD end ========

    # for src_file in files:
    # ======== newADD start======
    for processing_n, src_file in enumerate(files, start=1):
    # =========== newADD end ========
        dst_file = build_output_file_path(
            src_file=src_file,
            input_root=input_root,
            output_path=output_path,
            suffix=args_cli.suffix,
            input_is_file=input_is_file,
        )

        # if args_cli.skip_existing and dst_file.exists():
        #     print(f"[SKIP] Output already exists: {dst_file}")
        #     continue
        # ======== newADD start======
        if args_cli.skip_existing and dst_file.exists():
            skip_n += 1
            print("============================================================")
            print(f"[SKIP] processing={processing_n}/{total_n}, skip_n={skip_n}")
            print(f"[SKIP] Output already exists: {dst_file}")
            continue
        # =========== newADD end ========

        dst_file.parent.mkdir(parents=True, exist_ok=True)

        # print("============================================================")
        # print(f"[INFO] Converting...........:")
        # print(f"  input : {src_file}")
        # print(f"  output: {dst_file}")
        print("============================================================")
        # ======== newADD start======
        print(f"[INFO] processing={processing_n}/{total_n}, skip_n={skip_n}")
        # =========== newADD end ========
        print(f"[INFO] Converting...........:")
        print(f"  input : {src_file}")
        print(f"  output: {dst_file}")


        # Run the simulator
        run_simulator(
            sim,
            scene,
            joint_names=[
                "left_hip_pitch_joint",
                "left_hip_roll_joint",
                "left_hip_yaw_joint",
                "left_knee_joint",
                "left_ankle_pitch_joint",
                "left_ankle_roll_joint",
                "right_hip_pitch_joint",
                "right_hip_roll_joint",
                "right_hip_yaw_joint",
                "right_knee_joint",
                "right_ankle_pitch_joint",
                "right_ankle_roll_joint",
                "waist_yaw_joint",
                "waist_roll_joint",
                "waist_pitch_joint",
                "left_shoulder_pitch_joint",
                "left_shoulder_roll_joint",
                "left_shoulder_yaw_joint",
                "left_elbow_joint",
                "left_wrist_roll_joint",
                "left_wrist_pitch_joint",
                "left_wrist_yaw_joint",
                "right_shoulder_pitch_joint",
                "right_shoulder_roll_joint",
                "right_shoulder_yaw_joint",
                "right_elbow_joint",
                "right_wrist_roll_joint",
                "right_wrist_pitch_joint",
                "right_wrist_yaw_joint",
            ],
            motion_file=str(src_file),
            save_file=str(dst_file),
        )


if __name__ == "__main__":
    # run the main function
    main()
    # Isaac Sim 4.5 may block indefinitely in simulation_app.close() after a
    # headless, render-only conversion.  The NPZ is already closed by
    # np.savez; flush logs and let process teardown release simulator resources.
    import os
    import sys

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
    
