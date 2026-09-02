"""
Usage:
conda activate gmr
# 单个 npz
python /home/pcbysan/BysanRL/BeyondMimic/whole_body_tracking/scripts/replay_npz_local.py \
    --motion_path /home/pcbysan/BysanRL/data_goal/npz/walk1_subject1.npz

# 一个目录（会递归查找所有 .npz）
python /home/pcbysan/BysanRL/BeyondMimic/whole_body_tracking/scripts/replay_npz_local.py \
    --motion_path /data/BysanRL/data_goal/npz/amass_proto \
    --device cpu

Controls:
    1 : playback speed -0.5x
    2 : playback speed +0.5x
    5 : previous npz
    6 : next npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
# ======== newADD start======
parser.add_argument(
    "--motion_path",
    "--motion_file",
    dest="motion_path",
    type=str,
    required=True,
    help="Path to a .npz file or a directory containing .npz files.",
)
# =========== newADD end ========

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import carb.input
import omni.appwindow

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

##
# Pre-defined configs
##
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.mdp import MotionLoader


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

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
def collect_motion_files(motion_path: str) -> list[str]:
    """
    Resolve the input path into a sorted list of .npz files.

    - If `motion_path` is a file, it must be a `.npz`.
    - If `motion_path` is a directory, all `.npz` files under it will be
      collected recursively and sorted by path string.
    """
    path = Path(motion_path).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(f"motion_path does not exist: {path}")

    if path.is_file():
        if path.suffix.lower() != ".npz":
            raise ValueError(f"Expected a .npz file, but got: {path}")
        return [str(path)]

    motion_files = sorted(str(p) for p in path.rglob("*.npz"))
    if len(motion_files) == 0:
        raise FileNotFoundError(f"No .npz files found under directory: {path}")

    return motion_files


class MotionPlaylistController:
    """
    Manage:
    1) motion file playlist
    2) keyboard events
    3) playback speed
    4) fractional frame cursor
    """

    def __init__(self, motion_path: str, num_envs: int, device: str):
        self.motion_files = collect_motion_files(motion_path)
        self.num_envs = num_envs
        self.device = device

        self.env_ids = torch.arange(num_envs, dtype=torch.long, device=device)

        self.motion_idx = 0
        self.motion = None

        self.playback_speed = 1.0
        self.time_cursor = torch.zeros(num_envs, dtype=torch.float32, device=device)

        self._input = None
        self._keyboard = None
        self._keyboard_sub = None

        self._setup_keyboard()
        self._load_motion(self.motion_idx, reset_time=True)
        self._print_help()

    def _setup_keyboard(self):
        self._input = carb.input.acquire_input_interface()
        app_window = omni.appwindow.get_default_app_window()
        if app_window is None:
            raise RuntimeError("Failed to acquire default app window for keyboard input.")
        self._keyboard = app_window.get_keyboard()
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard, self._on_keyboard_event
        )

    def close(self):
        if self._input is not None and self._keyboard is not None and self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _print_help(self):
        print("=" * 80)
        print("[Controls]")
        print("  1 : playback speed -0.5x")
        print("  2 : playback speed +0.5x")
        print("  5 : previous npz")
        print("  6 : next npz")
        print("=" * 80)

    def _get_total_frames(self) -> int:
        total = self.motion.time_step_total
        if isinstance(total, torch.Tensor):
            total = total.reshape(-1)[0].item()
        return int(total)

    def _load_motion(self, motion_idx: int, reset_time: bool = True):
        self.motion_idx = motion_idx % len(self.motion_files)
        motion_file = self.motion_files[self.motion_idx]

        self.motion = MotionLoader(
            motion_file,
            self.env_ids,
            self.device,
        )

        if reset_time:
            self.time_cursor.zero_()

        print(
            f"[MOTION] ({self.motion_idx + 1}/{len(self.motion_files)}) "
            f"{motion_file}"
        )
        print(
            f"[STATE] total_frames={self._get_total_frames()} | "
            f"playback_speed={self.playback_speed:.1f}x"
        )

    def _change_speed(self, delta: float):
        self.playback_speed = max(0.5, self.playback_speed + delta)
        print(f"[SPEED] playback_speed = {self.playback_speed:.1f}x")

    def _switch_motion(self, step: int):
        self._load_motion(self.motion_idx + step, reset_time=True)

    def _on_keyboard_event(self, event):
        if event.type != carb.input.KeyboardEventType.KEY_PRESS:
            return False

        if event.input == carb.input.KeyboardInput.KEY_1:
            self._change_speed(-0.5)
            return True
        elif event.input == carb.input.KeyboardInput.KEY_2:
            self._change_speed(+0.5)
            return True
        elif event.input == carb.input.KeyboardInput.KEY_5:
            self._switch_motion(-1)
            return True
        elif event.input == carb.input.KeyboardInput.KEY_6:
            self._switch_motion(+1)
            return True

        return False

    def get_frame_ids(self) -> torch.Tensor:
        total_frames = self._get_total_frames()
        if total_frames <= 0:
            raise RuntimeError("Loaded motion has non-positive total frame count.")

        frame_ids = torch.floor(self.time_cursor).long()
        frame_ids = torch.clamp(frame_ids, min=0, max=total_frames - 1)
        return frame_ids

    def advance(self):
        total_frames = float(self._get_total_frames())
        self.time_cursor += self.playback_speed
        if total_frames > 0:
            self.time_cursor.remainder_(total_frames)
# =========== newADD end ========


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    # ======== newADD start======
    controller = MotionPlaylistController(
        motion_path=args_cli.motion_path,
        num_envs=scene.num_envs,
        device=sim.device,
    )
    # =========== newADD end ========

    try:
        # Simulation loop
        while simulation_app.is_running():
            # ======== newADD start======
            motion = controller.motion
            frame_ids = controller.get_frame_ids()
            # =========== newADD end ========

            root_states = robot.data.default_root_state.clone()
            # ======== newADD start======
            root_states[:, :3] = motion.body_pos_w[frame_ids][:, 0] + scene.env_origins
            root_states[:, 3:7] = motion.body_quat_w[frame_ids][:, 0]
            root_states[:, 7:10] = motion.body_lin_vel_w[frame_ids][:, 0]
            root_states[:, 10:] = motion.body_ang_vel_w[frame_ids][:, 0]

            robot.write_root_state_to_sim(root_states)
            robot.write_joint_state_to_sim(motion.joint_pos[frame_ids], motion.joint_vel[frame_ids])
            # =========== newADD end ========

            scene.write_data_to_sim()
            sim.render()  # We don't want physic (sim.step())
            scene.update(sim_dt)

            pos_lookat = root_states[0, :3].detach().cpu().numpy()
            sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)

            # ======== newADD start======
            controller.advance()
            # =========== newADD end ========

    finally:
        # ======== newADD start======
        controller.close()
        # =========== newADD end ========


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()