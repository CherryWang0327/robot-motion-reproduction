#!/usr/bin/env python3
"""Retarget either a GVHMR PT or canonical SMPL CSV to a robot motion."""

import argparse
import pathlib
import pickle
import time

import numpy as np

from general_motion_retargeting import GeneralMotionRetargeting as GMR
from general_motion_retargeting import RobotMotionViewer
from general_motion_retargeting.utils.smpl import (
    get_gvhmr_data_offline_fast,
    load_gvhmr_pred_file,
    load_smpl_csv_file,
)


ROBOTS = [
    "unitree_g1", "unitree_g1_with_hands", "unitree_h1", "unitree_h1_2",
    "booster_t1", "booster_t1_29dof", "stanford_toddy", "fourier_n1",
    "engineai_pm01", "kuavo_s45", "hightorque_hi", "galaxea_r1pro",
    "berkeley_humanoid_lite", "booster_k1", "pnd_adam_lite", "openloong",
    "tienkung", "fourier_gr3", "pal_talos",
]


def save_robot_motion(path, qpos_list, fps):
    """Save the established GMR schema as pickle or a field-equivalent NPZ."""
    qpos = np.asarray(qpos_list)
    motion_data = {
        "fps": float(fps),
        "root_pos": qpos[:, :3],
        "root_rot": qpos[:, 3:7][:, [1, 2, 3, 0]],  # GMR files use xyzw
        "dof_pos": qpos[:, 7:],
        "local_body_pos": None,
        "link_body_list": None,
    }
    output = pathlib.Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.lower() == ".npz":
        # Only persist actual numeric trajectory fields; the two optional fields are None.
        np.savez(output, fps=motion_data["fps"], root_pos=motion_data["root_pos"],
                 root_rot=motion_data["root_rot"], dof_pos=motion_data["dof_pos"])
    elif output.suffix.lower() in (".pkl", ".pickle"):
        with output.open("wb") as handle:
            pickle.dump(motion_data, handle)
    else:
        raise ValueError("robot output must end in .pkl, .pickle, or .npz")
    print(f"Saved to {output}")


def build_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="GVHMR .pt or canonical SMPL .csv (format inferred)")
    parser.add_argument("--gvhmr_pred_file", help="Legacy alias for a GVHMR .pt input")
    parser.add_argument("--robot", choices=ROBOTS, default="unitree_g1")
    parser.add_argument("--output", "--save_path", dest="output", help="Robot .pkl or .npz output")
    parser.add_argument("--fps", type=float, help="PT source FPS; also validates CSV FPS if supplied")
    parser.add_argument("--target_fps", type=float, default=30, help="GMR processing FPS (default: 30)")
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--no_viewer", action="store_true", help="Run retargeting without a MuJoCo window")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    input_path = args.input or args.gvhmr_pred_file
    if not input_path:
        raise ValueError("provide --input or the legacy --gvhmr_pred_file")
    if args.input and args.gvhmr_pred_file:
        raise ValueError("--input and --gvhmr_pred_file are mutually exclusive")
    if args.loop and args.no_viewer:
        raise ValueError("--loop cannot be combined with --no_viewer")

    model_folder = pathlib.Path(__file__).parent / ".." / "assets" / "body_models"
    if pathlib.Path(input_path).suffix.lower() == ".csv":
        loaded = load_smpl_csv_file(input_path, model_folder, expected_fps=args.fps)
    else:
        source_fps = args.fps
        if args.gvhmr_pred_file and source_fps is None:
            # Preserve the historical GVHMR entry point, whose pipeline is fixed at 30 FPS.
            source_fps = 30
            print("Legacy --gvhmr_pred_file input: using GVHMR's historical 30 FPS default")
        loaded = load_gvhmr_pred_file(input_path, model_folder, fps=source_fps)
    smplx_data, body_model, smplx_output, actual_human_height = loaded
    frames, aligned_fps = get_gvhmr_data_offline_fast(
        smplx_data, body_model, smplx_output, tgt_fps=args.target_fps
    )
    retarget = GMR(actual_human_height=actual_human_height, src_human="smplx", tgt_robot=args.robot)

    viewer = None
    if not args.no_viewer:
        stem = pathlib.Path(input_path).stem
        viewer = RobotMotionViewer(
            robot_type=args.robot, motion_fps=aligned_fps, transparent_robot=0,
            record_video=args.record_video, video_path=f"videos/{args.robot}_{stem}.mp4",
        )

    qpos_list = []
    index = 0
    fps_counter = 0
    fps_start = time.time()
    try:
        while args.loop or index < len(frames):
            frame = frames[index % len(frames)]
            qpos = retarget.retarget(frame)
            qpos_list.append(qpos)
            if viewer is not None:
                viewer.step(root_pos=qpos[:3], root_rot=qpos[3:7], dof_pos=qpos[7:],
                            human_motion_data=retarget.scaled_human_data,
                            human_pos_offset=np.zeros(3), show_human_body_name=False,
                            rate_limit=args.rate_limit)
            index += 1
            fps_counter += 1
            if viewer is not None and time.time() - fps_start >= 2:
                print(f"Actual rendering FPS: {fps_counter / (time.time() - fps_start):.2f}")
                fps_counter, fps_start = 0, time.time()
    finally:
        if viewer is not None:
            viewer.close()
    if args.output:
        save_robot_motion(args.output, qpos_list, aligned_fps)


if __name__ == "__main__":
    main()
