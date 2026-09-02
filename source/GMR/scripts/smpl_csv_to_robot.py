#!/usr/bin/env python3
"""Canonical SMPL CSV entry point for GMR retargeting."""

import argparse

from gvhmr_to_robot import ROBOTS, main as retarget_main


def main():
    parser = argparse.ArgumentParser(description="Retarget canonical SMPL CSV to a robot.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--robot", choices=ROBOTS, default="unitree_g1")
    parser.add_argument("--output", required=True, help="Output .pkl or standard field-equivalent .npz")
    parser.add_argument("--fps", type=float, help="Optional expected CSV FPS")
    parser.add_argument("--target_fps", type=float, default=30)
    parser.add_argument("--record_video", action="store_true")
    parser.add_argument("--rate_limit", action="store_true")
    parser.add_argument("--no_viewer", action="store_true")
    args = parser.parse_args()
    forwarded = ["--input", args.input, "--robot", args.robot, "--output", args.output,
                 "--target_fps", str(args.target_fps)]
    if args.fps is not None:
        forwarded += ["--fps", str(args.fps)]
    for flag in ("record_video", "rate_limit", "no_viewer"):
        if getattr(args, flag):
            forwarded.append("--" + flag)
    retarget_main(forwarded)


if __name__ == "__main__":
    main()
