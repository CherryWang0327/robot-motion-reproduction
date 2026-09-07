from __future__ import annotations

import argparse
from pathlib import Path

from .convert import convert_to_wbt
from .front import build_from_video
from .train import plan_training
from .validate import validate_npz, write_report


def main() -> None:
    parser = argparse.ArgumentParser(prog="g1-pipeline")
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Validate a WBT NPZ motion without training or robot access")
    validate.add_argument("--motion", required=True, type=Path)
    validate.add_argument("--output", required=True, type=Path)
    validate.add_argument("--expected-fps", type=float, default=50.0)
    convert = subparsers.add_parser("convert", help="Convert GMR PKL or ProtoMotions PT to a local WBT NPZ")
    convert.add_argument("--route", choices=("gmr", "protomotions"), required=True)
    convert.add_argument("--input", required=True, type=Path)
    convert.add_argument("--run-dir", required=True, type=Path)
    convert.add_argument("--dry-run", action="store_true")
    train = subparsers.add_parser("train", help="Validate and plan a local Whole-Body Tracking training run")
    train.add_argument("--motion", required=True, type=Path)
    train.add_argument("--run-dir", required=True, type=Path)
    train.add_argument("--run-name", required=True)
    train.add_argument("--max-iterations", type=int, default=100000)
    train.add_argument("--num-envs", type=int, default=4096)
    train.add_argument("--execute", action="store_true", help="Actually start the long-running simulation job")
    build = subparsers.add_parser("build", help="Run video -> GVHMR -> GMR or ProtoMotions/PyRoki")
    build.add_argument("--video", required=True, type=Path)
    build.add_argument("--route", choices=("gmr", "protomotions"), required=True)
    build.add_argument("--run-dir", required=True, type=Path)
    build.add_argument("--dry-run", action="store_true")
    run = subparsers.add_parser("run", help="Run video -> retarget -> WBT NPZ -> validation (no training)")
    run.add_argument("--video", required=True, type=Path)
    run.add_argument("--route", choices=("gmr", "protomotions"), required=True)
    run.add_argument("--run-dir", required=True, type=Path)
    run.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    if args.command == "validate":
        report = validate_npz(args.motion, args.expected_fps)
        write_report(report, args.output)
        print(f"{report.status}: {args.motion} -> {args.output}")
        for finding in report.findings:
            print(f"{finding.level}: {finding.check}: {finding.message}")
        raise SystemExit({"PASS": 0, "WARN": 2, "FAIL": 1}[report.status])
    if args.command == "convert":
        output = convert_to_wbt(args.route, args.input, args.run_dir, args.dry_run)
        print(f"{'DRY_RUN' if args.dry_run else 'DONE'}: {args.route} -> {output}")
        return
    if args.command == "build":
        output = build_from_video(args.video, args.route, args.run_dir, args.dry_run)
        print(f"{'DRY_RUN' if args.dry_run else 'DONE'}: {args.route} -> {output}")
        return
    if args.command == "run":
        retargeted = build_from_video(args.video, args.route, args.run_dir, args.dry_run)
        output = convert_to_wbt(args.route, retargeted, args.run_dir, args.dry_run)
        print(f"{'DRY_RUN' if args.dry_run else 'DONE'}: {args.route} -> {output}")
        return
    plan_training(args.motion, args.run_dir, args.run_name, args.max_iterations, args.num_envs, args.execute)


if __name__ == "__main__":
    main()
