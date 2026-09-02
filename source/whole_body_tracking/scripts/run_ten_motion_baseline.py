#!/usr/bin/env python3
"""Run the fixed ten-motion baseline sequentially with plateau stopping and evaluation."""

from __future__ import annotations

import argparse
import json
import math
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

from analyze_training_convergence import analyze


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LOG_ROOT = PROJECT_ROOT / "logs" / "rsl_rl" / "g1_flat"
RESULT_ROOT = PROJECT_ROOT / "results" / "baseline"
MOTIONS = {
    "1438": PROJECT_ROOT / "inputs/1438/1438_gmr_50fps.npz",
    "1439": PROJECT_ROOT / "inputs/1439/1439_gmr_50fps.npz",
    "happy": PROJECT_ROOT / "inputs/happy/happy_gmr_50fps.npz",
    "taichi1": PROJECT_ROOT / "inputs/taichi1/taichi1_gmr_50fps.npz",
    "zhu0129": PROJECT_ROOT / "inputs/zhu0129/zhu0129_gmr_50fps.npz",
    "zhu0201_02_whswap_v2": PROJECT_ROOT
    / "inputs/zhu0201_02_whswap_v2/zhu0201_02_whswap_v2_gmr_50fps.npz",
    "zhu0201_03": PROJECT_ROOT / "inputs/zhu0201_03/zhu0201_03_gmr_50fps.npz",
    "zhu0201_04": PROJECT_ROOT / "inputs/zhu0201_04/zhu0201_04_gmr_50fps.npz",
    "zhu0201_upright_50fps": PROJECT_ROOT
    / "inputs/zhu0201_upright_50fps/zhu0201_upright_50fps_gmr_50fps.npz",
    "zhu0202_01": PROJECT_ROOT / "inputs/zhu0202_01/zhu0202_01_gmr_50fps.npz",
}


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {"protocol": {}, "motions": {}}
    return json.loads(path.read_text())


def _save_state(path: Path, state: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


def _checkpoints(run_dir: Path) -> list[Path]:
    checkpoints = []
    for path in run_dir.glob("model_*.pt"):
        try:
            iteration = int(path.stem.removeprefix("model_"))
        except ValueError:
            continue
        checkpoints.append((iteration, path))
    return [path for _, path in sorted(checkpoints)]


def _checkpoint_iteration(path: Path) -> int:
    return int(path.stem.removeprefix("model_"))


def _training_process_is_alive(entry: dict) -> bool:
    pid = entry.get("training_pid")
    if not isinstance(pid, int):
        return False
    try:
        command = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\x00", b" ").decode(errors="replace")
    except OSError:
        return False
    return "scripts/rsl_rl/train.py" in command


def _wait_for_run_dir(run_name: str, previous: set[Path], process: subprocess.Popen, poll_seconds: float) -> Path:
    pattern = f"*_{run_name}"
    while process.poll() is None:
        candidates = set(LOG_ROOT.glob(pattern)) - previous
        if candidates:
            return max(candidates, key=lambda path: path.stat().st_mtime)
        time.sleep(poll_seconds)
    raise RuntimeError(f"training exited before creating a run directory (exit code {process.returncode})")


def _stop_training(process: subprocess.Popen, timeout: float = 120.0) -> int:
    if process.poll() is not None:
        return process.returncode
    print("[ORCHESTRATOR] Sending SIGINT after a complete converged checkpoint was saved.", flush=True)
    os.killpg(process.pid, signal.SIGINT)
    try:
        return process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        print("[ORCHESTRATOR] Training did not exit after SIGINT; sending SIGTERM.", flush=True)
        os.killpg(process.pid, signal.SIGTERM)
        return process.wait(timeout=30.0)


def _run_checked(command: list[str], dry_run: bool) -> None:
    print("[ORCHESTRATOR] " + " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)


def _train_motion(
    motion: str,
    motion_file: Path,
    state_entry: dict,
    state_path: Path,
    state: dict,
    args: argparse.Namespace,
) -> tuple[Path, Path, dict]:
    run_name = f"baseline_{motion}_100k"
    command = [
        sys.executable,
        "scripts/rsl_rl/train.py",
        "--task",
        args.task,
        "--motion_file",
        str(motion_file),
        "--num_envs",
        str(args.num_envs),
        "--max_iterations",
        str(args.max_iterations),
        "--seed",
        "0",
        "--headless",
        "--logger",
        "tensorboard",
        "--run_name",
        run_name,
    ]
    resume_checkpoint = None
    previous_run = Path(state_entry["run_directory"]) if state_entry.get("run_directory") else None
    if state_entry.get("stage") in {"training", "interrupted"} and previous_run and previous_run.is_dir():
        previous_checkpoints = _checkpoints(previous_run)
        if previous_checkpoints:
            resume_checkpoint = previous_checkpoints[-1]
            command.extend(
                [
                    "--resume_checkpoint",
                    str(resume_checkpoint.resolve()),
                    "--resume_log_dir",
                    str(previous_run.resolve()),
                ]
            )
            print(
                f"[ORCHESTRATOR] Resuming {motion} from {resume_checkpoint} "
                f"(iteration {_checkpoint_iteration(resume_checkpoint)}).",
                flush=True,
            )
    print("[ORCHESTRATOR] " + " ".join(command), flush=True)
    if args.dry_run:
        return LOG_ROOT / f"<timestamp>_{run_name}", Path("model_<iteration>.pt"), {}

    previous = set(LOG_ROOT.glob(f"*_{run_name}"))
    process = subprocess.Popen(command, cwd=PROJECT_ROOT, start_new_session=True)
    run_dir = previous_run if resume_checkpoint is not None else _wait_for_run_dir(
        run_name, previous, process, args.poll_seconds
    )
    state_entry.update(stage="training", run_directory=str(run_dir.resolve()), training_pid=process.pid)
    _save_state(state_path, state)
    print(f"[ORCHESTRATOR] Run directory: {run_dir}", flush=True)

    convergence_report = None
    target_checkpoint_iteration = None
    try:
        while process.poll() is None:
            try:
                report = analyze(run_dir, args.max_iterations, args.window, args.min_iterations)
                print(
                    f"[ORCHESTRATOR] {motion}: iteration={report['iteration']} "
                    f"plateau={report['plateau']} converged={report['converged']}",
                    flush=True,
                )
                state_entry.update(
                    latest_iteration=report["iteration"],
                    plateau=report["plateau"],
                    converged=report["converged"],
                    latest_checkpoint=report["latest_checkpoint"],
                )
                _save_state(state_path, state)
                if report["converged"] and target_checkpoint_iteration is None:
                    convergence_report = report
                    target_checkpoint_iteration = int(math.ceil(report["iteration"] / 500.0) * 500)
                    print(
                        f"[ORCHESTRATOR] Plateau confirmed; waiting for checkpoint >= "
                        f"model_{target_checkpoint_iteration}.pt",
                        flush=True,
                    )
            except (OSError, ValueError) as exc:
                print(f"[ORCHESTRATOR] Waiting for complete TensorBoard metrics: {exc}", flush=True)

            checkpoints = _checkpoints(run_dir)
            if target_checkpoint_iteration is not None and checkpoints:
                checkpoint = checkpoints[-1]
                if _checkpoint_iteration(checkpoint) >= target_checkpoint_iteration:
                    _stop_training(process)
                    return run_dir, checkpoint, convergence_report or {}
            time.sleep(args.poll_seconds)
    except KeyboardInterrupt:
        _stop_training(process)
        state_entry["stage"] = "interrupted"
        _save_state(state_path, state)
        raise

    checkpoints = _checkpoints(run_dir)
    if not checkpoints:
        raise RuntimeError(f"training ended without a checkpoint: {run_dir}")
    checkpoint = checkpoints[-1]
    report = analyze(run_dir, args.max_iterations, args.window, args.min_iterations)
    return run_dir, checkpoint, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--motions", nargs="+", choices=MOTIONS, default=list(MOTIONS))
    parser.add_argument("--num_envs", type=int, default=4096)
    parser.add_argument("--task", default="Tracking-Flat-G1-v0")
    parser.add_argument("--min_iterations", type=int, default=30000)
    parser.add_argument("--max_iterations", type=int, default=100000)
    parser.add_argument("--window", type=int, default=1000)
    parser.add_argument("--poll_seconds", type=float, default=60.0)
    parser.add_argument("--state_file", type=Path, default=RESULT_ROOT / "orchestrator_state.json")
    parser.add_argument("--skip_evaluation", action="store_true")
    parser.add_argument(
        "--adopt_completed_run",
        action="append",
        default=[],
        metavar="MOTION=RUN_DIRECTORY",
        help="Reuse a completed external run instead of retraining that motion (repeatable).",
    )
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()
    if args.min_iterations < 2 * args.window:
        parser.error("--min_iterations must be at least two convergence windows")
    if args.max_iterations < args.min_iterations:
        parser.error("--max_iterations must be >= --min_iterations")

    missing = [str(path) for motion, path in MOTIONS.items() if motion in args.motions and not path.is_file()]
    if missing:
        parser.error("missing motion files: " + ", ".join(missing))

    state_path = args.state_file.expanduser().resolve()
    state = _load_state(state_path)
    state["protocol"] = {
        "seed": 0,
        "num_envs": args.num_envs,
        "min_iterations": args.min_iterations,
        "max_iterations": args.max_iterations,
        "window": args.window,
        "task": args.task,
    }
    for adoption in args.adopt_completed_run:
        if "=" not in adoption:
            parser.error("--adopt_completed_run must use MOTION=RUN_DIRECTORY")
        motion, run_text = adoption.split("=", 1)
        if motion not in MOTIONS:
            parser.error(f"unknown adopted motion: {motion}")
        if state.get("motions", {}).get(motion, {}).get("stage") == "complete":
            continue
        run_dir = Path(run_text).expanduser().resolve()
        checkpoints = _checkpoints(run_dir) if run_dir.is_dir() else []
        if not checkpoints:
            parser.error(f"adopted run has no checkpoint: {run_dir}")
        checkpoint = checkpoints[-1]
        try:
            adopted_report = analyze(run_dir, args.max_iterations, args.window, args.min_iterations)
        except (OSError, ValueError) as exc:
            parser.error(f"cannot analyze adopted run {run_dir}: {exc}")
        if not adopted_report["converged"] and _checkpoint_iteration(checkpoint) + 1 < args.max_iterations:
            parser.error(
                f"adopted run is not converged and has not reached {args.max_iterations} iterations; "
                "resume it instead of adopting it"
            )
        state["motions"].setdefault(motion, {}).update(
            stage="reporting",
            run_directory=str(run_dir),
            final_checkpoint=str(checkpoint.resolve()),
            final_iteration=_checkpoint_iteration(checkpoint),
            converged=adopted_report["converged"],
        )
    _save_state(state_path, state)

    for motion in args.motions:
        entry = state["motions"].setdefault(motion, {})
        if entry.get("stage") == "complete":
            print(f"[ORCHESTRATOR] Skipping completed motion: {motion}", flush=True)
            continue
        if entry.get("stage") == "training" and _training_process_is_alive(entry):
            parser.error(
                f"{motion} training process PID {entry['training_pid']} is still running; "
                "do not start a duplicate orchestrator"
            )
        motion_file = MOTIONS[motion]
        try:
            saved_run = Path(entry["run_directory"]) if entry.get("run_directory") else None
            saved_checkpoint = Path(entry["final_checkpoint"]) if entry.get("final_checkpoint") else None
            reuse_finished_training = bool(
                entry.get("stage") in {"reporting", "evaluating", "failed"}
                and saved_run is not None
                and saved_run.is_dir()
                and saved_checkpoint is not None
                and saved_checkpoint.is_file()
            )
            if reuse_finished_training:
                print(f"[ORCHESTRATOR] Reusing finished training for {motion}: {saved_run}", flush=True)
                run_dir, checkpoint = saved_run, saved_checkpoint
                report = analyze(run_dir, args.max_iterations, args.window, args.min_iterations)
            else:
                run_dir, checkpoint, report = _train_motion(
                    motion, motion_file, entry, state_path, state, args
                )
            if args.dry_run:
                continue
            entry.update(
                stage="reporting",
                final_checkpoint=str(checkpoint.resolve()),
                final_iteration=_checkpoint_iteration(checkpoint),
                converged=bool(report.get("converged", False)),
            )
            _save_state(state_path, state)

            _run_checked(
                [
                    sys.executable,
                    "scripts/export_training_report.py",
                    "--log_dir",
                    str(run_dir),
                    "--max_iterations",
                    str(args.max_iterations),
                    "--min_iterations",
                    str(args.min_iterations),
                    "--window",
                    str(args.window),
                ],
                False,
            )
            if not args.skip_evaluation:
                entry["stage"] = "evaluating"
                _save_state(state_path, state)
                evaluation_dir = RESULT_ROOT / motion / "evaluation"
                _run_checked(
                    [
                        sys.executable,
                        "scripts/rsl_rl/play.py",
                        "--task",
                        args.task,
                        "--motion_file",
                        str(motion_file),
                        "--load_run",
                        run_dir.name,
                        "--checkpoint",
                        checkpoint.name,
                        "--evaluation_output",
                        str(evaluation_dir),
                        "--video",
                        "--headless",
                    ],
                    False,
                )
                entry["evaluation_directory"] = str(evaluation_dir.resolve())
            entry["stage"] = "complete"
            _save_state(state_path, state)
        except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
            entry.update(stage="failed", error=str(exc))
            _save_state(state_path, state)
            print(f"[ORCHESTRATOR] Failed at motion {motion}: {exc}", file=sys.stderr, flush=True)
            raise SystemExit(1) from exc

    print(f"[ORCHESTRATOR] Requested motions finished. State: {state_path}", flush=True)


if __name__ == "__main__":
    main()
