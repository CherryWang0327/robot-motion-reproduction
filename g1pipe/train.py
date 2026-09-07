from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .validate import validate_npz


WBT_ROOT = Path("/home/unitree/projects/whole_body_tracking")
WBT_PYTHON = Path("/home/unitree/miniconda3/envs/whole_body_tracking/bin/python")
TRAIN_SCRIPT = WBT_ROOT / "scripts" / "rsl_rl" / "train.py"


def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def plan_training(
    motion: Path,
    run_dir: Path,
    run_name: str,
    max_iterations: int,
    num_envs: int,
    execute: bool,
) -> None:
    """Validate, record and optionally launch a normal WBT local-motion training run.

    Actual WBT creates its native checkpoints and TensorBoard files below its own
    ``logs/`` directory. This wrapper never edits WBT source/config files and
    keeps the command, launcher output and state record under ``run_dir``.
    """
    motion = motion.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    report = validate_npz(motion)
    training_dir = run_dir / "05_training"
    _write_json(training_dir / "input_validation.json", report.as_dict())
    if report.status != "PASS":
        raise RuntimeError(f"Training gate is {report.status}; see {training_dir / 'input_validation.json'}")
    if max_iterations < 1 or num_envs < 1:
        raise ValueError("max_iterations and num_envs must both be positive")
    command = [
        str(WBT_PYTHON), str(TRAIN_SCRIPT),
        "--task", "Tracking-Flat-G1-v0",
        "--motion_file", str(motion),
        "--num_envs", str(num_envs),
        "--max_iterations", str(max_iterations),
        "--seed", "0", "--headless",
        "--logger", "tensorboard", "--run_name", run_name,
    ]
    stage = {
        "stage": "wbt_train",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion": str(motion),
        "input_validation": str(training_dir / "input_validation.json"),
        "command": command,
        "working_directory": str(WBT_ROOT),
        "native_wbt_log_location": str(WBT_ROOT / "logs" / "rsl_rl" / "g1_flat"),
        "run_name": run_name,
        "max_iterations": max_iterations,
        "num_envs": num_envs,
        "robot_access": "none; simulation only",
    }
    _write_json(training_dir / "stage.json", stage)
    if not execute:
        print("PLAN ONLY: training was not started. Add --execute to start the simulation job.")
        print(" ".join(command))
        return
    log_path = training_dir / "launcher.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=WBT_ROOT, stdout=log, stderr=subprocess.STDOUT)
    stage["returncode"] = completed.returncode
    stage["status"] = "COMPLETE" if completed.returncode == 0 else "FAIL"
    _write_json(training_dir / "stage.json", stage)
    if completed.returncode != 0:
        raise RuntimeError(f"WBT training failed; inspect {log_path}")
