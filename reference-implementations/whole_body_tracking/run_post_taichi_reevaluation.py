#!/usr/bin/env python3
"""Wait for strict Proto training, then re-evaluate ten GMR and two Proto policies."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PROTO_STATE = ROOT / "results/retarget_comparison/orchestrator_state.json"
LOG = ROOT / "results/baseline/reevaluation_v2/orchestrator.log"
STATUS = ROOT / "results/baseline/reevaluation_v2/status.json"
PROTO_REEVAL_ROOT = ROOT / "results/retarget_comparison/reevaluation_v2"


def save_status(**values) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps(values, indent=2) + "\n")


def proto_finished() -> bool:
    if not PROTO_STATE.is_file():
        return False
    state = json.loads(PROTO_STATE.read_text())
    motions = state.get("motions", {})
    return all(motions.get(name, {}).get("stage") == "complete" for name in ("proto_happy", "proto_taichi1"))


def run_logged(command: list[str], stream) -> None:
    result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    if result.returncode:
        raise subprocess.CalledProcessError(result.returncode, command)


def reevaluate_proto(stream) -> None:
    state = json.loads(PROTO_STATE.read_text())
    for motion in ("proto_happy", "proto_taichi1"):
        entry = state["motions"][motion]
        source_name = motion.removeprefix("proto_")
        motion_file = ROOT / (
            f"retarget_comparison/protomotions/{source_name}/04_wbt_npz/"
            f"{source_name}_smpl_50fps_pyroki_proto.npz"
        )
        evaluation_dir = PROTO_REEVAL_ROOT / motion / "evaluation"
        command = [
            sys.executable,
            "scripts/rsl_rl/play.py",
            "--task", "Tracking-Flat-G1-v0",
            "--motion_file", str(motion_file),
            "--load_run", Path(entry["run_directory"]).name,
            "--checkpoint", Path(entry["final_checkpoint"]).name,
            "--evaluation_output", str(evaluation_dir),
            "--headless",
            "--video",
        ]
        run_logged(command, stream)


def main() -> None:
    while not proto_finished():
        save_status(stage="waiting_for_proto_taichi1", checked_at=time.time())
        time.sleep(60)
    command = [
        sys.executable,
        "scripts/rerun_detailed_evaluations.py",
        "--output_root",
        "results/baseline/reevaluation_v2",
        "--video",
        "--force",
    ]
    save_status(stage="reevaluating_gmr", command=command, started_at=time.time())
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        try:
            run_logged(command, stream)
            save_status(stage="reevaluating_proto", started_at=time.time())
            reevaluate_proto(stream)
        except subprocess.CalledProcessError as error:
            save_status(stage="failed", returncode=error.returncode, command=error.cmd, finished_at=time.time())
            raise SystemExit(error.returncode)
    save_status(
        stage="complete",
        returncode=0,
        gmr_output=str(ROOT / "results/baseline/reevaluation_v2"),
        proto_output=str(PROTO_REEVAL_ROOT),
        finished_at=time.time(),
    )


if __name__ == "__main__":
    main()

