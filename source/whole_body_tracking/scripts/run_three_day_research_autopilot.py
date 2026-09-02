#!/usr/bin/env python3
"""Resume-safe queue: teacher MuJoCo/deep-Taichi experiment, then Z ablation."""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
STATUS = ROOT / "results/three_day_autopilot/status.json"
REEVAL_STATUS = ROOT / "results/baseline/reevaluation_v2/status.json"
LOG = ROOT / "results/three_day_autopilot/autopilot.log"
MUJOCO_PYTHON = Path("/home/unitree/miniconda3/envs/gmr/bin/python")
PROTO_STATE = ROOT / "results/retarget_comparison/orchestrator_state.json"
DEEP_STATE = ROOT / "results/teacher_deep_taichi/orchestrator_state.json"


def save(stage: str, **extra) -> None:
    STATUS.parent.mkdir(parents=True, exist_ok=True)
    STATUS.write_text(json.dumps({"stage": stage, "updated_at": time.time(), **extra}, indent=2) + "\n")


def run(command: list[str]) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    with LOG.open("a") as stream:
        stream.write("\n[RUN] " + " ".join(command) + "\n")
        stream.flush()
        subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)


def reference_height_evidence() -> dict:
    evidence = {}
    for motion in ("happy", "taichi1"):
        gmr_path = ROOT / f"inputs/{motion}/{motion}_gmr_50fps.npz"
        proto_path = ROOT / (
            f"retarget_comparison/protomotions/{motion}/04_wbt_npz/"
            f"{motion}_smpl_50fps_pyroki_proto.npz"
        )
        with np.load(gmr_path) as data:
            gmr_z = np.asarray(data["body_pos_w"])[:, 0, 2]
        with np.load(proto_path) as data:
            proto_z = np.asarray(data["body_pos_w"])[:, 0, 2]
        n = min(len(gmr_z), len(proto_z))
        delta = proto_z[:n] - gmr_z[:n]
        evidence[motion] = {
            "frames": n,
            "pelvis_z_difference_mean_m": float(delta.mean()),
            "pelvis_z_difference_median_m": float(np.median(delta)),
            "pelvis_z_difference_std_m": float(delta.std()),
        }
    return evidence


def mujoco_test(label: str, motion_file: Path, policy_file: Path) -> None:
    run([
        str(MUJOCO_PYTHON), "scripts/run_mujoco_fall_test.py",
        "--motion_file", str(motion_file), "--policy_file", str(policy_file),
        "--output", f"results/mujoco_fall_tests/{label}.json",
    ])


def exported_policy(state_file: Path, motion: str) -> Path:
    state = json.loads(state_file.read_text())
    run_dir = Path(state["motions"][motion]["run_directory"])
    return run_dir / "exported/policy.onnx"


def write_teacher_comparison() -> None:
    original = json.loads((ROOT / "results/retarget_comparison/reevaluation_v2/proto_taichi1/evaluation/summary.json").read_text())
    deep = json.loads((ROOT / "results/teacher_deep_taichi/proto_taichi1_deep/evaluation/summary.json").read_text())
    result = {
        "controlled_change": {
            "original_network": [512, 256, 128],
            "deep_network": [512, 1024, 256, 128],
            "unchanged_task": "Tracking-Flat-G1-v0 environment via Tracking-Flat-G1-Deep-v0 registry alias",
        },
        "original_rmse": {key: value["rmse"] for key, value in original["metrics"].items()},
        "deep_rmse": {key: value["rmse"] for key, value in deep["metrics"].items()},
        "mujoco_original": json.loads((ROOT / "results/mujoco_fall_tests/proto_taichi1_original.json").read_text()),
        "mujoco_deep": json.loads((ROOT / "results/mujoco_fall_tests/proto_taichi1_deep.json").read_text()),
    }
    path = ROOT / "results/teacher_deep_taichi/COMPARISON.json"
    path.write_text(json.dumps(result, indent=2) + "\n")


def main() -> None:
    save("waiting_for_unified_reevaluation")
    while True:
        if REEVAL_STATUS.is_file():
            state = json.loads(REEVAL_STATUS.read_text())
            if state.get("stage") == "complete":
                break
            if state.get("stage") == "failed":
                save("blocked", reason="unified_reevaluation_failed", reevaluation=state)
                raise SystemExit(1)
        time.sleep(60)

    save("summarizing_unified_reevaluation")
    run([
        sys.executable, "scripts/summarize_detailed_evaluations.py",
        "--result_root", "results/baseline/reevaluation_v2",
        "--output", "results/baseline/reevaluation_v2/DETAILED_METRICS.md",
    ])
    save("mujoco_testing_original_networks")
    proto = json.loads(PROTO_STATE.read_text())
    for label, state_name, source_name in (
        ("proto_happy_original", "proto_happy", "happy"),
        ("proto_taichi1_original", "proto_taichi1", "taichi1"),
    ):
        motion_file = ROOT / (
            f"retarget_comparison/protomotions/{source_name}/04_wbt_npz/"
            f"{source_name}_smpl_50fps_pyroki_proto.npz"
        )
        policy_file = Path(proto["motions"][state_name]["run_directory"]) / "exported/policy.onnx"
        mujoco_test(label, motion_file, policy_file)

    save("training_teacher_deep_taichi")
    run([
        sys.executable, "scripts/run_teacher_deep_taichi_training.py",
        "--task", "Tracking-Flat-G1-Deep-v0", "--num_envs", "4096",
        "--min_iterations", "30000", "--max_iterations", "100000",
        "--window", "1000", "--poll_seconds", "60",
    ])
    save("mujoco_testing_deep_taichi")
    deep_motion = ROOT / "retarget_comparison/protomotions/taichi1/04_wbt_npz/taichi1_smpl_50fps_pyroki_proto.npz"
    mujoco_test("proto_taichi1_deep", deep_motion, exported_policy(DEEP_STATE, "proto_taichi1_deep"))
    write_teacher_comparison()

    evidence = reference_height_evidence()
    evidence_path = ROOT / "results/three_day_autopilot/reference_height_evidence.json"
    evidence_path.write_text(json.dumps(evidence, indent=2) + "\n")
    consistent = all(evidence[name]["pelvis_z_difference_mean_m"] > 0.04 for name in evidence)
    if not consistent:
        save("complete_without_ablation", reason="height_bias_not_consistent", evidence=evidence)
        return

    save("creating_height_aligned_references", evidence=evidence)
    run([sys.executable, "scripts/create_height_aligned_ablation.py"])
    run([
        sys.executable, "scripts/validate_motion_inputs.py",
        "retarget_comparison/height_aligned/happy/happy_pyroki_height_aligned_50fps.npz",
        "retarget_comparison/height_aligned/taichi1/taichi1_pyroki_height_aligned_50fps.npz",
        "--output", "results/height_aligned_ablation/input_validation.json",
    ])
    save("training_height_aligned_ablation", evidence=evidence)
    run([
        sys.executable, "scripts/run_height_aligned_ablation_training.py",
        "--num_envs", "4096", "--min_iterations", "30000",
        "--max_iterations", "100000", "--window", "1000", "--poll_seconds", "60",
    ])
    save("complete", evidence=evidence)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        save("failed", returncode=error.returncode, command=error.cmd)
        raise SystemExit(error.returncode) from error
