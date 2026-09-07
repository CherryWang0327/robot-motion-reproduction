from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

from .gmr import convert_gmr_pkl_to_pt
from .proto_preview import render_native_protomotions
from .validate import validate_npz, write_report


WBT_ROOT = Path("/home/unitree/projects/whole_body_tracking")
WBT_PYTHON = Path("/home/unitree/miniconda3/envs/whole_body_tracking/bin/python")
PT_TO_NPZ = WBT_ROOT / "scripts" / "pt_to_npz_local.py"
WBT_PLAY = WBT_ROOT / "scripts" / "rsl_rl" / "play.py"
REFERENCE_RENDERER = Path(__file__).with_name("render_reference.py")
PREVIEW_RUN = "2026-08-07_15-45-48_baseline_happy_100k"
PREVIEW_CHECKPOINT = "model_30000.pt"


def _write_stage(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _render_reference_preview(route: str, motion: Path, run_dir: Path) -> None:
    """Render the exact validated reference trajectory without a learned policy."""
    stage_name = "05_gmr_reference_preview" if route == "gmr" else "05_protomotions_reference_preview"
    stage_dir = run_dir / stage_name
    video = stage_dir / "g1_reference_motion.mp4"
    label = "GMR G1 reference replay" if route == "gmr" else "PyRoki G1 reference replay"
    command = [str(WBT_PYTHON), str(REFERENCE_RENDERER), "--motion", str(motion), "--output", str(video), "--title", label]
    record = {"stage": stage_name, "created_at": datetime.now(timezone.utc).isoformat(), "motion": str(motion), "command": command, "kind": "kinematic_reference_replay", "training_started": False, "robot_access": "none", "status": "RUNNING"}
    _write_stage(stage_dir / "stage.json", record)
    completed = subprocess.run(command, cwd=WBT_ROOT, text=True, capture_output=True)
    (stage_dir / "preview.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    record["status"] = "COMPLETE" if completed.returncode == 0 and video.is_file() else "WARN"
    record["video"] = str(video) if video.is_file() else None
    record["returncode"] = completed.returncode
    _write_stage(stage_dir / "stage.json", record)


def _render_g1_preview(motion: Path, run_dir: Path) -> None:
    """Record a G1 tracking preview without starting a new training job."""
    stage_dir = run_dir / "06_wbt_policy_preview"
    stage_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(WBT_PYTHON), str(WBT_PLAY),
        "--task", "Tracking-Flat-G1-v0",
        "--motion_file", str(motion),
        "--evaluation_output", str(stage_dir),
        "--video", "--headless", "--num_envs", "1",
        "--load_run", PREVIEW_RUN, "--checkpoint", PREVIEW_CHECKPOINT,
    ]
    record = {
        "stage": "wbt_policy_preview",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "motion": str(motion),
        "command": command,
        "kind": "pretrained_policy_tracking_preview",
        "training_started": False,
        "robot_access": "none",
        "status": "RUNNING",
    }
    _write_stage(stage_dir / "stage.json", record)
    completed = subprocess.run(command, cwd=WBT_ROOT, text=True, capture_output=True)
    (stage_dir / "preview.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    videos = sorted((stage_dir / "video").glob("*.mp4"))
    if completed.returncode == 0 and videos:
        preview = stage_dir / "g1_reference_tracking_preview.mp4"
        videos[-1].replace(preview)
        record["video"] = str(preview)
        record["status"] = "COMPLETE"
    else:
        record["returncode"] = completed.returncode
        record["status"] = "WARN"
        record["message"] = "WBT motion passed validation, but the optional G1 preview was not recorded."
    _write_stage(stage_dir / "stage.json", record)


def convert_to_wbt(route: str, source: Path, run_dir: Path, dry_run: bool = False) -> Path:
    """Convert a GMR PKL or ProtoMotions PT into a validated local WBT NPZ."""
    source = source.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    stage_dir = run_dir / "03_wbt"
    stage_dir.mkdir(parents=True, exist_ok=True)
    if route == "gmr":
        intermediate = run_dir / "02_retarget" / f"{source.stem}_gmr.pt"
        if not dry_run:
            convert_gmr_pkl_to_pt(source, intermediate)
        quaternion_order = "xyzw"
    elif route == "protomotions":
        intermediate = source
        quaternion_order = "xyzw"  # verified project contract for grs
    else:
        raise ValueError(f"Unsupported route: {route}")
    output = stage_dir / f"{intermediate.stem}_proto.npz"
    command = [
        str(WBT_PYTHON), str(PT_TO_NPZ),
        "--input_path", str(intermediate),
        "--output_path", str(output),
        "--output_fps", "50",
        "--input_root_rot_order", quaternion_order,
        "--device", "cpu", "--headless",
    ]
    stage = {
        "stage": "convert_to_wbt",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "route": route,
        "source": str(source),
        "intermediate": str(intermediate),
        "output": str(output),
        "command": command,
        "dry_run": dry_run,
        "control_access": "none",
    }
    if dry_run:
        stage["status"] = "DRY_RUN"
        _write_stage(stage_dir / "stage.json", stage)
        return output
    completed = subprocess.run(command, cwd=WBT_ROOT, text=True, capture_output=True)
    (stage_dir / "conversion.log").write_text(completed.stdout + completed.stderr, encoding="utf-8")
    if completed.returncode != 0:
        stage["status"] = "FAIL"
        stage["returncode"] = completed.returncode
        _write_stage(stage_dir / "stage.json", stage)
        raise RuntimeError(f"WBT conversion failed; inspect {stage_dir / 'conversion.log'}")
    if not output.is_file():
        stage["status"] = "FAIL"
        _write_stage(stage_dir / "stage.json", stage)
        raise RuntimeError(f"WBT converter returned success but did not create {output}")
    report = validate_npz(output)
    write_report(report, run_dir / "04_validation" / "report.json")
    stage["status"] = report.status
    stage["validation_report"] = str(run_dir / "04_validation" / "report.json")
    _write_stage(stage_dir / "stage.json", stage)
    if report.status == "FAIL":
        raise RuntimeError("Conversion produced an invalid WBT NPZ; see validation report")
    _render_reference_preview(route, output, run_dir)
    if route == "protomotions":
        motionlibs = list((run_dir / "04_proto_pt").glob("*_motionlib.pt"))
        if len(motionlibs) == 1:
            render_native_protomotions(motionlibs[0], run_dir)
    _render_g1_preview(output, run_dir)
    return output
