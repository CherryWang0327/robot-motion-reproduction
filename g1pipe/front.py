from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path


GVHMR_ROOT = Path("/home/unitree/projects/GVHMR")
GMR_ROOT = Path("/home/unitree/projects/GMR")
PROTO_ROOT = Path("/home/unitree/projects/ProtoMotions")
GVHMR_PYTHON = Path("/home/unitree/miniconda3/envs/gvhmr5070/bin/python")
GMR_PYTHON = Path("/home/unitree/miniconda3/envs/gmr/bin/python")
PROTO_PYTHON = Path("/home/unitree/miniconda3/envs/protomotions/bin/python")
PYROKI_PYTHON = Path("/home/unitree/miniconda3/envs/pyroki/bin/python")


def _record(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _run(stage_dir: Path, command: list[str], cwd: Path, dry_run: bool, extra_env: dict[str, str] | None = None) -> None:
    record = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "command": command,
        "cwd": str(cwd),
        "dry_run": dry_run,
        "robot_access": "none",
    }
    if dry_run:
        record["status"] = "DRY_RUN"
        _record(stage_dir / "stage.json", record)
        return
    record["status"] = "RUNNING"
    _record(stage_dir / "stage.json", record)
    log_path = stage_dir / "stage.log"
    with log_path.open("w", encoding="utf-8") as log:
        completed = subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, env={**os.environ, **(extra_env or {})})
    record["returncode"] = completed.returncode
    record["status"] = "COMPLETE" if completed.returncode == 0 else "FAIL"
    _record(stage_dir / "stage.json", record)
    if completed.returncode:
        raise RuntimeError(f"Stage failed; inspect {log_path}")


def _single(root: Path, pattern: str, description: str) -> Path:
    files = list(root.rglob(pattern))
    if len(files) != 1:
        raise RuntimeError(f"Expected one {description} under {root}, found {len(files)}")
    return files[0]


def build_from_video(video: Path, route: str, run_dir: Path, dry_run: bool = False) -> Path:
    """Build a GMR PKL or final ProtoMotions/PyRoki PT from one input video."""
    video = video.expanduser().resolve()
    run_dir = run_dir.expanduser().resolve()
    if not video.is_file():
        raise FileNotFoundError(video)
    if route not in {"gmr", "protomotions"}:
        raise ValueError(route)

    gvhmr_dir = run_dir / "01_gvhmr"
    _run(gvhmr_dir, [str(GVHMR_PYTHON), "tools/demo/demo.py", "--video", str(video), "--output_root", str(gvhmr_dir)], GVHMR_ROOT, dry_run, {"TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD": "1"})
    hmr = gvhmr_dir / "hmr4d_results.pt" if dry_run else _single(gvhmr_dir, "hmr4d_results.pt", "GVHMR result")
    smpl_dir = run_dir / "02_smpl"
    smpl_csv = smpl_dir / f"{video.stem}_smpl_global_30fps.csv"
    _run(smpl_dir, [str(GVHMR_PYTHON), "scripts/gvhmr_to_smpl_csv.py", "--input", str(hmr), "--output", str(smpl_csv), "--fps", "30"], GVHMR_ROOT, dry_run)
    if route == "gmr":
        output = run_dir / "03_retarget" / f"{video.stem}_g1.pkl"
        # The Unitree desktop runs Xorg on :1, while web-worker processes do
        # not inherit DISPLAY.  Reuse that local display when available so
        # GMR can create its native --record_video MP4 for the results page.
        gmr_display = os.environ.get("DISPLAY")
        # :1002 is the Unitree user's active NoMachine desktop; :1 belongs
        # to the GDM login screen and rejects the unitree session cookie.
        if not gmr_display and Path("/tmp/.X11-unix/X1002").exists():
            gmr_display = ":1002"
        gmr_env: dict[str, str] = {}
        if gmr_display:
            gmr_env["DISPLAY"] = gmr_display
            desktop_auth = Path("/home/unitree/.Xauthority")
            if desktop_auth.is_file():
                gmr_env["XAUTHORITY"] = str(desktop_auth)
        can_record_gmr = bool(gmr_display)
        gmr_command = [str(GMR_PYTHON), "scripts/smpl_csv_to_robot.py", "--input", str(smpl_csv), "--robot", "unitree_g1", "--output", str(output), "--fps", "30", "--target_fps", "30"]
        if can_record_gmr:
            gmr_command.append("--record_video")
        else:
            # GMR's viewer uses GLFW.  A headless machine must not fail the
            # data-producing retarget stage just because no desktop exists.
            gmr_command.append("--no_viewer")
        _run(output.parent, gmr_command, GMR_ROOT, dry_run, gmr_env)
        if not dry_run and not output.is_file():
            raise RuntimeError(f"GMR did not create {output}")
        if not dry_run:
            videos = sorted((GMR_ROOT / "videos").glob(f"unitree_g1_{smpl_csv.stem}*.mp4"), key=lambda path: path.stat().st_mtime)
            if videos:
                # Keep the native GMR viewer recording alongside the later
                # headless WBT-reference replay.  They answer different
                # questions and must not overwrite one another.
                shutil.copy2(videos[-1], output.parent / "gmr_native_reference.mp4")
        return output

    # Both routes consume the exported SMPL CSV. Preserve its GVHMR Y-up
    # values; ProtoMotions owns the single conversion via --source-y-up.
    motion_dir = run_dir / "03_proto_motion"
    amass_dir = motion_dir / "amass"
    amass_dir.mkdir(parents=True, exist_ok=True)
    amass_npz = amass_dir / f"{video.stem}.npz"
    _run(motion_dir, [str(PROTO_PYTHON), "-m", "g1pipe.smpl_csv", "--input", str(smpl_csv), "--output", str(amass_npz)], PROTO_ROOT, dry_run, {"PYTHONPATH": str(Path(__file__).resolve().parents[1])})
    _run(motion_dir, [str(PROTO_PYTHON), "data/scripts/convert_amass_to_proto.py", str(amass_dir), "--humanoid-type", "smpl", "--output-fps", "50", "--source-y-up", "--force-remake"], PROTO_ROOT, dry_run, {"PYTHONPATH": str(PROTO_ROOT)})

    packed_dir = run_dir / "04_proto_pt"
    motion_yaml = motion_dir / f"{video.stem}.yaml"
    motion_yaml.write_text(f"motions:\n- file: amass/{video.stem}.motion\n  weight: 1.0\n", encoding="utf-8")
    motionlib = packed_dir / f"{video.stem}_motionlib.pt"
    _run(packed_dir, [str(PROTO_PYTHON), "protomotions/components/motion_lib.py", "--motion-path", str(motion_yaml), "--output-file", str(motionlib), "--device", "cpu"], PROTO_ROOT, dry_run, {"PYTHONPATH": str(PROTO_ROOT)})

    retarget_dir = run_dir / "05_retarget"
    retarget_dir.mkdir(parents=True, exist_ok=True)
    script = PROTO_ROOT / "scripts" / "retarget_amass_to_robot.sh"
    _run(retarget_dir, ["bash", str(script), str(PROTO_PYTHON), str(PYROKI_PYTHON), str(packed_dir), str(retarget_dir), "g1", "1", "0"], PROTO_ROOT, dry_run)
    if dry_run:
        return retarget_dir / f"{video.stem}_motionlib_pyroki.pt"
    return _single(retarget_dir, "*_pyroki.pt", "PyRoki G1 PT")
