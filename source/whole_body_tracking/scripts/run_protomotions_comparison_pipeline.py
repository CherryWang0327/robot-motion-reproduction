#!/usr/bin/env python3
"""Build ProtoMotions/PyRoki references for the happy/taichi1 GMR comparison."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = ROOT / "retarget_comparison" / "protomotions"
STATE_FILE = ROOT / "results" / "baseline" / "orchestrator_state.json"
PROTO_ROOT = Path("/home/unitree/projects/ProtoMotions")
PROTO_PYTHON = Path("/home/unitree/miniconda3/envs/protomotions/bin/python")
PYROKI_PYTHON = Path("/home/unitree/miniconda3/envs/pyroki/bin/python")
WBT_PYTHON = Path("/home/unitree/miniconda3/envs/whole_body_tracking/bin/python")
PT_TO_NPZ = ROOT / "scripts" / "pt_to_npz_local.py"
MOTIONS = ("happy", "taichi1")


def ensure_baseline_complete() -> None:
    if not STATE_FILE.is_file():
        raise RuntimeError(f"baseline state is missing: {STATE_FILE}")
    state = json.loads(STATE_FILE.read_text())
    incomplete = [
        name
        for name, entry in state.get("motions", {}).items()
        if entry.get("stage") != "complete"
    ]
    if len(state.get("motions", {})) != 10 or incomplete:
        raise RuntimeError(
            "GPU pipeline is locked until the ten-motion baseline is 10/10 complete; "
            f"incomplete={incomplete or 'state does not contain ten motions'}"
        )


def run(command: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    rendered = " ".join(command)
    print(f"[RUN cwd={cwd}] {rendered}", flush=True)
    if dry_run:
        return
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as log:
        log.write(f"\n[RUN cwd={cwd}] {rendered}\n")
        log.flush()
        subprocess.run(command, cwd=cwd, stdout=log, stderr=subprocess.STDOUT, check=True)


def convert(motion: str, dry_run: bool) -> None:
    source = WORK_ROOT / motion / "source" / f"{motion}_smpl_50fps.csv"
    output = WORK_ROOT / motion / "01_smpl_motion"
    run(
        [
            str(PROTO_PYTHON),
            "data/scripts/all_convert_amass_to_proto.py",
            "--input-root-dir", str(source),
            "--output-root-dir", str(output),
            "--humanoid-type", "smpl",
            "--output-fps", "50",
            "--csv-fps", "50",
            "--force-remake",
        ],
        PROTO_ROOT,
        WORK_ROOT / motion / "logs" / "01_convert.log",
        dry_run,
    )


def pack(motion: str, dry_run: bool) -> None:
    run(
        [
            str(PROTO_PYTHON),
            "protomotions/components/batch_pack_motion_to_pt.py",
            "--input-dir", str(WORK_ROOT / motion / "01_smpl_motion"),
            "--output-dir", str(WORK_ROOT / motion / "02_smpl_pt"),
            "--device", "cuda",
            "--recursive",
            "--force-remake",
        ],
        PROTO_ROOT,
        WORK_ROOT / motion / "logs" / "02_pack.log",
        dry_run,
    )


def retarget(motion: str, dry_run: bool, chunk_len: int) -> None:
    source_script = PROTO_ROOT / "scripts" / "retarget_amass_to_robot.sh"
    script_text = source_script.read_text()
    old = "--chunk-len 1000"
    if old not in script_text:
        raise RuntimeError(f"expected chunk setting not found in {source_script}")
    script_text = script_text.replace(old, f"--chunk-len {chunk_len}")
    temporary_path: Path | None = None
    if not dry_run:
        with tempfile.NamedTemporaryFile(
            mode="w", prefix="wbt_retarget_", suffix=".sh", dir="/tmp", delete=False
        ) as temporary:
            temporary.write(script_text)
            temporary_path = Path(temporary.name)
    script_path = temporary_path or source_script
    run(
        [
            "bash", str(script_path),
            str(PROTO_PYTHON), str(PYROKI_PYTHON),
            str(WORK_ROOT / motion / "02_smpl_pt"),
            str(WORK_ROOT / motion / "03_g1_pyroki"),
            "g1", "1", "0",
        ],
        PROTO_ROOT,
        WORK_ROOT / motion / "logs" / "03_retarget.log",
        dry_run,
    )
    if temporary_path is not None:
        temporary_path.unlink(missing_ok=True)
    if not dry_run:
        outputs = list((WORK_ROOT / motion / "03_g1_pyroki").rglob("*_pyroki.pt"))
        if not outputs:
            raise RuntimeError(
                f"PyRoki produced no final PT for {motion}; inspect "
                f"{WORK_ROOT / motion / 'logs' / '03_retarget.log'}"
            )


def npz(motion: str, dry_run: bool) -> None:
    run(
        [
            str(WBT_PYTHON), str(PT_TO_NPZ),
            "--input_path", str(WORK_ROOT / motion / "03_g1_pyroki"),
            "--output_path", str(WORK_ROOT / motion / "04_wbt_npz"),
            "--input_fps", "50",
            "--output_fps", "50",
            "--device", "cpu",
            "--headless",
            "--skip_existing",
        ],
        ROOT,
        WORK_ROOT / motion / "logs" / "04_npz.log",
        dry_run,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--phase", choices=("convert", "pack", "retarget", "npz", "all"), default="all"
    )
    parser.add_argument("--motions", nargs="+", choices=MOTIONS, default=list(MOTIONS))
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--pyroki-chunk-len",
        type=int,
        default=1000,
        help="Frames solved per JAX graph; keep the tutorial default for controlled comparison.",
    )
    parser.add_argument(
        "--ignore-baseline-lock",
        action="store_true",
        help="Expert-only override; may contend with an active baseline GPU job.",
    )
    args = parser.parse_args()
    if args.pyroki_chunk_len < 2:
        parser.error("--pyroki-chunk-len must be at least 2")
    if not args.ignore_baseline_lock:
        ensure_baseline_complete()

    phases = ("convert", "pack", "retarget", "npz") if args.phase == "all" else (args.phase,)
    functions = {"convert": convert, "pack": pack, "npz": npz}
    for motion in args.motions:
        source = WORK_ROOT / motion / "source" / f"{motion}_smpl_50fps.csv"
        if not source.is_file():
            parser.error(f"missing staged CSV: {source}")
        for phase in phases:
            if phase == "retarget":
                retarget(motion, args.dry_run, args.pyroki_chunk_len)
            else:
                functions[phase](motion, args.dry_run)


if __name__ == "__main__":
    try:
        main()
    except (OSError, RuntimeError, subprocess.CalledProcessError) as exc:
        print(f"[ERROR] {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
