from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from . import db

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    job = next(item for item in db.all_jobs() if item["name"] == args.name)
    run = ROOT / "runs" / args.name
    db.update(args.name, "RUNNING")
    log = run / "00_job" / "launcher.log"
    command = [sys.executable, "-m", "g1pipe.cli", "run", "--video", str(ROOT / job["video"]), "--route", job["route"], "--run-dir", str(run)]
    with log.open("w", encoding="utf-8") as stream:
        result = subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT)
    db.update(args.name, "COMPLETE" if result.returncode == 0 else "FAIL", None if result.returncode == 0 else f"pipeline exited with {result.returncode}")


if __name__ == "__main__":
    main()
