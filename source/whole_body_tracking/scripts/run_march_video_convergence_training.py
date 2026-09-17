#!/usr/bin/env python3
"""Train march-video PyRoki then GMR with the shared convergence protocol."""

from pathlib import Path

import run_ten_motion_baseline as runner


ROOT = Path(__file__).resolve().parents[1]
runner.MOTIONS = {
    "march_video_pyroki": ROOT / "inputs/march_video/proto_50fps/march_video_motionlib_pyroki_proto.npz",
    "march_video_gmr": ROOT / "inputs/march_video/march_video_gmr_50fps.npz",
}
runner.RESULT_ROOT = ROOT / "results" / "march_video"


if __name__ == "__main__":
    runner.main()
