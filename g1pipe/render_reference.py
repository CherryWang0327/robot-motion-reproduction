"""Headless, reproducible G1 reference-motion preview renderer.

This intentionally renders the validated WBT reference state, not a learned
policy rollout.  It is therefore evidence for the upstream GMR/PyRoki motion.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FFMpegWriter


# G1 rigid-body order used by whole_body_tracking's exported NPZ.  The exact
# mesh is not needed to verify the retargeted reference: these links show the
# body topology and its world-space motion without any viewer or desktop.
EDGES = (
    (0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6),
    (0, 10), (10, 11), (11, 12), (12, 13), (13, 14), (14, 15),
    (0, 17), (17, 18), (18, 19), (19, 20),
    (19, 23), (23, 24), (24, 25), (25, 26), (26, 27), (27, 28), (28, 29),
    (19, 31), (31, 32), (32, 33), (33, 34), (34, 35), (35, 36), (36, 37),
)


def render(motion: Path, output: Path, title: str, output_fps: int = 30, max_seconds: float | None = None) -> None:
    data = np.load(motion)
    body = np.asarray(data["body_pos_w"], dtype=np.float32)
    input_fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    if body.ndim != 3 or body.shape[-1] != 3:
        raise ValueError(f"body_pos_w must be (T, B, 3), got {body.shape}")
    if body.shape[1] < 30:
        raise ValueError(f"expected at least 30 G1 bodies, got {body.shape[1]}")
    stride = max(1, round(input_fps / output_fps))
    frames = body[::stride]
    fps = input_fps / stride
    if max_seconds is not None:
        frames = frames[: max(1, round(max_seconds * fps))]
    output.parent.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(6.4, 5.2), facecolor="#08111d")
    ax = fig.add_subplot(111, projection="3d")
    ax.set_facecolor("#08111d")
    ax.tick_params(colors="#91a8b8")
    ax.set_xlabel("x (m)", color="#91a8b8")
    ax.set_ylabel("y (m)", color="#91a8b8")
    ax.set_zlabel("z (m)", color="#91a8b8")
    writer = FFMpegWriter(fps=fps, codec="libx264", extra_args=["-pix_fmt", "yuv420p", "-movflags", "+faststart"])
    with writer.saving(fig, str(output), dpi=110):
        for index, pose in enumerate(frames):
            ax.cla()
            center = pose[0]
            ax.set_xlim(center[0] - 1.4, center[0] + 1.4)
            ax.set_ylim(center[1] - 1.4, center[1] + 1.4)
            ax.set_zlim(0.0, 2.2)
            ax.view_init(elev=16, azim=-68)
            ax.set_title(f"{title}  |  {index / fps:.2f}s", color="white", pad=10)
            for start, end in EDGES:
                if start < len(pose) and end < len(pose):
                    ax.plot(*zip(pose[start], pose[end]), color="#4fd1ff", linewidth=2.5)
            ax.scatter(pose[:, 0], pose[:, 1], pose[:, 2], c="#eef8fc", s=11)
            writer.grab_frame()
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--motion", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="G1 reference replay")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max-seconds", type=float, default=None, help="Smoke-test cap; omit for the complete motion.")
    parser.add_argument("--stage-file", type=Path, default=None, help="Optional stage.json written after a successful render.")
    parser.add_argument("--stage-name", default="g1_reference_preview")
    args = parser.parse_args()
    render(args.motion, args.output, args.title, args.fps, args.max_seconds)
    if args.stage_file:
        args.stage_file.parent.mkdir(parents=True, exist_ok=True)
        args.stage_file.write_text(json.dumps({
            "stage": args.stage_name,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "motion": str(args.motion), "video": str(args.output),
            "kind": "kinematic_reference_replay", "training_started": False,
            "robot_access": "none", "status": "COMPLETE",
        }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
