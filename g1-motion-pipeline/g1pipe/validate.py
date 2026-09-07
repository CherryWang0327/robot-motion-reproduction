from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import EXPECTED_BODIES, EXPECTED_FPS, EXPECTED_JOINTS, MOTION_FIELD_ALIASES, WBT_REQUIRED_FIELDS


@dataclass
class Finding:
    level: str
    check: str
    message: str


@dataclass
class Report:
    motion: str
    findings: list[Finding] = field(default_factory=list)
    schema: dict[str, list[int]] = field(default_factory=dict)

    def add(self, level: str, check: str, message: str) -> None:
        self.findings.append(Finding(level, check, message))

    @property
    def status(self) -> str:
        levels = {item.level for item in self.findings}
        return "FAIL" if "FAIL" in levels else "WARN" if "WARN" in levels else "PASS"

    def as_dict(self) -> dict[str, Any]:
        return {"motion": self.motion, "status": self.status, "schema": self.schema, "findings": [finding.__dict__ for finding in self.findings]}


def _first_field(files: list[str], aliases: tuple[str, ...]) -> str | None:
    return next((key for key in aliases if key in files), None)


def validate_npz(path: Path, expected_fps: float = EXPECTED_FPS) -> Report:
    report = Report(str(path))
    if not path.is_file():
        report.add("FAIL", "file", "Input file does not exist")
        return report
    try:
        archive = np.load(path, allow_pickle=False)
    except Exception as exc:
        report.add("FAIL", "npz_read", f"Cannot read NPZ: {exc}")
        return report
    try:
        files = list(archive.files)
        report.schema = {key: list(np.asarray(archive[key]).shape) for key in files}
        report.add("PASS", "npz_read", f"Read {len(files)} fields")
        missing_wbt = [key for key in WBT_REQUIRED_FIELDS if key not in files]
        if missing_wbt:
            report.add("FAIL", "wbt_schema", "Missing WBT fields: " + ", ".join(missing_wbt))
        else:
            frame_count = int(np.asarray(archive["joint_pos"]).shape[0])
            mismatched = [key for key in WBT_REQUIRED_FIELDS[1:] if np.asarray(archive[key]).shape[0] != frame_count]
            if mismatched:
                report.add("FAIL", "frame_count", "Frame-count mismatch: " + ", ".join(mismatched))
            else:
                report.add("PASS", "frame_count", f"All WBT fields have {frame_count} frames")
        for key in files:
            value = np.asarray(archive[key])
            if np.issubdtype(value.dtype, np.number) and not np.isfinite(value).all():
                report.add("FAIL", "finite_values", f"{key} contains NaN or Inf")
        joints_key = _first_field(files, MOTION_FIELD_ALIASES["joint_positions"])
        if joints_key is None:
            report.add("FAIL", "joint_schema", "No recognised joint-position field")
        else:
            joints = np.asarray(archive[joints_key])
            if joints.ndim < 2:
                report.add("FAIL", "joint_schema", f"{joints_key} must have at least 2 dimensions")
            elif joints.shape[-1] != EXPECTED_JOINTS:
                report.add("FAIL", "joint_count", f"{joints_key} has {joints.shape[-1]} joints; expected {EXPECTED_JOINTS}")
            else:
                report.add("PASS", "joint_count", f"{joints_key} has {EXPECTED_JOINTS} joints")
        body_key = _first_field(files, MOTION_FIELD_ALIASES["root_positions"])
        if body_key == "body_pos_w":
            bodies = np.asarray(archive[body_key])
            if bodies.ndim != 3 or bodies.shape[1:] != (EXPECTED_BODIES, 3):
                report.add("FAIL", "body_schema", f"{body_key} shape {tuple(bodies.shape)}; expected (T, {EXPECTED_BODIES}, 3)")
            else:
                report.add("PASS", "body_schema", f"{body_key} has {EXPECTED_BODIES} bodies")
        rotation_key = _first_field(files, MOTION_FIELD_ALIASES["root_rotations"])
        if rotation_key is not None:
            rotations = np.asarray(archive[rotation_key])
            if rotations.shape[-1] != 4:
                report.add("FAIL", "quaternion_schema", f"{rotation_key} last dimension is not 4")
            else:
                norms = np.linalg.norm(rotations.reshape(-1, 4), axis=1)
                if np.any(np.abs(norms - 1.0) > 0.02):
                    report.add("WARN", "quaternion_norm", "Some root quaternions are not near unit length")
                else:
                    report.add("PASS", "quaternion_norm", "Root quaternions are near unit length")
        else:
            report.add("WARN", "quaternion_schema", "No recognised root-rotation field")
        fps_key = next((key for key in ("fps", "motion_fps", "frequency") if key in files), None)
        if fps_key is None:
            report.add("WARN", "fps", f"No FPS field found; expected {expected_fps:g} FPS must be recorded externally")
        else:
            fps = float(np.asarray(archive[fps_key]).reshape(-1)[0])
            report.add("PASS" if abs(fps - expected_fps) < 1e-6 else "FAIL", "fps", f"{fps_key}={fps:g}; expected {expected_fps:g}")
    finally:
        archive.close()
    return report


def write_report(report: Report, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report.as_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
