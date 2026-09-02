#!/usr/bin/env python3
"""CPU-only extended metric pilot for the ten-motion GMR baseline."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BASELINE = ROOT / "results" / "baseline"
OUT = BASELINE / "extended_metrics"
STATE = BASELINE / "orchestrator_state.json"


def vector_metrics(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    magnitude = np.linalg.norm(values, axis=-1)
    return {
        "component_rms": float(np.sqrt(np.mean(values**2))),
        "magnitude_mean": float(magnitude.mean()),
        "magnitude_rms": float(np.sqrt(np.mean(magnitude**2))),
        "peak_component_abs": float(np.max(np.abs(values))),
        "peak_magnitude": float(magnitude.max()),
    }


def reference_metrics(path: Path) -> dict:
    with np.load(path) as data:
        fps = float(np.asarray(data["fps"]).reshape(-1)[0])
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
        body_lin = np.asarray(data["body_lin_vel_w"], dtype=np.float64)
        body_ang = np.asarray(data["body_ang_vel_w"], dtype=np.float64)
    dt = 1.0 / fps
    acceleration = np.gradient(joint_vel, dt, axis=0)
    jerk = np.gradient(acceleration, dt, axis=0)
    return {
        "fps": fps,
        "frames": int(joint_vel.shape[0]),
        "duration_seconds": float(joint_vel.shape[0] / fps),
        "joint_velocity": vector_metrics(joint_vel),
        "joint_acceleration": vector_metrics(acceleration),
        "joint_jerk": vector_metrics(jerk),
        "pelvis_linear_velocity": vector_metrics(body_lin[:, 0]),
        "pelvis_angular_velocity": vector_metrics(body_ang[:, 0]),
    }


def detailed_foot_metrics(path: Path, fps: float) -> dict | None:
    if not path.is_file():
        return None
    with path.open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    names = ("left_ankle_roll_link", "right_ankle_roll_link")
    speeds = []
    boundary_events = []
    total_distance = 0.0
    total_contact_frames = 0
    for row_index, row in enumerate(rows):
        for name in names:
            distance = float(row[f"foot_slip_step.{name}"])
            contact = int(row[f"contact.{name}"])
            total_distance += distance
            if contact:
                total_contact_frames += 1
                speed = distance * fps
                speeds.append(speed)
                if row_index == len(rows) - 1:
                    boundary_events.append({
                        "step": int(row["step"]),
                        "reference_frame": int(row["reference_frame"]),
                        "foot": name,
                        "speed_mps": speed,
                    })
    values = np.asarray(speeds, dtype=np.float64)
    core_values = np.asarray(
        [value for value, row_index in ((float(row[f"foot_slip_step.{name}"]) * fps, i)
         for i, row in enumerate(rows) for name in names if int(row[f"contact.{name}"]))
         if row_index != len(rows) - 1],
        dtype=np.float64,
    )
    return {
        "distance_total_m": total_distance,
        "contact_frames_total": total_contact_frames,
        "speed_mean_mps": float(values.mean()) if values.size else None,
        "speed_rms_mps": float(np.sqrt(np.mean(values**2))) if values.size else None,
        "speed_p95_mps": float(np.quantile(values, 0.95)) if values.size else None,
        "speed_max_mps": float(values.max()) if values.size else None,
        "boundary_events": boundary_events,
        "excluding_final_step": {
            "speed_mean_mps": float(core_values.mean()) if core_values.size else None,
            "speed_rms_mps": float(np.sqrt(np.mean(core_values**2))) if core_values.size else None,
            "speed_p95_mps": float(np.quantile(core_values, 0.95)) if core_values.size else None,
            "speed_max_mps": float(core_values.max()) if core_values.size else None,
        },
    }


def fmt(value, digits=4):
    return "N/A" if value is None else f"{value:.{digits}f}"


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    state = json.loads(STATE.read_text())
    records = []
    for summary_path in sorted(BASELINE.glob("*/evaluation/summary.json")):
        motion = summary_path.parents[1].name
        summary = json.loads(summary_path.read_text())
        reference_path = Path(summary["motion_file"])
        reference = reference_metrics(reference_path)
        planned = int(summary["planned_steps"])
        recorded = int(summary["recorded_steps"])
        completion_ratio = recorded / planned if planned else 0.0
        details = summary.get("detailed_metrics")
        feet = details.get("feet", {}) if details else {}
        slip = None
        if feet:
            slip = sum(float(item["slip_distance_during_contact"]) for item in feet.values())
        energy = details.get("energy", {}) if details else {}
        foot_metrics = detailed_foot_metrics(summary_path.parent / "tracking_errors_detailed.csv", reference["fps"])
        joint_vel_rmse = float(summary["metrics"]["error_joint_vel"]["rmse"])
        ref_joint_vel_mag_rms = reference["joint_velocity"]["magnitude_rms"]
        record = {
            "motion": motion,
            "final_iteration": state["motions"].get(motion, {}).get("final_iteration"),
            "checkpoint": summary["checkpoint"],
            "motion_file": str(reference_path),
            "fps": reference["fps"],
            "frames": reference["frames"],
            "duration_seconds": reference["duration_seconds"],
            "completion_ratio": completion_ratio,
            "completion_score": 100.0 * completion_ratio,
            "completed_without_termination": bool(summary["completed_without_termination"]),
            "termination_reasons": summary.get("termination_reasons", []),
            "success_single_start": bool(summary["completed_without_termination"] and recorded == planned),
            "error_rmse": {key: float(value["rmse"]) for key, value in summary["metrics"].items()},
            "reference_dynamics": reference,
            "normalized_joint_velocity_error": joint_vel_rmse / ref_joint_vel_mag_rms,
            "foot_slip_distance_total_m": foot_metrics["distance_total_m"] if foot_metrics else slip,
            "foot_slip_speed_mean_mps": foot_metrics["speed_mean_mps"] if foot_metrics else None,
            "foot_slip_speed_rms_mps": foot_metrics["speed_rms_mps"] if foot_metrics else None,
            "foot_slip_speed_p95_mps": foot_metrics["speed_p95_mps"] if foot_metrics else None,
            "foot_slip_speed_max_mps": foot_metrics["speed_max_mps"] if foot_metrics else None,
            "foot_slip_boundary_events": foot_metrics["boundary_events"] if foot_metrics else None,
            "foot_slip_excluding_final_step": foot_metrics["excluding_final_step"] if foot_metrics else None,
            "mean_absolute_joint_power": energy.get("mean_absolute_joint_power"),
            "absolute_joint_energy": energy.get("absolute_joint_energy"),
            "action_norm": None,
            "action_delta_norm": None,
            "root_yaw_tracking_error": None,
            "multi_start_success_rate": None,
        }
        records.append(record)

    (OUT / "extended_metrics.json").write_text(json.dumps(records, indent=2) + "\n")
    fields = [
        "motion", "final_iteration", "frames", "duration_seconds", "completion_score",
        "success_single_start", "anchor_pos_rmse", "body_pos_rmse", "body_rot_rmse",
        "joint_pos_rmse", "joint_vel_rmse", "reference_joint_vel_magnitude_rms",
        "normalized_joint_velocity_error", "reference_joint_acc_component_rms",
        "reference_joint_jerk_component_rms", "foot_slip_distance_total_m",
        "foot_slip_speed_mean_mps", "foot_slip_speed_rms_mps", "foot_slip_speed_p95_mps", "foot_slip_speed_max_mps",
        "mean_absolute_joint_power", "absolute_joint_energy",
    ]
    with (OUT / "extended_metrics.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for r in records:
            writer.writerow({
                "motion": r["motion"],
                "final_iteration": r["final_iteration"],
                "frames": r["frames"],
                "duration_seconds": r["duration_seconds"],
                "completion_score": r["completion_score"],
                "success_single_start": r["success_single_start"],
                "anchor_pos_rmse": r["error_rmse"]["error_anchor_pos"],
                "body_pos_rmse": r["error_rmse"]["error_body_pos"],
                "body_rot_rmse": r["error_rmse"]["error_body_rot"],
                "joint_pos_rmse": r["error_rmse"]["error_joint_pos"],
                "joint_vel_rmse": r["error_rmse"]["error_joint_vel"],
                "reference_joint_vel_magnitude_rms": r["reference_dynamics"]["joint_velocity"]["magnitude_rms"],
                "normalized_joint_velocity_error": r["normalized_joint_velocity_error"],
                "reference_joint_acc_component_rms": r["reference_dynamics"]["joint_acceleration"]["component_rms"],
                "reference_joint_jerk_component_rms": r["reference_dynamics"]["joint_jerk"]["component_rms"],
                "foot_slip_distance_total_m": r["foot_slip_distance_total_m"],
                "foot_slip_speed_mean_mps": r["foot_slip_speed_mean_mps"],
                "foot_slip_speed_rms_mps": r["foot_slip_speed_rms_mps"],
                "foot_slip_speed_p95_mps": r["foot_slip_speed_p95_mps"],
                "foot_slip_speed_max_mps": r["foot_slip_speed_max_mps"],
                "mean_absolute_joint_power": r["mean_absolute_joint_power"],
                "absolute_joint_energy": r["absolute_joint_energy"],
            })

    lines = [
        "# Ten-motion GMR baseline: extended metric pilot",
        "",
        "CPU-only analysis of existing artifacts; no policy was rerun. Completion is a single deterministic rollout from frame 0.",
        "",
        "| Motion | Iter | Completion | Anchor pos | Body pos | Joint pos | Joint vel | Ref joint-vel RMS | Normalized joint-vel error |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        e = r["error_rmse"]
        lines.append(
            f"| {r['motion']} | {r['final_iteration']} | {r['completion_score']:.1f}% | "
            f"{e['error_anchor_pos']:.4f} | {e['error_body_pos']:.4f} | "
            f"{e['error_joint_pos']:.4f} | {e['error_joint_vel']:.4f} | "
            f"{r['reference_dynamics']['joint_velocity']['magnitude_rms']:.4f} | "
            f"{r['normalized_joint_velocity_error']:.4f} |"
        )
    lines += [
        "",
        "Normalized joint-velocity error = policy joint-velocity error RMSE / reference joint-velocity vector-magnitude RMS. It is a diagnostic ratio, not a replacement for raw RMSE.",
        "",
        "## Detailed metrics currently available",
        "",
        "| Motion | Foot slip total | Slip mean | Slip RMS | Slip P95 | Slip max | Mean joint power | Joint energy |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for r in records:
        lines.append(
            f"| {r['motion']} | {fmt(r['foot_slip_distance_total_m'])} | "
            f"{fmt(r['foot_slip_speed_mean_mps'])} | {fmt(r['foot_slip_speed_rms_mps'])} | "
            f"{fmt(r['foot_slip_speed_p95_mps'])} | {fmt(r['foot_slip_speed_max_mps'])} | "
            f"{fmt(r['mean_absolute_joint_power'])} | {fmt(r['absolute_joint_energy'])} |"
        )
    lines += [
        "",
        "Only `zhu0201_upright_50fps` and `zhu0202_01` have detailed evaluation artifacts. N/A values cannot be recovered reliably from legacy summaries.",
        "Raw slip RMS/max are contaminated by a final-step discontinuity in both detailed evaluations (about 72 m/s). These raw values are retained for provenance, not interpreted as physical slip. See JSON `foot_slip_boundary_events` and `foot_slip_excluding_final_step` for the explicitly labeled diagnostic view.",
        "",
        "## Metrics requiring a future unified rollout",
        "",
        "- Signed anchor XYZ and bias-corrected Z RMSE for the first eight actions.",
        "- Root-yaw tracking mean/RMSE/P95/max.",
        "- Action norm and action-delta norm.",
        "- Foot-slip speed mean/RMS/P95/max.",
        "- Uniform multi-start completion/success/fall rate and failure-reason counts.",
        "",
        "These require simulator state/action capture and are intentionally deferred while `proto_taichi1` occupies the GPU.",
    ]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n")
    print(f"Wrote {len(records)} motions to {OUT}")


if __name__ == "__main__":
    main()
