#!/usr/bin/env python3
"""Zero-training evaluation of short high-dynamics policy-reference phase faults."""
import argparse, csv, json, sys
from dataclasses import fields
from pathlib import Path
import numpy as np
from isaaclab.app import AppLauncher

sys.path.insert(0, str(Path(__file__).resolve().parent / "rsl_rl"))
import cli_args  # noqa: E402

ROOT = Path("/home/unitree/projects/whole_body_tracking")
CHECKPOINT = ROOT / "logs/rsl_rl/g1_flat/2026-08-05_09-09-29_taichi1_30k_local_resume/model_29999.pt"
MOTION = ROOT / "inputs/taichi1/taichi1_gmr_50fps.npz"
TASK = "Tracking-Flat-G1-PhaseFaultEval-v0"
ERRORS = ("error_anchor_pos", "error_anchor_rot", "error_body_pos", "error_body_rot", "error_joint_pos", "error_joint_vel")

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--smoke-test", action="store_true")
parser.add_argument("--run-small-grid", action="store_true", help="One selected window x four fault types x 20 frames.")
parser.add_argument("--window-index", type=int, default=0)
parser.add_argument("--output-dir", type=Path, default=ROOT / "results/reference_phase_faults")
parser.add_argument("--motion-file", type=Path, default=MOTION)
parser.add_argument("--checkpoint-path", type=Path, default=CHECKPOINT)
parser.add_argument("--num-envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--disable-fabric", action="store_true")
cli_args.add_rsl_rl_args(parser); AppLauncher.add_app_launcher_args(parser)
args, hydra_args = parser.parse_known_args(); sys.argv = [sys.argv[0], *hydra_args]
app = AppLauncher(args).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402
from isaaclab.envs import ManagerBasedRLEnvCfg  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper  # noqa: E402
from isaaclab_tasks.utils.hydra import hydra_task_config  # noqa: E402
import whole_body_tracking.tasks  # noqa: E402,F401


def dynamic_windows(path: Path, window=30, guard=50, count=3):
    data = np.load(path); vel = np.asarray(data["joint_vel"], dtype=np.float64)
    fps = float(np.asarray(data["fps"]).reshape(-1)[0])
    acc = np.diff(vel, axis=0) * fps; jerk = np.diff(acc, axis=0) * fps
    score = np.zeros(len(vel)); score[1:] = np.linalg.norm(acc, axis=1); score[2:] += .05 * np.linalg.norm(jerk, axis=1)
    means = np.convolve(score, np.ones(window) / window, mode="valid"); picked=[]
    for start in np.argsort(means)[::-1]:
        end = int(start + window - 1)
        if start < guard or end >= len(vel) - guard: continue
        if all(end < s or start > e for s, e, _ in picked):
            picked.append((int(start), end, float(means[start])))
            if len(picked) == count: break
    return [{"start_frame": s, "end_frame": e, "start_seconds": s/fps, "end_seconds": e/fps,
             "score": q, "score_definition": "30-frame mean(L2 joint acceleration + 0.05*L2 joint jerk)",
             "selection_reason": "top non-overlapping score after excluding first/last 1 s"} for s,e,q in sorted(picked)]


def configure(cfg):
    cfg.scene.num_envs=args.num_envs; cfg.commands.motion.motion_file=str(args.motion_file.resolve())
    cfg.observations.policy.enable_corruption=False
    for f in fields(cfg.events): setattr(cfg.events, f.name, None)
    cfg.commands.motion.debug_vis=False; cfg.commands.motion.pose_range={k:(0.,0.) for k in cfg.commands.motion.pose_range}
    cfg.commands.motion.velocity_range={k:(0.,0.) for k in cfg.commands.motion.velocity_range}; cfg.commands.motion.joint_position_range=(0.,0.); cfg.episode_length_s=1.e9


def place_zero(env):
    c=env.unwrapped.command_manager.get_term("motion"); ids=torch.arange(env.unwrapped.num_envs,device=env.unwrapped.device); r=c.robot; c.time_steps.zero_()
    r.write_joint_state_to_sim(torch.clamp(c.joint_pos,r.data.soft_joint_pos_limits[:,:,0],r.data.soft_joint_pos_limits[:,:,1]),c.joint_vel,env_ids=ids)
    r.write_root_state_to_sim(torch.cat([c.body_pos_w[:,0],c.body_quat_w[:,0],c.body_lin_vel_w[:,0],c.body_ang_vel_w[:,0]],-1),env_ids=ids)
    env.unwrapped.scene.write_data_to_sim(); env.unwrapped.sim.forward(); env.unwrapped.scene.update(dt=env.unwrapped.physics_dt); c.time_steps.fill_(-1); c._update_command(); return c


def segment(rows, start, end):
    part=[r for r in rows if start <= r["step"] < end]
    return {n: float(np.mean([r[n] for r in part])) if part else None for n in ERRORS}


def run(wrapped, policy, window, mode, magnitude, out, step_limit=None):
    c=wrapped.unwrapped.command_manager.get_term("motion"); c.cfg.phase_fault_mode=mode; c.cfg.phase_fault_start_step=window["start_frame"]; c.cfg.phase_fault_duration_steps=10; c.cfg.phase_fault_magnitude_steps=magnitude; c.cfg.phase_fault_seed=args.seed
    wrapped.reset(); c=place_zero(wrapped); obs,_=wrapped.get_observations(); rows=[]; done=None; total=c.motion.time_step_total-1
    if step_limit is not None: total=min(total, step_limit)
    for step in range(total):
        with torch.no_grad(): action=policy(obs); obs,_,dones,_=wrapped.step(action)
        rows.append({"step":step,"clean_reference_frame":int(c.time_steps[0]),"policy_reference_frame":int(c.policy_source_time_step[0]),"fault_active":bool(window["start_frame"] <= int(c.time_steps[0]) < window["start_frame"]+10 and mode!="clean"), **{n:float(c.metrics[n][0]) for n in ERRORS}})
        if bool(dones[0]): done=step; break
    out.mkdir(parents=True,exist_ok=True)
    with (out/"tracking_errors.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=rows[0]); w.writeheader(); w.writerows(rows)
    with (out/"observation_fault_trace.json").open("w") as f: json.dump([{k:r[k] for k in ("step","clean_reference_frame","policy_reference_frame","fault_active")} for r in rows],f,indent=2)
    # Loop row N is clean reference frame N+1. Keep all segmented metrics
    # aligned to the selected clean-motion window rather than loop indexing.
    start=window["start_frame"]-1; summary={"mode":mode,"magnitude_steps":magnitude,"fault_start_frame":window["start_frame"],"fault_duration_steps":10,"completed_full_motion":done is None,"first_termination_frame":done,"episode_steps":len(rows),"segments":{"pre":segment(rows,start-20,start),"fault":segment(rows,start,start+10),"post":segment(rows,start+10,start+70)}}
    with (out/"summary.json").open("w") as f: json.dump(summary,f,indent=2)
    return summary,rows


@hydra_task_config(TASK,"rsl_rl_cfg_entry_point")
def main(env_cfg: ManagerBasedRLEnvCfg, agent_cfg: RslRlOnPolicyRunnerCfg):
    if not args.smoke_test and not args.run_small_grid: parser.error("Use --smoke-test or --run-small-grid.")
    root=args.output_dir.resolve(); root.mkdir(parents=True,exist_ok=True); windows=dynamic_windows(args.motion_file)
    with (root/"fault_windows.json").open("w") as f: json.dump({"fps":50,"windows":windows},f,indent=2)
    window=windows[args.window_index]; configure(env_cfg); env=gym.make(TASK,cfg=env_cfg); wrapped=RslRlVecEnvWrapper(env); runner=OnPolicyRunner(wrapped,agent_cfg.to_dict(),log_dir=None,device=agent_cfg.device); runner.load(str(args.checkpoint_path)); policy=runner.get_inference_policy(device=wrapped.unwrapped.device)
    cases=[("clean",0)] + ([(m,20) for m in ("phase_lag","phase_lead","freeze","jitter_delay")] if args.smoke_test or args.run_small_grid else [])
    flat=[]
    for mode,mag in cases:
        smoke_limit=window["end_frame"]+31 if args.smoke_test else None
        s,_=run(wrapped,policy,window,mode,mag,root/f"window_{args.window_index}_{mode}_{mag:02d}",smoke_limit)
        flat.append({"window_index":args.window_index,"mode":mode,"magnitude_steps":mag,"completed_full_motion":s["completed_full_motion"],"first_termination_frame":s["first_termination_frame"],**{f"fault_{n}":s["segments"]["fault"][n] for n in ERRORS},**{f"post_{n}":s["segments"]["post"][n] for n in ERRORS}})
    with (root/"summary.csv").open("w",newline="") as f: w=csv.DictWriter(f,fieldnames=flat[0]); w.writeheader(); w.writerows(flat)
    wrapped.close()

if __name__=="__main__": main(); app.close()
