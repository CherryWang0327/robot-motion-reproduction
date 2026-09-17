# Reference temporal fault evaluation

## Scope

This adds the isolated evaluation task `Tracking-Flat-G1-RefFaultEval-v0`.
It does not modify the behavior or registration of `Tracking-Flat-G1-v0`,
the source NPZ, reward definitions, termination definitions, or checkpoints.

The clean `MotionCommand` remains the source for initialization, reward,
termination, and tracking metrics.  Only the policy observation terms below
read the separate fault stream:

- `command` (58 dimensions: 29 joint positions + 29 joint velocities);
- `motion_anchor_pos_b`;
- `motion_anchor_ori_b`.

## Added files

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/reference_faults.py`
  - `ReferenceFaultMotionCommand` owns a separate resettable ring buffer for
    each parallel environment, and exposes the policy-only fault stream.
  - Parameters: `reference_delay_steps`, `reference_freeze_steps`, and
    `reference_freeze_start_step` (default 100 frames = 2 seconds at 50 Hz).
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/reference_fault_observations.py`
  - Provides the three faulted observation functions.
- `scripts/evaluate_reference_faults.py`
  - Runs the required smoke protocol or an explicitly requested 5 x 4 grid.

## Modified integration files

- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/mdp/__init__.py`
  exports the new fault MDP functions.
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/flat_env_cfg.py`
  defines `G1FlatRefFaultEvalEnvCfg`.
- `source/whole_body_tracking/whole_body_tracking/tasks/tracking/config/g1/__init__.py`
  registers `Tracking-Flat-G1-RefFaultEval-v0`.

## Evaluation commands

Use the Conda environment `whole_body_tracking`:

```bash
cd /home/unitree/projects/whole_body_tracking
source /home/unitree/miniconda3/etc/profile.d/conda.sh
conda activate whole_body_tracking
```

Run the smoke test first:

```bash
bash -c 'tabs(){ :; }; source /home/unitree/projects/IsaacLab/isaaclab.sh -p scripts/evaluate_reference_faults.py --smoke-test --max-steps 105 --headless --disable-fabric'
```

After smoke approval, run the complete grid and record a video for every
condition:

```bash
bash -c 'tabs(){ :; }; source /home/unitree/projects/IsaacLab/isaaclab.sh -p scripts/evaluate_reference_faults.py --run-grid --video --headless --disable-fabric'
```

Results are written under `results/reference_faults/`, including `summary.csv`,
`fault_grid.json`, per-condition summaries, tracking logs, observation traces,
and (with `--video`) videos.

## Validation log — 2026-09-03

- `python -m py_compile` passed for all added/modified Python files.
- `git diff --check` passed.
- Motion inspected read-only: `taichi1_gmr_50fps.npz` has 4003 frames at 50 Hz
  (80.06 seconds), with `joint_pos` shape `(4003, 29)`.
- The default execution namespace hid `/dev/nvidia*`, but an elevated host-GPU
  execution context exposed the RTX 5070 Ti (driver `580.173.02`) successfully.
- The 105-step smoke completed on `cuda:0` for all three fault conditions.
  `delay=0, freeze=0` completed 105/105 steps. `delay=2` verified a two-frame
  lag from smoke step 2 onward. `freeze=2` held reference frame 99 for clean
  frames 100 and 101, then resumed at frame 102.
- An independent 105-step `Tracking-Flat-G1-v0` run was compared row-by-row to
  the no-fault task. The maximum absolute difference for all six primary
  tracking errors was exactly `0.0`; both completed without termination.
- Evidence is saved in `results/reference_faults/smoke_test.json`,
  `results/reference_faults/summary.csv`, and
  `results/reference_faults/baseline_original_task/`.

## Rollback

The repository was already dirty before this work. Do not use broad `git
restore` commands. To remove this feature only, delete the three added files,
then remove the corresponding fault-MDP exports, isolated config class, and
task registration listed above.

## Current execution progress — 2026-09-03

### Completed validation

- GPU access is working in the host-GPU execution context: RTX 5070 Ti,
  NVIDIA driver `580.173.02`, `cuda:0`.
- The 105-step smoke completed. The independent original-task comparison and
  fault-stream checks are recorded in `results/reference_faults/smoke_test.json`.
- Static validation passed: `py_compile` and `git diff --check`.

### Full grid status

- The full 5 x 4 grid was confirmed and is currently running as one clean
  process, PID `265417`:

  ```text
  scripts/evaluate_reference_faults.py --run-grid --headless --disable-fabric
  ```

- At the last check the process had run for 4 minutes 44 seconds and used about
  2.7 GiB GPU memory. The user-owned training process remains untouched.
- The first full condition, `delay_00_freeze_00`, has been regenerated and its
  `summary.json` timestamp is 15:21. The process writes a condition summary
  only after that condition completes, so additional entries appear gradually.
- Existing smoke directories (`delay_00_freeze_02` and `delay_02_freeze_00`)
  are retained; they are not evidence of full-grid completion.

### Video status

- The numeric grid is running without video. Isaac Sim 4.5's rendering-kit
  launch path failed when `--video` enabled (no active physics scene during
  rendering-context initialization). This does not affect the headless physics
  evaluation path.
- Representative videos remain pending a separate rendering-context fix; do
  not interpret their absence as a missing numeric condition.

### Safe monitoring

```bash
ps -p 265417 -o pid,etime,stat,cmd
nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader
find results/reference_faults -mindepth 1 -maxdepth 1 -type d -name 'delay_*' -printf '%f\n' | sort
```
