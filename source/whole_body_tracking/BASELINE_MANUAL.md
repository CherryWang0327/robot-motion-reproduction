# Ten-Motion Whole-Body Tracking Baseline Manual

## 1. Fixed experiment protocol

This baseline uses one independent policy per motion and one run per motion.

- Task: `Tracking-Flat-G1-v0`
- Seed: `0` only
- Training environments: `4096`
- Minimum training: `30000` iterations
- Maximum training: `100000` iterations
- Checkpoint interval: `500` iterations
- Stop: the first saved checkpoint after the convergence test passes at or after iteration 30000
- Success: deterministic evaluation reaches the last reference frame without early termination
- Tracking errors: reported as measurements, not used as the primary pass/fail threshold

Do not tune rewards, PPO settings, termination thresholds, or environment settings for an individual motion. Run only one
GPU training job at a time unless the available GPU memory has been independently verified.

## 2. Smoke-test decision

A 16-environment smoke training cannot predict convergence and is not part of the baseline. It is not required for these
ten files because they have already been checked to share the required schema (50 FPS, 29 joints, 30 bodies), contain no
NaN/Inf, and the training path has been exercised by `taichi1`.

Instead, run the CPU-only input validation once. If it passes, start each formal 4096-environment run directly. During the
first few iterations, verify that TensorBoard events and `model_0.pt` appear. A smoke run becomes necessary only after code,
robot configuration, or NPZ generation changes, or if formal training fails during startup.

## 3. Enter the project

```bash
cd /home/unitree/projects/whole_body_tracking
```

All commands below use this interpreter:

```text
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python
```

## 4. Validate all ten inputs once

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/validate_motion_inputs.py \
  /home/unitree/projects/whole_body_tracking/inputs/1438/1438_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/1439/1439_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/happy/happy_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/taichi1/taichi1_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/zhu0129/zhu0129_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/zhu0201_02_whswap_v2/zhu0201_02_whswap_v2_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/zhu0201_03/zhu0201_03_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/zhu0201_04/zhu0201_04_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/zhu0201_upright_50fps/zhu0201_upright_50fps_gmr_50fps.npz \
  /home/unitree/projects/whole_body_tracking/inputs/zhu0202_01/zhu0202_01_gmr_50fps.npz \
  --output results/baseline/input_validation.json
```

Proceed only if all ten lines say `PASS`.

## 5. Running safely with tmux

### What tmux is

`tmux` is a terminal-session manager. A command launched inside a tmux session keeps running on the machine when the SSH
window is closed or the network disconnects. It does not change the training algorithm, files, CUDA configuration, or GPU
results. This machine already has `/usr/bin/tmux` version 3.2a installed.

Using tmux has low risk and is recommended for this multi-day baseline. Its important limitations are:

- tmux protects against an SSH/network disconnection, but not a machine shutdown, reboot, power loss, OOM, or application
  crash. Checkpoint recovery handles those cases.
- Detaching from a session is safe; killing a tmux session terminates the programs inside it.
- Closing a normal SSH window without tmux may send a hangup signal to its processes. Do not rely on that behavior.
- tmux does not reduce GPU or disk usage. Continue to run only one training job at a time.

### Create the persistent baseline session

```bash
tmux new-session -s wbt_baseline
```

Inside the new tmux terminal:

```bash
cd /home/unitree/projects/whole_body_tracking
```

Then launch the orchestrator command from the next section.

### Safely leave training running

Press these keys in sequence (do not type the words):

```text
Ctrl+B
release both keys
D
```

This is called **detach**. It does not stop training.

### List and reconnect to sessions

```bash
tmux list-sessions
```

Reconnect:

```bash
tmux attach-session -t wbt_baseline
```

If tmux says the session is attached elsewhere, use:

```bash
tmux attach-session -d -t wbt_baseline
```

This detaches the old display and attaches the same still-running session here; it does not restart training.

### Scroll through old terminal output

Inside tmux, press `Ctrl+B`, release, then press `[`. Use arrow/PageUp/PageDown to scroll and press `Q` to leave scroll
mode.

### Stop intentionally

To pause the automated baseline safely, attach to the session and press `Ctrl+C` once in the orchestrator terminal. The
orchestrator sends an interrupt to the current trainer, preserves all completed checkpoints, and marks the motion as
interrupted. Wait until the shell prompt returns, then type:

```bash
exit
```

Do not use the following command unless intentionally terminating everything inside that session:

```text
tmux kill-session -t wbt_baseline
```

If a reboot is planned, use the safe `Ctrl+C` procedure first.

## 6. Formal training commands

### Recommended: one command for all ten motions

The orchestrator already contains all ten absolute motion paths. It trains one motion at a time, monitors convergence,
waits for a complete checkpoint, stops after a plateau at/after 30000, exports the report, records the deterministic
full-motion video, and then starts the next motion:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_ten_motion_baseline.py
```

Run it in a persistent terminal multiplexer (`tmux` or `screen`) so closing an SSH window does not terminate a multi-day
job. Progress is stored in `results/baseline/orchestrator_state.json`; motions whose stage is `complete` are skipped when
the command is run again. If training is interrupted after a checkpoint was saved, running the same command again resumes
that motion from its latest checkpoint and trains only the remaining iterations toward the 100000 total budget. Existing
logs and checkpoints are never deleted. At most the iterations since the latest 500-iteration checkpoint must be repeated.

### Recovery after interruption

For runs started by the orchestrator, recovery uses the same command that originally started it. For example:

```bash
cd /home/unitree/projects/whole_body_tracking
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_ten_motion_baseline.py \
  --adopt_completed_run \
  taichi1=/home/unitree/projects/whole_body_tracking/logs/rsl_rl/g1_flat/2026-08-05_09-09-29_taichi1_30k_local_resume
```

Recovery behavior:

1. Motions marked `complete` are skipped.
2. If the recorded training PID is still alive, the new orchestrator refuses to start a duplicate.
3. If the process is dead, the interrupted motion resumes from the largest complete `model_*.pt` checkpoint.
4. Policy, optimizer, normalizers, and iteration number are restored.
5. TensorBoard events and new checkpoints are appended to the same run directory.
6. Training continues toward a total of 100000 iterations; it does not add another 100000.
7. At most the work after the latest 500-iteration checkpoint is repeated.
8. If interruption occurred during reporting/evaluation, training is reused and only the unfinished stage is repeated.

The persistent recovery state is:

```text
results/baseline/orchestrator_state.json
```

Do not delete or manually edit this file during the experiment.

The currently running taichi1 job predates the orchestrator. If it stops before convergence, find its latest checkpoint:

```bash
ls -1v \
  /home/unitree/projects/whole_body_tracking/logs/rsl_rl/g1_flat/2026-08-05_09-09-29_taichi1_30k_local_resume/model_*.pt \
  | tail -n 1
```

Then resume it by substituting the printed checkpoint path below:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/taichi1/taichi1_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_taichi1_100k \
  --resume_checkpoint <PRINTED_CHECKPOINT_PATH> \
  --resume_log_dir /home/unitree/projects/whole_body_tracking/logs/rsl_rl/g1_flat/2026-08-05_09-09-29_taichi1_30k_local_resume
```

After taichi1 finishes and passes convergence (or reaches 100000), use `--adopt_completed_run` so the orchestrator reports
and evaluates it without duplicate training.

Before committing GPU time, print every generated command without running it:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_ten_motion_baseline.py \
  --dry_run --state_file /tmp/whole_body_tracking_baseline_dry_run.json
```

To run only selected motions, for example after handling the existing `taichi1` separately:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_ten_motion_baseline.py \
  --motions 1438 1439 happy zhu0129 zhu0201_02_whswap_v2 \
  zhu0201_03 zhu0201_04 zhu0201_upright_50fps zhu0202_01
```

If the existing taichi1 run has finished and contains its final checkpoint, adopt it and let the orchestrator report/evaluate
it instead of training a duplicate:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/run_ten_motion_baseline.py \
  --adopt_completed_run \
  taichi1=/home/unitree/projects/whole_body_tracking/logs/rsl_rl/g1_flat/2026-08-05_09-09-29_taichi1_30k_local_resume
```

Do not use `--adopt_completed_run` until the external training process has exited and its final checkpoint is completely
written.

### Monitor the entire multi-day run from another terminal

This command requires no run-directory or motion path. It automatically reads the orchestrator state and refreshes every
60 seconds:

```bash
cd /home/unitree/projects/whole_body_tracking
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/watch_ten_motion_baseline.py \
  --watch 60
```

It displays all ten stages, completed count, the current motion, iteration, progress toward the 100000 maximum, ETA,
plateau/convergence decision, and latest checkpoint. `Ctrl+C` in this monitoring terminal stops only the monitor; it does
not stop the orchestrator or training.

### Manual fallback commands

Run these commands sequentially, not simultaneously. `taichi1` is already running/finished in the existing directory, so
do not launch a duplicate unless that run is invalid.

### 1438

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/1438/1438_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_1438_100k
```

### 1439

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/1439/1439_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_1439_100k
```

### happy

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/happy/happy_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_happy_100k
```

### taichi1 (only if a clean restart is required)

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/taichi1/taichi1_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_taichi1_100k
```

### zhu0129

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/zhu0129/zhu0129_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_zhu0129_100k
```

### zhu0201_02_whswap_v2

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/zhu0201_02_whswap_v2/zhu0201_02_whswap_v2_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_zhu0201_02_whswap_v2_100k
```

### zhu0201_03

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/zhu0201_03/zhu0201_03_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_zhu0201_03_100k
```

### zhu0201_04

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/zhu0201_04/zhu0201_04_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_zhu0201_04_100k
```

### zhu0201_upright_50fps

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/zhu0201_upright_50fps/zhu0201_upright_50fps_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_zhu0201_upright_50fps_100k
```

### zhu0202_01

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/zhu0202_01/zhu0202_01_gmr_50fps.npz \
  --num_envs 4096 --max_iterations 100000 --seed 0 --headless \
  --logger tensorboard --run_name baseline_zhu0202_01_100k
```

The training process prints its exact run directory in the form:

```text
logs/rsl_rl/g1_flat/YYYY-MM-DD_HH-MM-SS_baseline_<motion>_100k
```

Copy that exact directory for all remaining commands. In this manual it is called `<RUN_DIRECTORY>`.

## 7. Startup check for each formal run

After training starts, open another terminal and inspect the run directory:

```bash
find <RUN_DIRECTORY> -maxdepth 1 -type f -printf '%f %k KB\n' | sort
```

Continue only when an `events.out.tfevents...` file and `model_0.pt` exist and the training terminal shows finite reward,
episode length, and tracking metrics. If startup fails, fix the data/configuration problem; do not count it as a baseline
failure.

## 8. Monitor convergence

Run this in a second terminal. It is CPU-only and may remain active while training uses the GPU.

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/analyze_training_convergence.py \
  --log_dir <RUN_DIRECTORY> \
  --max_iterations 100000 \
  --min_iterations 30000 \
  --window 1000 \
  --watch 60
```

Interpretation:

- Before iteration 30000, `converged=False` is expected and training must continue.
- At/after 30000, `converged=True` means the previous and recent 1000-iteration windows meet all frozen criteria:
  reward change below 1%, all six error changes below 2%, recent mean episode length at least 495, and finite metrics.
- `plateau=True, converged=False` means a curve plateau exists but the episode-length or finite-value requirement failed.
- At 100000 without convergence, let the run end and record `not_converged`.

When `converged=True`, read `latest checkpoint` from the monitor. Because checkpoints are saved every 500 iterations,
wait until a checkpoint at or after the reported converged iteration has been fully written. Record its filename as
`<FINAL_CHECKPOINT>`, then stop training with `Ctrl+C` in the training terminal. Never delete earlier checkpoints.

## 9. Export the final training report

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/export_training_report.py \
  --log_dir <RUN_DIRECTORY> \
  --max_iterations 100000 \
  --min_iterations 30000 \
  --window 1000
```

Verify:

```bash
find <RUN_DIRECTORY>/training_report -maxdepth 1 -type f -printf '%f %k KB\n' | sort
```

Required files:

- `reward_curve.png`: training mean-reward curve
- `tracking_error_curve.png`: six training tracking-error curves
- `training_metrics.csv`: raw scalar history
- `convergence.json`: final plateau/convergence decision and final metrics

Read the convergence result:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python -m json.tool \
  <RUN_DIRECTORY>/training_report/convergence.json
```

## 10. Deterministic full-motion evaluation and video

Use the same motion file used for training. Replace `<MOTION>`, `<ABSOLUTE_MOTION_PATH>`, `<RUN_DIRECTORY_NAME>` (the
directory name only), and `<FINAL_CHECKPOINT>` with the recorded values.

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file <ABSOLUTE_MOTION_PATH> \
  --load_run <RUN_DIRECTORY_NAME> \
  --checkpoint <FINAL_CHECKPOINT> \
  --evaluation_output /home/unitree/projects/whole_body_tracking/results/baseline/<MOTION>/evaluation \
  --video --headless
```

Example for the existing taichi1 run, if `model_29999.pt` is the chosen final checkpoint:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/rsl_rl/play.py \
  --task Tracking-Flat-G1-v0 \
  --motion_file /home/unitree/projects/whole_body_tracking/inputs/taichi1/taichi1_gmr_50fps.npz \
  --load_run 2026-08-05_09-09-29_taichi1_30k_local_resume \
  --checkpoint model_29999.pt \
  --evaluation_output /home/unitree/projects/whole_body_tracking/results/baseline/taichi1/evaluation \
  --video --headless
```

Do not evaluate while a training process is using the same GPU.

## 11. Verify and interpret evaluation output

```bash
find results/baseline/<MOTION>/evaluation -maxdepth 3 -type f -printf '%p %k KB\n' | sort
```

Required files:

- `video/*.mp4`: final policy video
- `tracking_errors.csv`: errors for every evaluated reference frame
- `summary.json`: completion status, termination reason, and error statistics

Display the summary:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python -m json.tool \
  results/baseline/<MOTION>/evaluation/summary.json
```

Primary baseline success requires all of the following:

```text
completed_without_termination = true
first_termination_step = null
termination_reasons = []
recorded_steps = planned_steps
```

Then watch the complete MP4 once. If all four machine checks pass but the policy visibly does not follow the reference,
record `completed but poor tracking`; do not silently relabel it as a clean success. Tracking-error mean, RMSE, P95, and
maximum values remain required report columns even though no error threshold is used for primary success.

Failure interpretation:

- `anchor_pos`: root/torso height departed from the reference threshold.
- `anchor_ori`: torso orientation/balance departed from the reference threshold.
- `ee_body_pos`: ankle or wrist vertical tracking error exceeded the termination threshold.
- No termination but visibly wrong motion: completed execution with poor tracking; inspect P95/max errors and video.
- `converged=false` at 100000: optimization did not reach the frozen plateau criterion.
- Crash, OOM, missing fields, or corrupt checkpoint: system failure; fix and rerun, not an algorithmic baseline failure.

## 12. Record one final row per motion

Use this table; do not report multi-seed statistics because this protocol uses seed 0 only.

| Motion | Final iteration | Converged | Complete | Anchor-pos RMSE | Body-pos RMSE | Body-rot RMSE | Result | Failure reason |
|---|---:|---|---|---:|---:|---:|---|---|
| 1438 | | | | | | | | |
| 1439 | | | | | | | | |
| happy | | | | | | | | |
| taichi1 | | | | | | | | |
| zhu0129 | | | | | | | | |
| zhu0201_02_whswap_v2 | | | | | | | | |
| zhu0201_03 | | | | | | | | |
| zhu0201_04 | | | | | | | | |
| zhu0201_upright_50fps | | | | | | | | |
| zhu0202_01 | | | | | | | | |

Use `baseline success` only when convergence and full-motion completion both pass. Use `baseline failure` when training
converges but deterministic execution terminates early. Use `not converged` when iteration 100000 is reached without the
convergence criterion. Because this is one seed per motion, describe the final aggregate as “X of 10 single-seed baseline
runs completed the full motion,” not as a statistical success probability.

## 13. Follow-up after all ten motions

1. Confirm every row has a curve, checkpoint, video, CSV, JSON, and failure label.
2. Group failures by termination reason and inspect the failure frame in the video/CSV.
3. Report the unmodified baseline first; do not retrain individual failures with special settings inside this table.
4. If later work is requested, select representative failures and create separate GMR/reward experiments. Those runs are
   follow-up experiments and must not replace the original baseline results.

## 14. Backfill detailed diagnostics without retraining

The standard summary stores aggregate errors. After GPU training has completely finished, re-run the saved checkpoints to
add anchor XYZ, per-joint position/velocity, per-body position/rotation, foot-contact/slip proxy, joint-limit margin, and
power/energy proxy data. This is evaluation only; it does not update or retrain a policy.

Do not run this while the orchestrator is training another policy on the same GPU. Once all ten motions are complete:

```bash
cd /home/unitree/projects/whole_body_tracking
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/rerun_detailed_evaluations.py
```

Existing videos are retained and no duplicate video is recorded by default. Detailed output is added to each evaluation:

```text
results/baseline/<motion>/evaluation/tracking_errors_detailed.csv
results/baseline/<motion>/evaluation/summary.json
```

Then create the readable cross-motion report:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/summarize_detailed_evaluations.py
```

Output:

```text
results/baseline/diagnostics/detailed_metrics.md
```

The foot-slip value is a screening proxy: accumulated horizontal ankle-link displacement while contact force exceeds the
sensor threshold. It is not an exact sole/contact-patch slip measurement and must be checked against video/contact data.
