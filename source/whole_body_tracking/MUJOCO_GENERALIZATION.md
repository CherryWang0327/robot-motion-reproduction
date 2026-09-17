# Taichi ONNX MuJoCo generalization baseline

Policy under test:

```text
logs/rsl_rl/g1_flat/2026-08-18_18-11-23_baseline_proto_zalign_taichi1_100k/exported/policy.onnx
```

Reference input:

```text
retarget_comparison/height_aligned/taichi1/taichi1_pyroki_height_aligned_50fps.npz
```

Run tests with the `gmr` environment, which supplies MuJoCo and ONNX Runtime:

```bash
/home/unitree/miniconda3/envs/gmr/bin/python \
  scripts/run_mujoco_generalization.py \
  --motion-file retarget_comparison/height_aligned/taichi1/taichi1_pyroki_height_aligned_50fps.npz \
  --policy-file logs/rsl_rl/g1_flat/2026-08-18_18-11-23_baseline_proto_zalign_taichi1_100k/exported/policy.onnx \
  --output results/mujoco_generalization/taichi_zalign_baseline.json
```

The evaluator never modifies the policy, motion, XML, or deployment files. It
records fall status, the first fall time, and minimum root-height/up-axis
signals in the requested JSON result.

## 2026-09-17 baseline scan

All runs played the full 4003-frame, 50 Hz reference. Fall thresholds were
root height below 0.45 m or root up-axis below 0.45.

| Condition | Outcome | Minimum root height | Minimum up-axis |
|---|---|---:|---:|
| Nominal | pass | 0.693 m | 0.957 |
| 0.20 friction scale | pass | 0.692 m | 0.957 |
| 1.25 body-mass scale | pass | 0.670 m | 0.946 |
| 0.20 friction + 1.25 mass + 300 N lateral push at 15 s | pass | 0.621 m | 0.930 |
| 0.20 friction + 1.25 mass + 600 N lateral push at 15 s | pass | 0.526 m | 0.717 |
| 0.20 friction + 1.25 mass + 800 N lateral push at 15 s | **fall at 15.698 s** | 0.055 m | -0.777 |

The scan establishes an observed recovery boundary between the 600 N and 800 N
combined lateral-push cases. It does not certify real-robot safety.

## Targeted robustness training

`Tracking-Flat-G1-Robust-v0` is an independent task; it leaves
`Tracking-Flat-G1-v0` unchanged. It expands the material randomization lower
bound to 0.20 and increases the existing random-velocity recovery events.
Use it only after GPU availability is restored:

```bash
/home/unitree/miniconda3/envs/whole_body_tracking/bin/python \
  scripts/rsl_rl/train.py \
  --task Tracking-Flat-G1-Robust-v0 \
  --motion_file retarget_comparison/height_aligned/taichi1/taichi1_pyroki_height_aligned_50fps.npz \
  --num_envs 4096 \
  --max_iterations 30000 \
  --seed 0 \
  --headless \
  --logger tensorboard \
  --run_name taichi1_proto_zalign_robust
```

Keep the actor/critic capacity unchanged for this first targeted ablation. Only
consider a deeper network if the new run improves neither the 600 N recovery
margin nor tracking quality after convergence.
