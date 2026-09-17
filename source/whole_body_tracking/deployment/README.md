# 160D robot-reference real deployment adapter

This directory is an independent development copy. It does not modify either
teacher reference script:

- `deploy_real_smpl_177.py` supplies the ROS odometry and safe runtime structure.
- `deploy_real4bydmimic.py` supplies the older robot-reference joint/control pattern.

## Deployment policy selection

The validated Taichi / `proto_zalign_taichi1` deployment policy is **separate
from the March motion pipeline**.  For the 160D deployment adapter, use this
Taichi export on the Unitree machine:

```text
/home/unitree/projects/whole_body_tracking/logs/rsl_rl/g1_flat/2026-08-18_18-11-23_baseline_proto_zalign_taichi1_100k/exported/policy.onnx
```

It is paired with the corresponding Taichi robot-reference motion input, not
with `march_video_gmr_50fps.npz` or any March-trained checkpoint.  March
training/replay is an independent GMR tracking workflow and must select its
own run directory, checkpoint, motion file, and any exported policy.

The ONNX file is deliberately not stored in this source repository; retrieve
it from the above Unitree path (or export it again from that training run).

## Observation contract

| Slice | Size | Meaning |
|---|---:|---|
| `0:58` | 58 | reference robot joint position + velocity |
| `58:61` | 3 | reference torso position relative to current torso, in current torso frame |
| `61:67` | 6 | relative torso orientation, first two rotation-matrix columns |
| `67:70` | 3 | root/base linear velocity in base frame |
| `70:73` | 3 | root/base angular velocity in base frame |
| `73:102` | 29 | measured joint position minus nominal default |
| `102:131` | 29 | measured joint velocity |
| `131:160` | 29 | previous policy action |

The first 58 values are robot-reference motion, not the 75D custom SMPL command
used by the 177D policy.

## Verified offline

Run from the repository root:

```bash
/home/unitree/miniconda3/envs/gmr/bin/python -m unittest discover -s deployment/tests -v

/home/unitree/miniconda3/envs/gmr/bin/python deployment/deploy_real_robotref_160.py \
  --offline-validate \
  --motion PATH_TO_WBT_MOTION.npz \
  --policy PATH_TO_160D_POLICY.onnx
```

Offline mode imports neither ROS nor Unitree DDS and cannot publish commands.

## Live dry-run gate

Live dry-run subscribes to Unitree low state and `/Odometry_2`, builds the 160D
observation, and runs ONNX inference. It does not create a low-command publisher.

Before using it, the odometry publisher owner must confirm:

1. `pose` is the world pose of `torso_link`, or provide a pelvis-to-torso position transform.
2. `twist.linear` is expressed in the robot base/pelvis frame.
3. ROS `header.frame_id` and `child_frame_id` semantics.
4. IMU mounting frame (`pelvis` or `torso`).

Control publishing is code-level disabled in this development revision. It must
remain disabled until a live no-command dry-run establishes frame semantics,
latency, first-frame values, and teacher/on-site approval. The acknowledgement
and frame gates are additional safeguards for a later reviewed revision; they do
not replace a suspension rig, clear area, or emergency stop.

## Known boundary

Offline numerical correctness and MuJoCo behavior can be verified on this machine.
The physical meaning, latency, sign, and frame of `/Odometry_2` cannot be certified
without inspecting the live publisher and running a no-command dry-run.
