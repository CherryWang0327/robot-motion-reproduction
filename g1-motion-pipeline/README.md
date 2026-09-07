# G1 Motion Pipeline

An auditable, human-in-the-loop pipeline that converts a monocular human-motion video into validated Unitree G1 reference motion for Whole-Body Tracking (WBT).

It wraps existing research/runtime components—GVHMR, GMR or ProtoMotions/PyRoki, and WBT—without modifying their source trees. Every run records commands, artifacts, validation results, and visual evidence in one self-contained run directory.

> **Safety boundary:** this project prepares and validates offline motion assets. It does not send control commands to a physical robot. Training requires explicit human approval.

## Why this project

Robot-motion workflows often fail silently at representation boundaries: coordinate conventions, frame rates, joint layouts, or a retargeted motion that looks plausible but is not usable by a downstream policy. This project makes those boundaries explicit and reviewable.

- Upload one video through a lightweight web console.
- Select either the **GMR** or **ProtoMotions / PyRoki** route.
- Inspect stage-level artifacts and videos before training.
- Validate WBT-ready motion automatically.
- Require human approval before GPU training is queued.

## Pipeline

```text
Input video
  │
  ├─ GVHMR → world-space human motion + SMPL export
  │
  ├─ Route A: GMR → Unitree G1 PKL → WBT NPZ
  │
  └─ Route B: AMASS (Y-up) → ProtoMotions MotionLib → PyRoki G1 PT → WBT NPZ
                                                        │
                                                  validation gate
                                                        │
                                              human approval for training
```

The ProtoMotions route follows the verified coordinate contract:

```text
GVHMR hmr4d_results.pt
→ Y-up AMASS NPZ
→ convert_amass_to_proto.py --source-y-up
→ .motion → MotionLib PT
→ SMPL Motion Tracker
→ PyRoki G1 PT → WBT NPZ
```

## Real-robot deployment contribution

In addition to the offline pipeline and web console, I authored a **160D real-robot policy deployment adapter** that connects WBT robot-reference motion to a Unitree G1 policy observation. The adapter selectively combines the safe ROS/DDS runtime structure of a 177D SMPL deployment with the 58D robot-reference observation semantics of a 154D deployment, while introducing its own NPZ input path, 160D observation contract, diagnostics, and fail-closed safety gates.

The public repository documents this engineering contribution without publishing physical-robot control code or infrastructure details. See [Real-Robot 160D Deployment Adapter](docs/real-robot-deployment-adapter.md).

## What the web console shows

Each run has one card per stage with a concrete status: **waiting**, **running**, **completed**, **failed**, or **warning**. Result pages provide stage-scoped evidence instead of one opaque output folder.

| Stage | Evidence shown |
|---|---|
| GVHMR | input video, camera-space reconstruction, world-space reconstruction |
| GMR | native GMR G1 replay (when a display session is available) |
| ProtoMotions / PyRoki | validated G1 reference-motion replay |
| WBT | input-validation report and pretrained-policy tracking replay |

Videos are designed for manual playback; the page should load media only when the viewer requests it.

## Demo videos

The following videos are original outputs from one completed offline run (`march8`). The animated previews are intentionally lightweight because GitHub does not reliably inline repository MP4 files. Click a preview to open its original MP4.

| GVHMR world-space reconstruction | Native GMR G1 reference replay | WBT policy tracking replay |
|---|---|---|
| [![GVHMR world-space reconstruction](assets/demo/gvhmr_world.gif)](assets/demo/gvhmr_world.mp4) | [![Native GMR G1 reference replay](assets/demo/gmr_g1.gif)](assets/demo/gmr_g1.mp4) | [![WBT policy tracking replay](assets/demo/wbt_tracking.gif)](assets/demo/wbt_tracking.mp4) |

Together they demonstrate the representation chain from reconstructed human motion, to retargeted G1 reference motion, to policy tracking.

## Run directory contract

```text
runs/<run-name>/
├── 00_job/                         # worker command and launcher log
├── 01_gvhmr/                       # human-motion reconstruction + videos
├── 02_smpl/                        # global SMPL CSV
├── 03_retarget/                    # GMR PKL, or ProtoMotions motion assets
├── 03_wbt/                         # WBT-ready NPZ
├── 04_validation/report.json       # FPS, joints, quaternions, finite values
├── 05_*_reference_preview/         # kinematic G1 reference replay
└── 06_wbt_policy_preview/          # pretrained-policy tracking replay
```

Every executable stage writes `stage.json` with the command, environment-facing paths, timestamp, return code, and status. This makes a result reproducible without treating a screenshot as proof.

## Validation gate

Before a motion can enter the training queue, the pipeline checks the WBT NPZ contract:

- expected frame rate;
- joint and rigid-body dimensions;
- finite numeric values;
- valid quaternion data;
- temporal consistency.

The training command is only planned by default. A separate approval action is required before a GPU job may start.

## Local setup

This orchestration repository expects the upstream projects and Conda environments to already exist on the execution machine:

- GVHMR
- GMR
- ProtoMotions and PyRoki
- Whole-Body Tracking

Install this project as an editable package, then run the local web console:

```bash
pip install -e .
python -m server.app --host 127.0.0.1 --port 8000
```

Open `http://127.0.0.1:8000` and upload a video. The pipeline worker is intentionally local to the GPU/runtime machine.

For the research-first command-line workflow, see the [Manual Reproduction Guide](docs/manual-reproduction.md). For the upload-to-approval web workflow, see [Web Console and Automation Architecture](docs/web-console.md).

## CLI examples

```bash
# Print the intended pipeline stages without running simulation.
g1-pipeline run \
  --video /path/to/input.mp4 \
  --route protomotions \
  --run-dir runs/demo_proto \
  --dry-run

# Run video → validated WBT motion. This does not start training.
g1-pipeline run \
  --video /path/to/input.mp4 \
  --route gmr \
  --run-dir runs/demo_gmr

# Validate an existing WBT asset.
g1-pipeline validate \
  --motion /path/to/motion.npz \
  --output runs/check/04_validation/report.json
```

## Portfolio / deployment note

For a public portfolio, deploy the UI, task metadata, and archived demo videos to infrastructure you personally control. Keep the GPU worker private. A live execution worker can be replaced with archived validated runs after access to a lab or company machine ends.

Do not publish credentials, internal IP addresses, proprietary checkpoints, company data, or robot-control code.

## Limitations and next steps

- The native ProtoMotions viewer is optional and depends on a compatible desktop/display session.
- A production deployment should separate the public web/API layer from the GPU worker through a queue and object storage.
- Media delivery should use on-demand loading and HTTP Range requests to avoid competing with remote desktop traffic.

## License

This repository orchestrates external projects with their own licenses. Verify the licenses and redistribution terms of all upstream assets before publishing a demo or model output.
