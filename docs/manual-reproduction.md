# Manual Reproduction Guide

This guide records the research-first workflow used to validate each representation boundary before it was automated by the web console.

## 1. Reconstruct world-space human motion

Run GVHMR on an input video, retain its native world-space result, and export the standard 80-column SMPL CSV. Inspect the input, camera-space, and world-space videos before continuing.

Expected evidence:

- `hmr4d_results.pt`: reconstructed motion result;
- standard SMPL CSV (`fps`, orientation, 63D body pose, translation, and betas) shared by both retargeting routes;
- camera-space and world-space reconstruction videos.

## 2. Choose and validate one retargeting route

### Route A: GMR

Use the shared SMPL CSV to generate a Unitree G1 PKL. Record the native GMR replay, then inspect joint limits, orientation, contact behavior, and the visual plausibility of the G1 motion.

### Route B: ProtoMotions and PyRoki

The validated coordinate contract is:

```text
GVHMR hmr4d_results.pt (provenance)
-> standard SMPL CSV (shared interface)
-> Y-up AMASS NPZ (derived from CSV)
-> convert_amass_to_proto.py --source-y-up
-> .motion -> MotionLib PT
-> SMPL Motion Tracker
-> PyRoki G1 PT
```

The Y-up conversion step is intentional. Skipping or changing this convention can produce visually invalid robot motion even when files are generated successfully.

## 3. Convert and validate WBT motion

Convert the selected G1 reference motion to WBT NPZ, then verify the format contract before any training request:

- frame rate;
- joint and rigid-body shapes;
- finite values;
- quaternion validity;
- temporal consistency.

## 4. Run a pretrained tracking preview

Use a pretrained WBT tracking policy only as an offline validation preview. A visually plausible preview does not replace numerical validation; both are required.

## Why keep this manual path

The command-line path is the ground truth for debugging: every automated stage maps to a previously verified manual operation. It is intentionally documented separately from the web console so failures can be isolated at the model, conversion, or orchestration layer.
