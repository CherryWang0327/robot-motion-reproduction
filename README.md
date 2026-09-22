# Robot Motion Reproduction

Private engineering archive for a video-to-Unitree-G1 motion-tracking workflow.

## Featured project

- [G1 Motion Pipeline](g1-motion-pipeline/): an auditable, human-in-the-loop pipeline from monocular video to validated Whole-Body Tracking motion, with stage-level visual evidence and manual approval before training.

This archive contains selected orchestration code, documentation, and the documented reproduction artifacts. See [ARCHIVE_SCOPE.md](ARCHIVE_SCOPE.md) for the publication boundary.

## Reproduction artifacts

- `programs/deploy_real/`: deployment program and its documented policy/motion inputs.
- `programs/playback_scripts/`: local playback and visualization helpers.
- `reproduction_assets/input/`: GVHMR SMPL CSV inputs and the related video CSV.
- `reproduction_assets/output/`: generated G1 NPZ outputs and the smoke-test output.

The bundled source documents describe coordinate conventions, frame rates, manifests, and deployment caveats.
