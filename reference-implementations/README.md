# Reference Deployment Implementations

This directory archives the three source files used for the 160D deployment comparison:

- `154D/deploy_real4bydmimic.py`
- `177D/deploy_real_smpl_177.py`
- `160D/deploy_real_robotref_160.py`

They are included as research references for code review and provenance of the 160D adaptation. The 160D adapter retains the 177D runtime structure and adapts the 154D robot-reference observation semantics.

## Safety notice

These files contain real-robot control paths and are **not a runnable public deployment package**. In particular, the archived 160D file has a dry-run branch but currently sets `dry_run = False` at module scope. Do not execute any file in this directory on hardware. The repository intentionally excludes its required runtime configuration, network setup, robot credentials, models, and motion assets.
