"""Optional native ProtoMotions policy recording, isolated from its checkout."""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import argparse
from datetime import datetime, timezone
from pathlib import Path


PROTO_ROOT = Path('/home/unitree/projects/ProtoMotions')
ISAACLAB_PYTHON = Path('/home/unitree/miniconda3/envs/isaaclab/bin/python')
INFERENCE = PROTO_ROOT / 'protomotions' / 'inference_agent.py'
# This is intentionally the verified March.md SMPL tracker checkpoint.  It
# consumes the MotionLib PT before PyRoki retargeting; never feed it G1 PT.
CHECKPOINT = PROTO_ROOT / 'data' / 'pretrained_models' / 'motion_tracker' / 'smpl' / 'last.ckpt'
HOOK_ROOT = Path(__file__).with_name('proto_record_hook')


def _write(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


def render_native_protomotions(motion: Path, run_dir: Path) -> None:
    """Record the verified SMPL Motion Tracker rollout through the desktop.

    This is evidence for the ProtoMotions policy itself.  It is deliberately
    optional: a missing desktop stack must never invalidate an already checked
    PyRoki/WBT motion or prevent manual approval for training.
    """
    stage_dir = run_dir / '06_protomotions_native_preview'
    stage_file = stage_dir / 'stage.json'
    output = stage_dir / 'protomotions_smpl_tracker.mp4'
    command = [
        str(ISAACLAB_PYTHON), str(INFERENCE),
        '--checkpoint', str(CHECKPOINT), '--motion-file', str(motion),
        '--simulator', 'isaaclab', '--num-envs', '1',
    ]
    record = {
        'stage': 'protomotions_native_policy_preview',
        'created_at': datetime.now(timezone.utc).isoformat(),
        'motion': str(motion),
        'checkpoint': str(CHECKPOINT),
        'command': command,
        'kind': 'native_protomotions_smpl_motion_tracker',
        'training_started': False,
        'robot_access': 'none',
        'status': 'RUNNING',
    }
    stage_dir.mkdir(parents=True, exist_ok=True)
    desktop_display = os.environ.get('DISPLAY')
    # Use the user's NoMachine X server, not the inaccessible GDM :1 screen.
    if not desktop_display and Path('/tmp/.X11-unix/X1002').exists():
        desktop_display = ':1002'
    if not desktop_display:
        record.update({
            'status': 'WARN',
            'message': 'Native ProtoMotions MP4 needs a local desktop display. The G1 reference and WBT videos are still available.',
        })
        _write(stage_file, record)
        return
    if not ISAACLAB_PYTHON.is_file() or not CHECKPOINT.is_file():
        record.update({'status': 'WARN', 'message': 'IsaacLab Python or the G1 tracker checkpoint is unavailable.'})
        _write(stage_file, record)
        return
    started = datetime.now().timestamp()
    env = os.environ.copy()
    env['DISPLAY'] = desktop_display
    desktop_auth = Path('/home/unitree/.Xauthority')
    if desktop_auth.is_file():
        env['XAUTHORITY'] = str(desktop_auth)
    env['G1PIPE_PROTO_AUTO_RECORD'] = '1'
    env['PYTHONPATH'] = f'{HOOK_ROOT}:{env.get("PYTHONPATH", "")}'
    _write(stage_file, record)
    try:
        with (stage_dir / 'preview.log').open('w', encoding='utf-8') as log:
            completed = subprocess.run(command, cwd=PROTO_ROOT, stdout=log, stderr=subprocess.STDOUT, env=env, timeout=240)
    except subprocess.TimeoutExpired:
        record.update({'status': 'WARN', 'message': 'ProtoMotions native preview timed out after 4 minutes and was stopped.'})
        _write(stage_file, record)
        return
    candidates = sorted((PROTO_ROOT / 'output' / 'renderings').glob('*.mp4'), key=lambda path: path.stat().st_mtime)
    candidates = [path for path in candidates if path.stat().st_mtime >= started - 2]
    if completed.returncode == 0 and candidates:
        shutil.copy2(candidates[-1], output)
        record.update({'status': 'COMPLETE', 'video': str(output), 'returncode': 0})
    else:
        record.update({'status': 'WARN', 'returncode': completed.returncode, 'message': 'ProtoMotions native preview did not produce an MP4; inspect preview.log.'})
    _write(stage_file, record)


def main() -> None:
    parser = argparse.ArgumentParser(description='Record an optional native ProtoMotions G1 rollout.')
    parser.add_argument('--motion', type=Path, required=True)
    parser.add_argument('--run-dir', type=Path, required=True)
    args = parser.parse_args()
    render_native_protomotions(args.motion, args.run_dir)


if __name__ == '__main__':
    main()
