"""Opt-in auto-recorder for ProtoMotions inference.

Enabled only when this directory is prepended to PYTHONPATH.  It leaves the
ProtoMotions checkout untouched and records exactly one complete simple-test
rollout when a non-headless viewer is supplied by Xvfb.
"""
from __future__ import annotations

import os


if os.environ.get("G1PIPE_PROTO_AUTO_RECORD") == "1":
    from protomotions.agents.base_agent.agent import BaseAgent

    _original = BaseAgent.simple_test_policy

    def _recording_simple_test(self, *args, **kwargs):
        simulator = self.env.simulator
        if getattr(simulator, "headless", True):
            raise RuntimeError("ProtoMotions auto recording requires a virtual display; run through xvfb-run.")
        simulator._toggle_video_record()
        try:
            return _original(self, *args, **kwargs)
        finally:
            simulator._toggle_video_record()
            # The recorder finalizes its MP4 on the following render call.
            simulator.render()

    BaseAgent.simple_test_policy = _recording_simple_test
