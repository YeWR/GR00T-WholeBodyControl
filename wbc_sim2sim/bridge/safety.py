"""Runtime safety monitor for the DDS bridge.

Detects a fall (quaternion tilt or root height below thresholds) and, once
triggered, linearly tapers the commanded ``kp`` / ``kd`` / ``tau`` down to
zero over ``taper_steps`` so motors release smoothly instead of fighting a
fallen pose. The taper is one-way — once armed, the bridge should be
stopped and restarted.

No auto-reset: on real hardware, "reset" is a human lifting the robot.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class SafetyCfg:
    # Trigger thresholds (checked every control step AFTER warmup).
    min_root_z: float = 0.35
    min_quat_w: float = 0.60          # below ~53° tilt from upright
    # How many control steps to spend tapering Kp/Kd from 1.0 → 0.
    taper_steps: int = 25             # 0.5 s at 50 Hz
    # Absolute torque clamp applied to the feed-forward ``tau`` term. Kp/Kd
    # are scaled by the taper ratio; ``tau`` is hard-clipped at all times.
    tau_clip: float = 200.0


class SafetyMonitor:
    """Tracks fall state + produces a per-step scale factor for kp/kd."""

    def __init__(self, cfg: SafetyCfg | None = None) -> None:
        self.cfg = cfg or SafetyCfg()
        self._tripped_step: int | None = None  # None = healthy
        self._last_reason: str = ""

    def update(self, step: int, root_z: float, quat_w: float, has_odo: bool) -> None:
        if self._tripped_step is not None:
            return
        # Only act on root_z when we actually have odostate; otherwise skip
        # that check (real robot won't publish it).
        if has_odo and np.isfinite(root_z) and root_z < self.cfg.min_root_z:
            self._tripped_step = step
            self._last_reason = f"root_z={root_z:.3f}<{self.cfg.min_root_z}"
            print(f"[safety] FALL @ step={step}  {self._last_reason} — tapering kp/kd")
            return
        if np.isfinite(quat_w) and quat_w < self.cfg.min_quat_w:
            self._tripped_step = step
            self._last_reason = f"quat_w={quat_w:.3f}<{self.cfg.min_quat_w}"
            print(f"[safety] FALL @ step={step}  {self._last_reason} — tapering kp/kd")

    @property
    def tripped(self) -> bool:
        return self._tripped_step is not None

    def gain_scale(self, step: int) -> float:
        """Return a 0..1 multiplier applied to kp / kd in the outgoing LowCmd."""
        if self._tripped_step is None:
            return 1.0
        elapsed = step - self._tripped_step
        if elapsed >= self.cfg.taper_steps:
            return 0.0
        return float(1.0 - elapsed / max(1, self.cfg.taper_steps))
