"""Command sources feed the WBC policy the bits it can't read from robot
state: velocity commands, EE targets, and the base-height goal.

Today: ``FixedCommand``. Stubs for ``KeyboardCommand`` and ``VRCommand``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol


@dataclass
class Command:
    """Everything the 308-d obs builder needs beyond robot state."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    left_ee_pos: tuple[float, float, float] = (0.25, 0.15, 0.10)
    right_ee_pos: tuple[float, float, float] = (0.25, -0.15, 0.10)
    left_ee_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    right_ee_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0)
    target_h: float = 0.74


class CommandSource(Protocol):
    """Any object that exposes a ``current()`` method returning a ``Command``."""

    def current(self) -> Command: ...
    def close(self) -> None: ...


@dataclass
class FixedCommand:
    """Hold a single ``Command`` for the whole rollout."""

    command: Command = field(default_factory=Command)

    def current(self) -> Command:
        return self.command

    def close(self) -> None:  # noqa: D401
        """Nothing to clean up."""
        return None


class KeyboardCommand:
    """WASD-style keyboard source. Reuses the pynput bindings that
    decoupled_wbc's demo uses: w/s=vx, a/d=vy, q/e=wz, 1/2=height,
    3..8=EE rpy. Runs a daemon Listener.

    Not implemented yet — ``wbc_sim2sim.bridge`` reserves the name so callers
    can import it as a stub.
    """

    def __init__(self, initial: Command | None = None) -> None:  # pragma: no cover
        raise NotImplementedError(
            "KeyboardCommand TODO: pynput Listener over Command mutations. "
            "See decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py"
        )


class VRCommand:
    """XRoboToolkit / Pico VR source. Stream EE targets + walk velocity from
    the headset. TODO: wire to ``gear_sonic/scripts/pico_manager_thread_server.py``.
    """

    def __init__(self) -> None:  # pragma: no cover
        raise NotImplementedError("VRCommand TODO: subscribe to pico manager ZMQ")
