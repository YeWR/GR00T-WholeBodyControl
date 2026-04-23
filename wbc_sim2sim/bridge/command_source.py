"""Command sources feed the WBC policy the bits it can't read from robot
state: velocity commands, EE targets, and the base-height goal.

``FixedCommand``, ``ScriptedCommand`` (step-function), and
``LerpScriptedCommand`` (piecewise-linear EE lerp) are all live. Stubs
remain for ``KeyboardCommand`` / ``VRCommand``.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np


Vec3 = tuple[float, float, float]


@dataclass
class Command:
    """Everything the 308-d obs builder needs beyond robot state."""

    vx: float = 0.0
    vy: float = 0.0
    wz: float = 0.0
    left_ee_pos: Vec3 = (0.25, 0.15, 0.10)
    right_ee_pos: Vec3 = (0.25, -0.15, 0.10)
    left_ee_rpy: Vec3 = (0.0, 0.0, 0.0)
    right_ee_rpy: Vec3 = (0.0, 0.0, 0.0)
    target_h: float = 0.74


class CommandSource(Protocol):
    """Any object exposing ``current(t: float)`` returning a ``Command``.

    Fixed sources can ignore ``t``; time-varying ones consume it.
    """

    def current(self, t: float = 0.0) -> Command: ...
    def close(self) -> None: ...


@dataclass
class FixedCommand:
    """Hold a single ``Command`` for the whole rollout."""

    command: Command = field(default_factory=Command)

    def current(self, t: float = 0.0) -> Command:  # noqa: ARG002
        return self.command

    def close(self) -> None:  # noqa: D401
        """Nothing to clean up."""
        return None


@dataclass
class Waypoint:
    """``t_from``: absolute bridge time (s) at which this waypoint becomes active."""
    t_from: float
    cmd: Command


class ScriptedCommand:
    """Step-function of ``Command`` over time. Waypoints must start at t=0.

    Mirrors LeggedLab-wbc ``sim2sim.commands.ScriptedCommand`` semantics.
    """

    def __init__(self, waypoints: list[Waypoint]):
        if not waypoints:
            raise ValueError("ScriptedCommand needs at least one waypoint")
        self._wps = sorted(waypoints, key=lambda w: w.t_from)
        if self._wps[0].t_from > 0.0:
            raise ValueError("First waypoint must start at t=0")

    def current(self, t: float = 0.0) -> Command:
        active = self._wps[0]
        for wp in self._wps:
            if wp.t_from <= t:
                active = wp
            else:
                break
        return active.cmd

    def close(self) -> None:
        return None


class LerpScriptedCommand:
    """Piecewise-linear EE lerp between waypoints; vel/height step-held.

    Mirrors LeggedLab-wbc ``sim2sim.commands.LerpScriptedCommand`` —
    matches the training ``EECommandGenerator`` lerp mode.
    """

    def __init__(self, waypoints: list[Waypoint], lerp_rpy: bool = True):
        if len(waypoints) < 2:
            raise ValueError("LerpScriptedCommand needs >=2 waypoints")
        self._wps = sorted(waypoints, key=lambda w: w.t_from)
        if self._wps[0].t_from > 0.0:
            raise ValueError("First waypoint must start at t=0")
        self._lerp_rpy = lerp_rpy

    def current(self, t: float = 0.0) -> Command:
        i = 0
        for k, wp in enumerate(self._wps):
            if wp.t_from <= t:
                i = k
            else:
                break
        a = self._wps[i]
        if i + 1 >= len(self._wps):
            return a.cmd
        b = self._wps[i + 1]
        span = b.t_from - a.t_from
        alpha = 0.0 if span <= 1e-6 else float(np.clip((t - a.t_from) / span, 0.0, 1.0))

        def _mix(key):
            va = np.asarray(getattr(a.cmd, key), dtype=np.float32)
            vb = np.asarray(getattr(b.cmd, key), dtype=np.float32)
            out = (1.0 - alpha) * va + alpha * vb
            return tuple(float(x) for x in out)

        return Command(
            vx=a.cmd.vx, vy=a.cmd.vy, wz=a.cmd.wz,
            target_h=a.cmd.target_h,
            left_ee_pos=_mix("left_ee_pos"),
            right_ee_pos=_mix("right_ee_pos"),
            left_ee_rpy=_mix("left_ee_rpy") if self._lerp_rpy else a.cmd.left_ee_rpy,
            right_ee_rpy=_mix("right_ee_rpy") if self._lerp_rpy else a.cmd.right_ee_rpy,
        )

    def close(self) -> None:
        return None


class KeyboardCommand:
    """WASD-style keyboard source. Bindings match
    ``decoupled_wbc/sim2mujoco/scripts/run_mujoco_gear_wbc.py``::

        w/s         vx ± 0.1 m/s
        a/d         vy ± 0.1 m/s
        q/e         wz ± 0.1 rad/s
        1/2         target_h ± 0.05 m (down/up)
        3/4         EE roll  (both hands, shared)
        5/6         EE pitch
        7/8         EE yaw
        z           reset to initial Command

    Runs a pynput ``Listener`` on a daemon thread. If pynput isn't installed
    or there's no X display, falls back to a FixedCommand silently (with a
    printed warning) so headless sims still work.
    """

    def __init__(self, initial: Command | None = None) -> None:
        import threading

        self._cmd = initial if initial is not None else Command()
        self._lock = threading.Lock()
        self._listener = None
        try:
            from pynput import keyboard as _kb
        except Exception as exc:  # noqa: BLE001
            print(f"[KeyboardCommand] pynput unavailable ({exc}); degrading to FixedCommand")
            return

        def on_press(key):
            try:
                k = key.char
            except AttributeError:
                return
            with self._lock:
                c = self._cmd
                if k == "w":
                    c = _rep(c, vx=c.vx + 0.1)
                elif k == "s":
                    c = _rep(c, vx=c.vx - 0.1)
                elif k == "a":
                    c = _rep(c, vy=c.vy + 0.1)
                elif k == "d":
                    c = _rep(c, vy=c.vy - 0.1)
                elif k == "q":
                    c = _rep(c, wz=c.wz + 0.1)
                elif k == "e":
                    c = _rep(c, wz=c.wz - 0.1)
                elif k == "1":
                    c = _rep(c, target_h=max(0.40, c.target_h - 0.05))
                elif k == "2":
                    c = _rep(c, target_h=min(0.80, c.target_h + 0.05))
                elif k in ("3", "4"):
                    d = 0.2 if k == "3" else -0.2
                    c = _rep(c,
                             left_ee_rpy=tuple(a + (d if i == 0 else 0.0)
                                               for i, a in enumerate(c.left_ee_rpy)),
                             right_ee_rpy=tuple(a + (d if i == 0 else 0.0)
                                                for i, a in enumerate(c.right_ee_rpy)))
                elif k in ("5", "6"):
                    d = 0.2 if k == "5" else -0.2
                    c = _rep(c,
                             left_ee_rpy=tuple(a + (d if i == 1 else 0.0)
                                               for i, a in enumerate(c.left_ee_rpy)),
                             right_ee_rpy=tuple(a + (d if i == 1 else 0.0)
                                                for i, a in enumerate(c.right_ee_rpy)))
                elif k in ("7", "8"):
                    d = 0.2 if k == "7" else -0.2
                    c = _rep(c,
                             left_ee_rpy=tuple(a + (d if i == 2 else 0.0)
                                               for i, a in enumerate(c.left_ee_rpy)),
                             right_ee_rpy=tuple(a + (d if i == 2 else 0.0)
                                                for i, a in enumerate(c.right_ee_rpy)))
                elif k == "z":
                    c = initial if initial is not None else Command()
                else:
                    return
                self._cmd = c
                print(f"[key] cmd: vx={c.vx:+.2f} vy={c.vy:+.2f} wz={c.wz:+.2f} "
                      f"h={c.target_h:.2f}  L_rpy={_fmt(c.left_ee_rpy)} "
                      f"R_rpy={_fmt(c.right_ee_rpy)}")

        try:
            self._listener = _kb.Listener(on_press=on_press)
            self._listener.daemon = True
            self._listener.start()
        except Exception as exc:  # noqa: BLE001
            print(f"[KeyboardCommand] listener failed ({exc}); degrading to FixedCommand")
            self._listener = None

    def current(self, t: float = 0.0) -> Command:  # noqa: ARG002
        with self._lock:
            return self._cmd

    def close(self) -> None:
        if self._listener is not None:
            try:
                self._listener.stop()
            except Exception:  # noqa: BLE001
                pass
            self._listener = None


def _rep(c: Command, **kw) -> Command:
    from dataclasses import replace
    return replace(c, **kw)


def _fmt(v: Vec3) -> str:
    return f"[{v[0]:+.2f},{v[1]:+.2f},{v[2]:+.2f}]"


class VRCommand:
    """XRoboToolkit / Pico VR source. Stream EE targets + walk velocity from
    the headset. TODO: wire to ``gear_sonic/scripts/pico_manager_thread_server.py``.
    """

    def __init__(self) -> None:  # pragma: no cover
        raise NotImplementedError("VRCommand TODO: subscribe to pico manager ZMQ")
