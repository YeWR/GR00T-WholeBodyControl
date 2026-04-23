"""Training-distribution sample trajectories for the 5 finalized policies.

All trajectories are 20 s (= 1000 bridge steps at 50 Hz). Ranges follow
``legged_lab/envs/base/base_config.py`` (EE pos) +
``base_env_config.py`` (vel) + ``g1_cluster_config.py`` (height).

Trajectory names match the ``--profile`` CLI flag:
  * ``walk``       — ScriptedCommand, vel sweep, EE at home
  * ``stand``      — LerpScriptedCommand, 5 EE waypoints, vel=0
  * ``stand_h``    — LerpScriptedCommand, EE waypoints + height sweep 0.60-0.78
  * ``balance_h``  — LerpScriptedCommand, height sweep only, EE at home
"""
from __future__ import annotations

from .command_source import (
    Command,
    LerpScriptedCommand,
    ScriptedCommand,
    Waypoint,
)


# Conservative EE waypoints (matches LeggedLab's ``build_stand_ee_demo``).
_STAND_EE_WAYPOINTS = [
    ((0.25, 0.15, 0.10), (0.25, -0.15, 0.10)),
    ((0.30, 0.10, 0.20), (0.30, -0.10, 0.20)),
    ((0.25, 0.20, 0.05), (0.25, -0.20, 0.05)),
    ((0.20, 0.15, 0.25), (0.20, -0.15, 0.25)),
    ((0.25, 0.15, 0.10), (0.25, -0.15, 0.10)),  # home
]


def build_walk() -> ScriptedCommand:
    """Velocity sweep drawing from training range (vx∈[-0.6,1.0], vy∈[-0.5,0.5],
    wz∈[-1.57,1.57]). Step-function over 20 s."""
    def _cmd(**kw) -> Command:
        return Command(**kw)
    return ScriptedCommand([
        Waypoint(0.0,  _cmd()),                 # 0-2 s : stand
        Waypoint(2.0,  _cmd(vx=0.5)),           # 2-7 s : forward
        Waypoint(7.0,  _cmd(vx=-0.3)),          # 7-10s : backward
        Waypoint(10.0, _cmd(vy=0.3)),           # 10-13s: strafe left
        Waypoint(13.0, _cmd(wz=0.8)),           # 13-17s: turn left
        Waypoint(17.0, _cmd(vx=0.4, wz=-0.4)),  # 17-20s: forward + yaw
    ])


def build_stand() -> LerpScriptedCommand:
    """EE lerp through 5 waypoints; velocity held at 0, h=0.74."""
    seg = 5.0
    wps = []
    for i, (lp, rp) in enumerate(_STAND_EE_WAYPOINTS):
        wps.append(Waypoint(i * seg, Command(left_ee_pos=lp, right_ee_pos=rp)))
    return LerpScriptedCommand(wps)


def build_stand_h() -> LerpScriptedCommand:
    """EE lerp + base height sweep in 0.60-0.78 (inside trained range 0.4-0.78)."""
    heights = [0.74, 0.65, 0.78, 0.60, 0.74]
    seg = 5.0
    wps = []
    for i, ((lp, rp), h) in enumerate(zip(_STAND_EE_WAYPOINTS, heights)):
        wps.append(Waypoint(
            i * seg,
            Command(left_ee_pos=lp, right_ee_pos=rp, target_h=h),
        ))
    return LerpScriptedCommand(wps)


def build_balance_h() -> LerpScriptedCommand:
    """Pure height sweep; EE held at home (ignored by balance_h's 284-d obs)."""
    heights = [0.74, 0.55, 0.78, 0.60, 0.74]
    seg = 5.0
    wps = [Waypoint(i * seg, Command(target_h=h)) for i, h in enumerate(heights)]
    return LerpScriptedCommand(wps)


PROFILES = {
    "walk":      build_walk,
    "stand":     build_stand,
    "stand_h":   build_stand_h,
    "balance_h": build_balance_h,
}
