"""CLI entry point for the DDS bridge.

Usage (two terminals):

    # terminal 1 — sonic sim (publishes rt/lowstate, waits for rt/lowcmd)
    conda activate sonic
    python gear_sonic/scripts/run_sim_loop.py \\
        --no-enable-onscreen --no-with-hands --interface lo

    # terminal 2 — bridge
    conda activate sonic
    python -m wbc_sim2sim.bridge.run_bridge \\
        --ckpt /home/weirui/codes/LeggedLab-wbc/finalized_models/stand_manipulation.pt \\
        --profile stand --steps 1000 --video /tmp/out.mp4

``--profile`` picks a training-distribution goal trajectory:
  walk | stand | stand_h | balance_h       (see trajectories.py)
Omit ``--profile`` for a fixed command assembled from --vx/--vy/--wz/--target-h.

Both processes must use the same ``DOMAIN_ID`` and ``INTERFACE`` (``lo`` in sim,
real NIC name on hardware).

``MUJOCO_GL=egl`` is exported before any mujoco import so the offscreen
shadow-render renderer has a valid OpenGL context.
"""
from __future__ import annotations

import argparse
import os as _os

# MUST be set BEFORE any mujoco import; the Renderer reads it during MjrContext.
_os.environ.setdefault("MUJOCO_GL", "egl")

from .bridge import Bridge, BridgeCfg
from .command_source import Command, CommandSource, FixedCommand, KeyboardCommand
from .height_scan import PlaneConstHeightScan
from .trajectories import PROFILES


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True, help="Path to LeggedLab-wbc finalized .pt")
    p.add_argument("--domain-id", type=int, default=0)
    p.add_argument("--interface", type=str, default="lo",
                   help="'lo' for sim, real NIC name on hardware")
    p.add_argument("--control-hz", type=float, default=50.0)
    p.add_argument("--device", type=str, default="cpu", choices=("cpu", "cuda"))
    p.add_argument("--steps", type=int, default=None,
                   help="Stop after this many policy steps (default: run until Ctrl-C)")
    p.add_argument("--profile", type=str, default=None, choices=list(PROFILES),
                   help="Scripted training-style goal trajectory.")
    p.add_argument("--keyboard", action="store_true",
                   help="Take commands from the keyboard (pynput). Overrides --profile.")
    # Only used if --profile is not set.
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--target-h", type=float, default=0.74)
    # Shadow-render video output.
    p.add_argument("--video", type=str, default=None,
                   help="Write an mp4 to this path (bridge-side MuJoCo renderer).")
    p.add_argument("--video-res", type=str, default="960x540",
                   help="'WxH' pixel size for the shadow-render video.")
    return p.parse_args()


def _build_command(args: argparse.Namespace) -> CommandSource:
    initial = Command(vx=args.vx, vy=args.vy, wz=args.wz, target_h=args.target_h)
    if args.keyboard:
        return KeyboardCommand(initial)
    if args.profile:
        return PROFILES[args.profile]()
    return FixedCommand(initial)


def _apply_obs_variant_from_ckpt(ckpt: str) -> None:
    """Install LeggedLab obs-builder + load_policy patches matching the ckpt's
    obs_dim (308 default, 284 for balance-family). Needs LeggedLab sim2sim on
    sys.path first — bridge.py triggers that on import."""
    import sys as _sys
    from pathlib import Path as _Path
    ll = _Path("/home/weirui/codes/LeggedLab-wbc/sim2sim")
    if str(ll) not in _sys.path:
        _sys.path.insert(0, str(ll))
    from ..obs_variants import install_obs_variant_patches, peek_obs_dim
    install_obs_variant_patches(peek_obs_dim(ckpt))


def main() -> None:
    args = parse_args()
    _apply_obs_variant_from_ckpt(args.ckpt)
    cfg = BridgeCfg(
        ckpt_path=args.ckpt,
        domain_id=args.domain_id,
        interface=args.interface,
        control_hz=args.control_hz,
        device=args.device,
        video_path=args.video,
        video_res=args.video_res,
    )
    bridge = Bridge(
        cfg=cfg,
        command=_build_command(args),
        height_scan=PlaneConstHeightScan(),
    )
    bridge.run(max_steps=args.steps)


if __name__ == "__main__":
    main()
