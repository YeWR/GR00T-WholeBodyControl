"""CLI entry point.

Usage (two terminals):

    # terminal 1 — sonic sim (publishes rt/lowstate, waits for rt/lowcmd)
    conda activate sonic
    python gear_sonic/scripts/run_sim_loop.py \\
        --wbc_config gear_sonic/utils/mujoco_sim/wbc_configs/g1_29dof_sonic_model12.yaml

    # terminal 2 — bridge
    conda activate sonic
    python -m wbc_sim2sim.bridge.run_bridge \\
        --ckpt /home/weirui/codes/LeggedLab-wbc/finalized_models/stand_manipulation.pt \\
        --steps 500

Both processes must use the same ``DOMAIN_ID`` and ``INTERFACE`` (``lo`` in sim,
real NIC name on hardware).
"""
from __future__ import annotations

import argparse

from .bridge import Bridge, BridgeCfg
from .command_source import Command, FixedCommand
from .height_scan import PlaneConstHeightScan


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
    p.add_argument("--vx", type=float, default=0.0)
    p.add_argument("--vy", type=float, default=0.0)
    p.add_argument("--wz", type=float, default=0.0)
    p.add_argument("--target-h", type=float, default=0.74)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = BridgeCfg(
        ckpt_path=args.ckpt,
        domain_id=args.domain_id,
        interface=args.interface,
        control_hz=args.control_hz,
        device=args.device,
    )
    command = FixedCommand(
        Command(vx=args.vx, vy=args.vy, wz=args.wz, target_h=args.target_h)
    )
    bridge = Bridge(cfg=cfg, command=command, height_scan=PlaneConstHeightScan())
    bridge.run(max_steps=args.steps)


if __name__ == "__main__":
    main()
