"""Support for the 284-d balance-family policies (no EE block).

LeggedLab-wbc's sim2sim hardcodes a 308-d obs layout (base 96 + EE 24 +
height_err 1 + height_scan 187). The ``balance_height*`` checkpoints were
trained without EE tracking, so their first LSTM layer has input size 284 =
308 − 24. We keep LeggedLab's obs_builder untouched and instead wrap
``build_actor_obs`` to optionally drop the EE block at the call site.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch


def peek_obs_dim(ckpt_path: str | Path) -> int:
    """Read the first LSTM layer's input size from an RSL-RL ckpt."""
    blob = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    w = blob["model_state_dict"]["memory_a.rnn.weight_ih_l0"]  # (4*hidden, obs_dim)
    return int(w.shape[1])


def install_obs_variant_patches(obs_dim: int) -> None:
    """Monkey-patch LeggedLab sim2sim's ObsCfg + obs_builder to match the
    checkpoint's obs_dim. Safe to call multiple times in the same process.
    """
    import sim2sim.config as ll_config
    import sim2sim.obs_builder as ll_obs
    from sim2sim.obs_builder import build_actor_obs as _orig_build

    if obs_dim == 308:
        # Default layout. Restore originals in case a prior 284 rollout
        # installed patches in the same process.
        if hasattr(ll_obs, "_orig_build_actor_obs"):
            ll_obs.build_actor_obs = ll_obs._orig_build_actor_obs
        ll_config.ObsCfg.total_dim = 308  # type: ignore[attr-defined]
        import sim2sim.policy as ll_policy
        if hasattr(ll_policy, "_orig_load_policy"):
            ll_policy.load_policy = ll_policy._orig_load_policy
        try:
            import scripts.run_mujoco as rm
            if hasattr(rm, "_orig_load_policy"):
                rm.load_policy = rm._orig_load_policy
            if hasattr(rm, "_orig_build_actor_obs"):
                rm.build_actor_obs = rm._orig_build_actor_obs
        except ImportError:
            pass
        return

    if obs_dim != 284:
        raise NotImplementedError(
            f"obs_dim={obs_dim} not supported. Known layouts: 308 (full),"
            f" 284 (no-EE balance family)."
        )

    # 284 = 96 base + 1 height_err + 187 scan.  Wrap build_actor_obs to strip
    # the 24-d EE block that _orig_build always emits.
    def build_balance_obs(state, cmd, prev_action, obs_cfg, scan_cfg):
        # Call original, then delete the EE slice [96:120).
        full = _orig_build(state, cmd, prev_action, obs_cfg, scan_cfg)
        assert full.shape[0] == 308, f"expected 308 from upstream, got {full.shape}"
        trimmed = np.concatenate([full[:96], full[120:]], axis=0)
        assert trimmed.shape[0] == 284
        return trimmed

    # Save originals before patching so we can restore for 308 later.
    if not hasattr(ll_obs, "_orig_build_actor_obs"):
        ll_obs._orig_build_actor_obs = ll_obs.build_actor_obs
    ll_config.ObsCfg.total_dim = 284  # type: ignore[attr-defined]
    ll_obs.build_actor_obs = build_balance_obs
    import scripts.run_mujoco as rm
    if not hasattr(rm, "_orig_build_actor_obs"):
        rm._orig_build_actor_obs = rm.build_actor_obs
    if not hasattr(rm, "_orig_load_policy"):
        rm._orig_load_policy = rm.load_policy
    rm.build_actor_obs = build_balance_obs

    import sim2sim.policy as ll_policy
    if not hasattr(ll_policy, "_orig_load_policy"):
        ll_policy._orig_load_policy = ll_policy.load_policy
    def load_policy_284(ckpt_path, **kw):
        kw.setdefault("obs_dim", 284)
        return ll_policy._orig_load_policy(ckpt_path, **kw)
    rm.load_policy = load_policy_284
    ll_policy.load_policy = load_policy_284  # so wbc_sim2sim.bridge's `_ll_policy.load_policy(...)` picks it up
