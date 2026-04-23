"""Height-scan sources. The LeggedLab-wbc actor consumes 187 samples of
``scanner_z - hit_z - 0.5`` arranged in a 17×11 grid centered on the torso.

Today: ``ZerosHeightScan`` (policy sees flat ground). Clear stubs for a
camera-based scanner (real robot) and a sim-raycast scanner (sanity baseline).
"""
from __future__ import annotations

from typing import Protocol

import numpy as np


class HeightScanSource(Protocol):
    """Stateless producer of a 187-d height scan."""

    def read(self) -> np.ndarray: ...
    def close(self) -> None: ...


class ZerosHeightScan:
    """Feeds zeros. Correct for flat ground when ``scale_height_scan=1`` and
    the raycaster origin sits at ``torso_z`` with offset 0.5 — matches the
    training distribution for plane rollouts.
    """

    def __init__(self) -> None:
        self._scan = np.zeros(187, dtype=np.float32)

    def read(self) -> np.ndarray:
        return self._scan.copy()

    def close(self) -> None:
        return None


class PlaneConstHeightScan:
    """Emulates LeggedLab-wbc's 187-bin raycast on a flat ground plane: every
    bin returns ``torso_z - 0.5``. On plane terrain this is what the raycaster
    produces (ground is at ``z=0``, sensor origin at ``torso_z``, offset 0.5).

    ``read_torso_z`` must be set to a callable returning the current torso
    world-z (the bridge wires this to ``FKHelper``). If unset, falls back to
    a constant ``fallback_torso_z``.
    """

    def __init__(self, fallback_torso_z: float = 0.85) -> None:
        self._fallback = float(fallback_torso_z)
        self._torso_z_fn = None

    def set_torso_z_source(self, fn):
        self._torso_z_fn = fn

    def read(self) -> np.ndarray:
        tz = self._torso_z_fn() if self._torso_z_fn is not None else self._fallback
        return np.full(187, tz - 0.5, dtype=np.float32)

    def close(self) -> None:
        return None


class CameraHeightScan:  # pragma: no cover
    """Raycast against a camera depth image. Real-robot path. TODO."""

    def __init__(self) -> None:
        raise NotImplementedError(
            "CameraHeightScan TODO: depth image → 17×11 grid in torso-frame. "
            "See HeightScanCfg in LeggedLab-wbc sim2sim for the expected layout "
            "(size_x=1.6, size_y=1.0, res=0.1, yaw_only attachment)."
        )


class SimRaycastHeightScan:  # pragma: no cover
    """Reuse LeggedLab's MuJoCo raycaster but against a shadow model driven
    by odostate. Useful as a sim-only baseline if bridge results diverge from
    the offline ``wbc_sim2sim`` runner.  TODO.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("SimRaycastHeightScan TODO")
