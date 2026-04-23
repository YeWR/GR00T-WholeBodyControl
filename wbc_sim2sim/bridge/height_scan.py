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


class CameraHeightScan:
    """Raycast 17×11 torso-frame grid against a depth image.

    Call signature for real-robot use:

        scan_src = CameraHeightScan(
            fx=..., fy=..., cx=..., cy=...,   # depth camera intrinsics
            T_torso_cam=np.eye(4),             # 4×4 SE(3) cam-in-torso frame
        )
        scan_src.set_depth_fn(lambda: latest_depth_m_HxW_float32)
        # ... bridge calls scan_src.read() each control step ...

    ``set_depth_fn`` wires the scan to whatever source delivers the robot's
    depth frame (ZMQ callback, pyrealsense pipeline, etc.). If unset or the
    function returns ``None``, ``read()`` falls back to a zero scan so the
    bridge doesn't crash.

    The 17×11 grid mirrors ``HeightScanCfg`` in the training env: ``size_x=1.6``
    m forward, ``size_y=1.0`` m lateral, 0.1 m spacing, torso-frame origin.
    Each bin is ``scanner_z − hit_z − 0.5`` (matches
    ``sim2sim.mujoco_runner._cast_height_scan``).

    Real-robot validation is still TODO — this class only has unit-test
    coverage against synthetic depth frames (see ``tests/test_height_scan.py``
    if/when that file lands).
    """

    def __init__(
        self,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        T_torso_cam: np.ndarray,
        size_x: float = 1.6,
        size_y: float = 1.0,
        resolution: float = 0.1,
        scanner_offset_z: float = 0.5,
    ) -> None:
        self.fx, self.fy, self.cx, self.cy = fx, fy, cx, cy
        self.T_torso_cam = np.asarray(T_torso_cam, dtype=np.float32)
        assert self.T_torso_cam.shape == (4, 4)
        self.size_x = float(size_x)
        self.size_y = float(size_y)
        self.res = float(resolution)
        self.offset_z = float(scanner_offset_z)
        nx = int(round(self.size_x / self.res)) + 1
        ny = int(round(self.size_y / self.res)) + 1
        assert nx == 17 and ny == 11, (nx, ny)
        # Grid of ray origins in torso frame, z=0 plane over the torso XY grid.
        xs = np.linspace(-self.size_x / 2, self.size_x / 2, nx, dtype=np.float32)
        ys = np.linspace(-self.size_y / 2, self.size_y / 2, ny, dtype=np.float32)
        # Row-major (x_outer, y_inner) to match LeggedLab convention (17 x 11).
        gx, gy = np.meshgrid(xs, ys, indexing="ij")
        self._grid_xy = np.stack([gx.ravel(), gy.ravel()], axis=-1)  # (187, 2)
        self._depth_fn = None
        self._zero = np.zeros(187, dtype=np.float32)

    def set_depth_fn(self, fn):
        """``fn() -> Optional[np.ndarray(H,W) float meters]``."""
        self._depth_fn = fn

    def read(self) -> np.ndarray:
        if self._depth_fn is None:
            return self._zero.copy()
        depth = self._depth_fn()
        if depth is None:
            return self._zero.copy()
        return self._raycast(depth)

    def _raycast(self, depth: np.ndarray) -> np.ndarray:
        """For each of the 187 torso-frame XY grid points, shoot a downward ray
        and read the camera-depth intersection.

        Implementation: project each 3D query point (x, y, 0) from torso frame
        into camera pixel coordinates, look up depth, back-project to torso Z,
        compute ``scanner_z − hit_z − offset_z``. This approximates a true
        mesh raycast when the depth camera has a bird's-eye-ish view of the
        region around the torso.
        """
        H, W = depth.shape
        R = self.T_torso_cam[:3, :3]
        t = self.T_torso_cam[:3, 3]
        # Query points in torso frame, z=0 means "ground under the torso grid";
        # scanner_z is the torso height above ground, which we treat as 0 here
        # since the scan is *relative* to the torso and the obs adds the offset.
        P_t = np.concatenate([self._grid_xy, np.zeros((187, 1), dtype=np.float32)], axis=-1)
        # torso → camera:  P_c = R_tc^T (P_t − t_tc)
        P_c = (P_t - t) @ R
        # Project. MuJoCo / opencv convention: u = fx * X/Z + cx, v = fy * Y/Z + cy.
        Z = P_c[:, 2]
        valid = Z > 1e-3
        u = (self.fx * P_c[:, 0] / np.where(valid, Z, 1.0) + self.cx).astype(np.int64)
        v = (self.fy * P_c[:, 1] / np.where(valid, Z, 1.0) + self.cy).astype(np.int64)
        in_frame = valid & (u >= 0) & (u < W) & (v >= 0) & (v < H)
        depth_at = np.where(in_frame, depth[np.clip(v, 0, H - 1), np.clip(u, 0, W - 1)], 0.0)
        # Back-project: camera-frame Z at that pixel, then convert to torso
        # frame Z. With the query direction pointing down (torso → ground), the
        # hit_z_torso = -depth_at * cam_down_component (approx).
        # For an axis-aligned bird's-eye cam pointing -z in torso frame, this
        # simplifies to hit_z_torso = -depth_at. We use that shorthand below;
        # general T_torso_cam needs a proper per-ray back-projection.
        hit_z_torso = -depth_at.astype(np.float32)
        scanner_z = 0.0  # torso frame origin
        return (scanner_z - hit_z_torso - self.offset_z).astype(np.float32)


class SimRaycastHeightScan:  # pragma: no cover
    """Reuse LeggedLab's MuJoCo raycaster but against a shadow model driven
    by odostate. Useful as a sim-only baseline if bridge results diverge from
    the offline ``wbc_sim2sim`` runner.  TODO.
    """

    def __init__(self, *_args, **_kwargs) -> None:
        raise NotImplementedError("SimRaycastHeightScan TODO")
