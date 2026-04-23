"""Entry point. ``run()`` forwards its argv to ``scripts/run_mujoco.py`` from
LeggedLab-wbc after patching ``resolve_scene`` and ``_build_patched_mjcf`` so
the rollout lands on sonic's MJCF instead of the vendored one.
"""
from __future__ import annotations

import os
import sys

from .scene import DEFAULT_SCENE_XML, LEGGED_SIM2SIM_DIR, build_patched_mjcf


def _install_patches(scene_xml_path):
    """Monkey-patch LeggedLab sim2sim to use sonic scene + tolerant mesh loader."""
    if str(LEGGED_SIM2SIM_DIR) not in sys.path:
        sys.path.insert(0, str(LEGGED_SIM2SIM_DIR))

    import sim2sim.mujoco_runner as mr
    import scripts.run_mujoco as rm

    mr._build_patched_mjcf = build_patched_mjcf
    rm.resolve_scene = lambda _terrain: str(scene_xml_path)
    return rm


def run(scene_xml_path=None):
    """Invoke the LeggedLab sim2sim CLI with our patches installed.

    ``sys.argv`` is expected to be the usual ``run_mujoco.py`` arguments
    (``--ckpt``, ``--steps``, ``--profile``, ``--video``, etc.).
    """
    scene_xml_path = scene_xml_path or DEFAULT_SCENE_XML
    os.environ.setdefault("MUJOCO_GL", "egl")
    rm = _install_patches(scene_xml_path)
    rm.main()
