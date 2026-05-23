"""Path constants + the MJCF patcher used by the sim2sim runner.

Why the patcher: the upstream ``_build_patched_mjcf`` inlines a single-level
``<include>`` and assumes ``meshes/`` is a flat directory of ``.STL`` files.
Sonic's ``meshes/`` has subdirectories (``images/``, ``meshes_2F85/``,
``old/``) and non-STL blobs, so we filter those out.
"""
from __future__ import annotations

import os
import re
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SCENE_XML = (
    REPO_ROOT / "gear_sonic" / "data" / "robots" / "g1" / "scene_29dof_sim2sim.xml"
)
# Resolve LeggedLab-wbc's sim2sim package. Override via $LEGGED_LAB_WBC_REPO
# (or $LEGGED_SIM2SIM_DIR for the path itself); fall back to ~/codes/LeggedLab-wbc
# which is where our install.md puts it.
_env_pkg = os.environ.get("LEGGED_SIM2SIM_DIR")
if _env_pkg:
    LEGGED_SIM2SIM_DIR = Path(_env_pkg).expanduser().resolve()
else:
    _ll_repo = os.environ.get(
        "LEGGED_LAB_WBC_REPO", str(Path.home() / "codes" / "LeggedLab-wbc")
    )
    LEGGED_SIM2SIM_DIR = Path(_ll_repo).expanduser().resolve() / "sim2sim"


def build_patched_mjcf(scene_xml_path: str | Path):
    """Inline sonic's robot MJCF and swap its motor actuators for the
    Isaac-gained position actuators the LeggedLab sim2sim runner expects.

    Delegates to the upstream helper for the actuator-block construction so we
    stay in lock-step with Isaac's ``KP``/``KD``/``TAU_MAX`` tables.
    """
    from sim2sim.config import ISAAC_JOINT_ORDER, KD, KP, TAU_MAX

    scene_path = Path(scene_xml_path).resolve()
    asset_dir = scene_path.parent
    scene_text = scene_path.read_text()

    inc_match = re.search(r'<include\s+file="([^"]+)"\s*/>', scene_text)
    if inc_match is None:
        raise RuntimeError(f"{scene_path}: no <include file=...> to inline")
    inc_path = (asset_dir / inc_match.group(1)).resolve()
    inc_text = inc_path.read_text()
    inner_match = re.search(r"<mujoco[^>]*>(.*)</mujoco>", inc_text, re.DOTALL)
    if inner_match is None:
        raise RuntimeError(f"{inc_path}: could not strip outer <mujoco> tag")
    inner = inner_match.group(1)

    lines = ["<actuator>"]
    for i, jname in enumerate(ISAAC_JOINT_ORDER):
        lines.append(
            f'    <position name="{jname}" joint="{jname}" '
            f'kp="{float(KP[i]):g}" kv="{float(KD[i]):g}" '
            f'forcerange="{-float(TAU_MAX[i]):g} {float(TAU_MAX[i]):g}"/>'
        )
    lines.append("  </actuator>")
    pos_block = "\n  ".join(lines)
    inner_new, n = re.subn(
        r"<actuator>.*?</actuator>", pos_block, inner, count=1, flags=re.DOTALL
    )
    if n == 0:
        raise RuntimeError(f"{inc_path}: no <actuator> block to replace")
    merged = scene_text.replace(inc_match.group(0), inner_new)

    assets: dict[str, bytes] = {}
    mesh_dir = asset_dir / "meshes"
    for fn in mesh_dir.iterdir():
        if fn.is_file() and fn.suffix.upper() == ".STL":
            assets[f"meshes/{fn.name}"] = fn.read_bytes()
    return merged, assets
