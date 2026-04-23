"""Mapping between Isaac Lab joint order (what the RL policy consumes/produces)
and the Unitree G1 SDK joint index (MJCF tree order, what ``rt/lowstate`` and
``rt/lowcmd`` use).

SDK order (from ``unitree_sdk2_python/example/g1/low_level/g1_low_level_example.py``):

    0-5   left leg   (hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll)
    6-11  right leg  (same)
    12    waist_yaw
    13    waist_roll
    14    waist_pitch
    15-21 left arm   (shoulder_pitch, shoulder_roll, shoulder_yaw, elbow,
                      wrist_roll, wrist_pitch, wrist_yaw)
    22-28 right arm  (same)

Isaac order (from ``LeggedLab-wbc/sim2sim/sim2sim/config.py::ISAAC_JOINT_ORDER``)
is the interleaved breadth-first variant the Isaac Lab articulation produces.
We derive the permutation by name match so future ordering drift is caught.
"""
from __future__ import annotations

import numpy as np


SDK_JOINT_NAMES: tuple[str, ...] = (
    # Left leg (0-5)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # Right leg (6-11)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # Waist (12-14)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # Left arm (15-21)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # Right arm (22-28)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
assert len(SDK_JOINT_NAMES) == 29


def _isaac_joint_names() -> tuple[str, ...]:
    # Local import because LeggedLab-wbc/sim2sim is only on sys.path after
    # runner.py has run. Callers should import that first via
    # wbc_sim2sim.runner._install_patches or manually.
    from sim2sim.config import ISAAC_JOINT_ORDER

    return ISAAC_JOINT_ORDER


def isaac_to_sdk_perm() -> np.ndarray:
    """Return ``p`` such that ``sdk_vec = isaac_vec[p]`` reorders an Isaac-order
    29-vector into SDK/MJCF tree order. Equivalently, ``p[i]`` is the Isaac
    index of the joint that the SDK places at slot ``i``."""
    isaac = _isaac_joint_names()
    isaac_index = {name: i for i, name in enumerate(isaac)}
    p = np.zeros(29, dtype=np.int64)
    for sdk_slot, name in enumerate(SDK_JOINT_NAMES):
        if name not in isaac_index:
            raise KeyError(f"{name} not in ISAAC_JOINT_ORDER — mapping broken")
        p[sdk_slot] = isaac_index[name]
    return p


def sdk_to_isaac_perm() -> np.ndarray:
    """Return ``q`` such that ``isaac_vec = sdk_vec[q]``."""
    p = isaac_to_sdk_perm()
    q = np.empty_like(p)
    q[p] = np.arange(29)
    return q
