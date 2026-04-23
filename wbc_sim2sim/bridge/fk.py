"""Forward kinematics helper.

The 308-d policy obs includes each EE's body-frame xyz + RPY. On the robot
we don't have those tensors on the wire — we compute them ourselves from the
current joint positions. We reuse our wrapper MJCF (sonic's G1 29-DoF) as the
kinematic model so geometry stays consistent with sim2sim rollouts.

Conventions must match LeggedLab-wbc sim2sim byte-for-byte — especially the
``% (2π)`` wrap in ``euler_xyz_from_quat``. Without it the policy's obs
normalizer sees right-hand wrists ~5σ away from the training mean and
outputs huge action vectors (~125 vs the expected ~11).
"""
from __future__ import annotations

from pathlib import Path

import mujoco
import numpy as np

from ..scene import DEFAULT_SCENE_XML, build_patched_mjcf

# LeggedLab utils live on sys.path because runner.py / bridge.py inject it.
from sim2sim import utils as ll_utils  # noqa: E402


LEFT_EE_BODY = "left_wrist_pitch_link"
RIGHT_EE_BODY = "right_wrist_pitch_link"
PELVIS_BODY = "pelvis"
TORSO_BODY = "torso_link"


class FKHelper:
    """Tiny MuJoCo-backed FK. Not a real-time object: one instance per process."""

    def __init__(self, scene_xml: str | Path | None = None) -> None:
        xml, assets = build_patched_mjcf(scene_xml or DEFAULT_SCENE_XML)
        self.model = mujoco.MjModel.from_xml_string(xml, assets)
        self.data = mujoco.MjData(self.model)

        self._pelvis = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, PELVIS_BODY)
        self._torso = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, TORSO_BODY)
        self._left_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, LEFT_EE_BODY)
        self._right_ee = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_EE_BODY)
        assert self._pelvis >= 0 and self._left_ee >= 0 and self._right_ee >= 0 and self._torso >= 0

        from .joint_mapping import SDK_JOINT_NAMES

        self._sdk_qposadr = np.empty(29, dtype=np.int64)
        for i, jname in enumerate(SDK_JOINT_NAMES):
            jid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            if jid < 0:
                raise KeyError(f"joint {jname} not in MJCF")
            self._sdk_qposadr[i] = self.model.jnt_qposadr[jid]

    def update(self, joint_pos_sdk: np.ndarray, base_pos: np.ndarray, base_quat_wxyz: np.ndarray):
        """Compute EE pose in pelvis body frame + torso world z (for height scan).

        Matches LeggedLab-wbc ``mujoco_runner._ee_body``:
          ``ee_pos_b = R_pelvis^T (ee_pos_w - pelvis_pos)``
          ``ee_quat_b = conj(pelvis_quat) ⊗ ee_quat_w``
          ``ee_rpy_b = euler_xyz_from_quat(ee_quat_b)``  (wrapped to [0, 2π))

        Returns ``(left_pos_body, right_pos_body, left_rpy_body, right_rpy_body, torso_z_world)``.
        """
        self.data.qpos[0:3] = base_pos
        self.data.qpos[3:7] = base_quat_wxyz
        for i, adr in enumerate(self._sdk_qposadr):
            self.data.qpos[adr] = joint_pos_sdk[i]
        mujoco.mj_kinematics(self.model, self.data)

        pelvis_pos = self.data.xpos[self._pelvis].copy()
        pelvis_quat = self.data.xquat[self._pelvis].copy()  # wxyz
        pelvis_quat_conj = ll_utils.quat_conjugate(pelvis_quat)
        torso_z = float(self.data.xpos[self._torso][2])

        def _ee(body_id: int):
            pos_w = self.data.xpos[body_id].copy()
            quat_w = self.data.xquat[body_id].copy()
            pos_b = ll_utils.quat_rotate_inverse(pelvis_quat, pos_w - pelvis_pos)
            quat_b = ll_utils.quat_mul(pelvis_quat_conj, quat_w)
            rpy_b = ll_utils.euler_xyz_from_quat(quat_b)
            return pos_b.astype(np.float32), rpy_b.astype(np.float32)

        lp, lrpy = _ee(self._left_ee)
        rp, rrpy = _ee(self._right_ee)
        return lp, rp, lrpy, rrpy, torso_z
