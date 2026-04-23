"""Thread-safe cache of the latest LowState / OdoState / IMU samples received
from Unitree SDK2 channels. The subscriber callbacks run on SDK threads; the
policy loop reads a consistent snapshot under a lock.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
import time

import numpy as np


@dataclass
class RobotStateSnapshot:
    joint_pos_sdk: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    joint_vel_sdk: np.ndarray = field(default_factory=lambda: np.zeros(29, dtype=np.float32))
    # IMU: body-frame ang vel (gyro), base quaternion (wxyz, world → body).
    ang_vel_body: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    quat_wxyz: np.ndarray = field(default_factory=lambda: np.array([1, 0, 0, 0], dtype=np.float32))
    # Odometry (sim only — not available on the physical robot without extra infra).
    root_pos_world: np.ndarray = field(default_factory=lambda: np.zeros(3, dtype=np.float32))
    has_odo: bool = False
    last_lowstate_tick: int = 0  # ms, monotonic on Unitree SDK side
    last_update_monotonic: float = 0.0


class StateCache:
    def __init__(self) -> None:
        self._lock = Lock()
        self._snap = RobotStateSnapshot()

    def update_lowstate(self, msg) -> None:
        """Called by the ``rt/lowstate`` subscriber thread."""
        n = 29
        q = np.fromiter(
            (msg.motor_state[i].q for i in range(n)), dtype=np.float32, count=n
        )
        dq = np.fromiter(
            (msg.motor_state[i].dq for i in range(n)), dtype=np.float32, count=n
        )
        quat = np.asarray(msg.imu_state.quaternion, dtype=np.float32)  # wxyz
        gyro = np.asarray(msg.imu_state.gyroscope, dtype=np.float32)  # body frame
        with self._lock:
            self._snap.joint_pos_sdk[:] = q
            self._snap.joint_vel_sdk[:] = dq
            self._snap.quat_wxyz[:] = quat
            self._snap.ang_vel_body[:] = gyro
            self._snap.last_lowstate_tick = int(msg.tick)
            self._snap.last_update_monotonic = time.monotonic()

    def update_odostate(self, msg) -> None:
        """Called by the ``rt/odostate`` subscriber thread (sim only)."""
        pos = np.asarray(msg.position, dtype=np.float32)
        with self._lock:
            self._snap.root_pos_world[:] = pos
            self._snap.has_odo = True

    def snapshot(self) -> RobotStateSnapshot:
        """Return a deep-ish copy of the cache (arrays copied, scalars snapshot)."""
        with self._lock:
            snap = RobotStateSnapshot(
                joint_pos_sdk=self._snap.joint_pos_sdk.copy(),
                joint_vel_sdk=self._snap.joint_vel_sdk.copy(),
                ang_vel_body=self._snap.ang_vel_body.copy(),
                quat_wxyz=self._snap.quat_wxyz.copy(),
                root_pos_world=self._snap.root_pos_world.copy(),
                has_odo=self._snap.has_odo,
                last_lowstate_tick=self._snap.last_lowstate_tick,
                last_update_monotonic=self._snap.last_update_monotonic,
            )
        return snap
