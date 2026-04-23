"""Main bridge. Subscribes to ``rt/lowstate`` (+ optional ``rt/odostate``),
runs the LeggedLab-wbc policy at 50 Hz, publishes ``rt/lowcmd``.

Flow:
    lowstate → snapshot → (reorder SDK→Isaac) → obs_builder → policy.act
           → action (Isaac) → q_target = action*0.25 + DEFAULT_QPOS
           → reorder Isaac→SDK → publish LowCmd_ with per-joint Kp/Kd

Run against sonic's MuJoCo sim (``gear_sonic/scripts/run_sim_loop.py``) on the
same DOMAIN_ID. On the physical robot, swap INTERFACE away from ``lo``.
"""
from __future__ import annotations

import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

# LeggedLab sim2sim provides the obs builder + RSL-RL policy loader + gains.
# We cannot import from wbc_sim2sim.scene alone because some of these modules
# live in the external sim2sim package.
LEGGED_SIM2SIM_DIR = Path("/home/weirui/codes/LeggedLab-wbc/sim2sim")
if str(LEGGED_SIM2SIM_DIR) not in sys.path:
    sys.path.insert(0, str(LEGGED_SIM2SIM_DIR))

from sim2sim.config import DEFAULT_QPOS, KD, KP, HeightScanCfg, ObsCfg  # noqa: E402
from sim2sim.obs_builder import RobotState  # noqa: E402
# Import the modules (not the functions) so ``install_obs_variant_patches``
# can override ``sim2sim.{obs_builder.build_actor_obs, policy.load_policy}``
# after import and the bridge still picks up overrides at call time.
import sim2sim.obs_builder as _ll_obs  # noqa: E402
import sim2sim.policy as _ll_policy  # noqa: E402

from .command_source import Command, CommandSource, FixedCommand  # noqa: E402
from .fk import FKHelper  # noqa: E402
from .height_scan import HeightScanSource, ZerosHeightScan  # noqa: E402
from .joint_mapping import (  # noqa: E402
    SDK_JOINT_NAMES,
    isaac_to_sdk_perm,
    sdk_to_isaac_perm,
)
from .safety import SafetyCfg, SafetyMonitor  # noqa: E402
from .state_cache import StateCache  # noqa: E402


@dataclass
class BridgeCfg:
    ckpt_path: str
    domain_id: int = 0
    interface: str = "lo"
    control_hz: float = 50.0
    device: str = "cpu"
    max_wait_state_s: float = 30.0
    action_scale: float = 0.25
    clip_action: float = 100.0
    # How long a lowstate sample can be stale before we print a warning.
    stale_warn_s: float = 0.2
    # Drive the robot to DEFAULT_QPOS for this many control steps before
    # letting the policy take over. The ``_start_default_hold_thread`` path
    # already holds the robot at default during policy load; keep this small
    # so we hand off to the policy while the robot is still upright.
    warmup_steps: int = 5
    # Optional shadow-render video output — bridge loads a second MuJoCo
    # MjModel (the FKHelper's), syncs qpos from every lowstate / odostate,
    # renders at 50 fps via EGL offscreen.
    video_path: str | None = None
    video_res: str = "960x540"
    # Runtime safety: tilt / root-z triggered kp/kd taper.
    safety: SafetyCfg | None = None
    enable_safety: bool = True
    # Soft-start: linearly ramp kp/kd from 0 → full over this many seconds at
    # bridge startup. Matters a lot on real hardware where initial joint q can
    # differ from DEFAULT_QPOS by 1+ rad (folded stance), and a step-load of
    # kp=200 would demand hundreds of N·m on frame 1.
    soft_start_s: float = 2.0
    # On exit (normal, exception, or SIGINT/SIGTERM), publish this many
    # control cycles of a "damp" LowCmd — kp=0, kd=kd_damp, q=current q — so
    # motors settle smoothly instead of latching the last policy command.
    shutdown_damp_steps: int = 25
    shutdown_kd: float = 2.0


class Bridge:
    def __init__(
        self,
        cfg: BridgeCfg,
        command: CommandSource | None = None,
        height_scan: HeightScanSource | None = None,
    ) -> None:
        self.cfg = cfg
        self.command = command or FixedCommand()
        self.height_scan = height_scan or ZerosHeightScan()
        self.state = StateCache()
        self.isaac2sdk = isaac_to_sdk_perm()
        self.sdk2isaac = sdk_to_isaac_perm()
        self._last_torso_z = 0.85  # updated each obs build; PlaneConstHeightScan reads this

        # If the height_scan source is a PlaneConstHeightScan, wire its
        # torso_z source to our freshly FK'd torso world-z so scan mirrors
        # the raycaster's output for flat ground.
        from .height_scan import PlaneConstHeightScan
        if isinstance(self.height_scan, PlaneConstHeightScan):
            self.height_scan.set_torso_z_source(lambda: self._last_torso_z)

        self.obs_cfg = ObsCfg()
        self.scan_cfg = HeightScanCfg()

        # Lazily initialised in ``_init_channels`` because importing the SDK
        # has side effects (spins up a cyclonedds process).
        self._lowstate_sub = None
        self._odostate_sub = None
        self._lowcmd_pub = None
        self._lowcmd_msg = None
        self._crc = None
        # Filled by .run() after DDS is up but before the heavy policy load.
        self.fk = None
        self.policy = None
        self.hidden = None
        self.prev_action_isaac = None
        self._hold_stop = threading.Event()
        self._hold_thread: threading.Thread | None = None
        # Shadow-render video (lazy init in _init_video).
        self._video_writer = None
        self._renderer = None
        self._cam_id = -1
        # Safety monitor — None means disabled via cfg.enable_safety=False.
        self._safety: SafetyMonitor | None = (
            SafetyMonitor(self.cfg.safety) if self.cfg.enable_safety else None
        )
        # Per-step kp/kd scale (1.0 healthy, ↘ 0 during fall taper).
        self._kp_sdk = None  # set in _prime_lowcmd
        self._kd_sdk = None
        # Set by the signal handler — polled by the main policy loop.
        self._stop_requested = False
        # Wall-clock t0 for soft-start ramp (set in _start_default_hold_thread).
        self._start_t: float = 0.0

    # ------------------------------------------------------------------ SDK

    def _init_channels(self) -> None:
        from unitree_sdk2py.core.channel import (  # noqa: E402
            ChannelFactoryInitialize,
            ChannelPublisher,
            ChannelSubscriber,
        )
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_, LowState_
        from unitree_sdk2py.utils.crc import CRC

        ChannelFactoryInitialize(self.cfg.domain_id, self.cfg.interface)

        self._lowstate_sub = ChannelSubscriber("rt/lowstate", LowState_)
        self._lowstate_sub.Init(self._on_lowstate, 10)

        try:
            from unitree_sdk2py.idl.unitree_hg.msg.dds_ import OdoState_

            self._odostate_sub = ChannelSubscriber("rt/odostate", OdoState_)
            self._odostate_sub.Init(self._on_odostate, 10)
        except Exception as exc:  # noqa: BLE001
            # Not critical — real robot doesn't publish odostate. Print once.
            print(f"[bridge] skipping rt/odostate subscription: {exc}")

        self._lowcmd_pub = ChannelPublisher("rt/lowcmd", LowCmd_)
        self._lowcmd_pub.Init()
        self._lowcmd_msg = unitree_hg_msg_dds__LowCmd_()
        self._crc = CRC()
        self._prime_lowcmd()

    def _prime_lowcmd(self) -> None:
        """Lock Kp/Kd to the Isaac training gains in SDK order, zero everything
        else. We rewrite ``q`` each step; the rest stays constant."""
        from .joint_mapping import isaac_to_sdk_perm as _perm

        p = _perm()
        kp_sdk = KP[p].astype(np.float32)  # Wait: KP is Isaac-order. p reorders Isaac→SDK.
        # Actually: sdk_vec = isaac_vec[p] when p[i] = isaac_index_of_sdk_slot_i
        # (that's our convention). For Kp we want sdk-order values, so indexing
        # Isaac-order KP by p gives us the right thing.
        kd_sdk = KD[p].astype(np.float32)
        self._kp_sdk = kp_sdk
        self._kd_sdk = kd_sdk
        for i in range(29):
            m = self._lowcmd_msg.motor_cmd[i]
            m.mode = 1  # PMSM enabled
            m.q = 0.0
            m.dq = 0.0
            m.kp = float(kp_sdk[i])
            m.kd = float(kd_sdk[i])
            m.tau = 0.0

    def _on_lowstate(self, msg) -> None:
        self.state.update_lowstate(msg)

    def _on_odostate(self, msg) -> None:
        self.state.update_odostate(msg)

    # ------------------------------------------------------------------ obs

    def _build_obs(self, snap, command: Command) -> np.ndarray:
        # SDK → Isaac reorder for joint state.
        q_isaac = snap.joint_pos_sdk[self.sdk2isaac]
        dq_isaac = snap.joint_vel_sdk[self.sdk2isaac]
        # Projected gravity in body frame: rotate [0,0,-1] by conjugate of base quat.
        gproj = _rot_inverse_quat(snap.quat_wxyz, np.array([0.0, 0.0, -1.0], dtype=np.float32))
        # FK: EE in body frame. Feed world base pose (we use identity+odo_z for
        # purposes of computing relative vectors — only base orientation matters).
        lp, rp, lrpy, rrpy, torso_z = self.fk.update(
            snap.joint_pos_sdk, snap.root_pos_world, snap.quat_wxyz
        )
        self._last_torso_z = torso_z
        root_z = float(snap.root_pos_world[2]) if snap.has_odo else command.target_h

        robot_state = RobotState(
            ang_vel_b=snap.ang_vel_body.astype(np.float32),
            projected_gravity_b=gproj.astype(np.float32),
            joint_pos=q_isaac.astype(np.float32),
            joint_vel=dq_isaac.astype(np.float32),
            root_z=root_z,
            left_ee_pos_body=lp,
            right_ee_pos_body=rp,
            left_ee_rpy_body=lrpy,
            right_ee_rpy_body=rrpy,
            height_scan=self.height_scan.read(),
        )
        cmd_dict = {
            "vx": command.vx,
            "vy": command.vy,
            "wz": command.wz,
            "left_ee_pos": command.left_ee_pos,
            "right_ee_pos": command.right_ee_pos,
            "left_ee_rpy": command.left_ee_rpy,
            "right_ee_rpy": command.right_ee_rpy,
            "target_h": command.target_h,
        }
        return _ll_obs.build_actor_obs(
            robot_state, cmd_dict, self.prev_action_isaac, self.obs_cfg, self.scan_cfg
        )

    # ------------------------------------------------------------------ main

    def _wait_for_state(self) -> bool:
        start = time.monotonic()
        while time.monotonic() - start < self.cfg.max_wait_state_s:
            snap = self.state.snapshot()
            if snap.last_update_monotonic > 0:
                return True
            time.sleep(0.02)
        return False

    def _start_default_hold_thread(self) -> None:
        """Publish DEFAULT_QPOS at 50 Hz in a daemon thread. Used while the
        main thread is busy loading the policy so sonic's sim doesn't see
        multi-second command outages and let the robot flop to the floor.

        Kp/Kd are ramped from 0 → full over ``cfg.soft_start_s`` so motors
        don't yank the joints hard if initial q is far from ``DEFAULT_QPOS``
        (common on a real robot starting from a folded stance).
        """
        period = 1.0 / self.cfg.control_hz
        default_sdk = DEFAULT_QPOS[self.isaac2sdk].astype(np.float32)
        for i in range(29):
            self._lowcmd_msg.motor_cmd[i].q = float(default_sdk[i])

        self._start_t = time.monotonic()
        soft = max(1e-3, float(self.cfg.soft_start_s))

        def loop():
            while not self._hold_stop.is_set():
                ramp = min(1.0, (time.monotonic() - self._start_t) / soft)
                for i in range(29):
                    self._lowcmd_msg.motor_cmd[i].kp = float(self._kp_sdk[i] * ramp)
                    self._lowcmd_msg.motor_cmd[i].kd = float(self._kd_sdk[i] * ramp)
                self._lowcmd_msg.crc = self._crc.Crc(self._lowcmd_msg)
                self._lowcmd_pub.Write(self._lowcmd_msg)
                time.sleep(period)

        self._hold_thread = threading.Thread(target=loop, daemon=True)
        self._hold_thread.start()

    def _stop_default_hold_thread(self) -> None:
        self._hold_stop.set()
        if self._hold_thread is not None:
            self._hold_thread.join(timeout=1.0)

    def _init_video(self) -> None:
        if not self.cfg.video_path:
            return
        import os as _os
        _os.environ.setdefault("MUJOCO_GL", "egl")
        import imageio.v2 as imageio
        import mujoco
        from pathlib import Path as _Path

        w, h = (int(x) for x in self.cfg.video_res.lower().split("x"))
        # The FKHelper's model was already loaded for kinematics; reuse it.
        model = self.fk.model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), w)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), h)
        self._renderer = mujoco.Renderer(model, height=h, width=w)
        _Path(self.cfg.video_path).parent.mkdir(parents=True, exist_ok=True)
        self._video_writer = imageio.get_writer(
            self.cfg.video_path,
            fps=int(self.cfg.control_hz),
            codec="libx264", quality=8, macro_block_size=1,
        )

    def _render_frame(self) -> None:
        """Render one frame of the FK-side MuJoCo (already has current qpos from
        the latest _build_obs call)."""
        if self._renderer is None or self._video_writer is None:
            return
        import mujoco
        cam = mujoco.MjvCamera()
        mujoco.mjv_defaultFreeCamera(self.fk.model, cam)
        cam.lookat[:] = self.fk.data.xpos[self.fk._pelvis]
        cam.distance = 3.0
        cam.azimuth = 135.0
        cam.elevation = -20.0
        self._renderer.update_scene(self.fk.data, camera=cam)
        self._video_writer.append_data(self._renderer.render())

    def _close_video(self) -> None:
        if self._video_writer is not None:
            self._video_writer.close()
            print(f"[bridge] saved video to {self.cfg.video_path}")
            self._video_writer = None

    def _shutdown_damp(self) -> None:
        """Publish a few cycles of kp=0, kd=shutdown_kd, q=current_q so the
        robot ends in a low-torque follow mode instead of latching the last
        policy command. Best-effort: swallows exceptions so we always exit.
        """
        if self._lowcmd_msg is None or self._lowcmd_pub is None:
            return
        try:
            period = 1.0 / self.cfg.control_hz
            snap = self.state.snapshot()
            q_now = snap.joint_pos_sdk.astype(np.float32).copy()
            kd_damp = float(self.cfg.shutdown_kd)
            for i in range(29):
                m = self._lowcmd_msg.motor_cmd[i]
                m.q = float(q_now[i])
                m.dq = 0.0
                m.kp = 0.0
                m.kd = kd_damp
                m.tau = 0.0
            for _ in range(int(self.cfg.shutdown_damp_steps)):
                self._lowcmd_msg.crc = self._crc.Crc(self._lowcmd_msg)
                self._lowcmd_pub.Write(self._lowcmd_msg)
                time.sleep(period)
            print(f"[bridge] shutdown damp sent ({self.cfg.shutdown_damp_steps} cycles, "
                  f"kd={kd_damp})")
        except Exception as exc:  # noqa: BLE001
            print(f"[bridge] shutdown damp failed: {exc}")

    def _install_signal_handlers(self) -> None:
        """Route SIGINT / SIGTERM through a flag so the policy loop exits
        cleanly and ``finally`` (shutdown damp) runs. Without this, SIGTERM
        kills Python mid-publish and the robot may latch the last command.
        """
        import signal

        def handler(signum, _frame):  # noqa: ARG001
            print(f"[bridge] signal {signum} received, shutting down")
            self._stop_requested = True

        for sig in (signal.SIGINT, signal.SIGTERM):
            try:
                signal.signal(sig, handler)
            except (ValueError, OSError):
                # Not main thread — best-effort.
                pass

    def run(self, max_steps: int | None = None) -> None:
        # 0. Catch Ctrl-C / SIGTERM so finally (shutdown damp) actually runs.
        self._install_signal_handlers()
        # 1. Bring up DDS channels ASAP so we can start publishing a holding
        #    command within ~100 ms of sim start — before gravity collapses
        #    the robot from qpos0.
        self._init_channels()
        self._start_default_hold_thread()
        # 2. Heavy imports: load the RSL-RL policy (MuJoCo + torch, ~2 s on cpu,
        #    longer first time on cuda). The hold thread keeps publishing.
        self.fk = FKHelper()
        self.policy = _ll_policy.load_policy(self.cfg.ckpt_path, device=self.cfg.device)
        self.hidden = self.policy.reset(batch=1)
        self.prev_action_isaac = np.zeros(self.policy.io.action_dim, dtype=np.float32)
        # 3. Sync: ensure we actually got lowstate from the sim.
        if not self._wait_for_state():
            raise RuntimeError(
                f"no rt/lowstate received in {self.cfg.max_wait_state_s}s. "
                "Is the sonic sim running on the same DOMAIN_ID / INTERFACE?"
            )
        print("[bridge] first lowstate received. Starting policy loop.")
        # 4. Hand control to the main loop below.
        self._stop_default_hold_thread()
        self._init_video()

        period = 1.0 / self.cfg.control_hz
        default_sdk = DEFAULT_QPOS[self.isaac2sdk].astype(np.float32)

        # Warmup: hold at Isaac default for a few seconds so the sim settles
        # from qpos0 + gravity drift into a pose the policy was trained on.
        for w in range(self.cfg.warmup_steps):
            for i in range(29):
                self._lowcmd_msg.motor_cmd[i].q = float(default_sdk[i])
            self._lowcmd_msg.crc = self._crc.Crc(self._lowcmd_msg)
            self._lowcmd_pub.Write(self._lowcmd_msg)
            if w % 50 == 0:
                snap = self.state.snapshot()
                rz = snap.root_pos_world[2] if snap.has_odo else float("nan")
                print(f"[bridge] warmup step={w:3d} root_z={rz:.3f}")
            time.sleep(period)

        step = 0
        self._t_elapsed = 0.0
        next_tick = time.monotonic()
        try:
            while (max_steps is None or step < max_steps) and not self._stop_requested:
                next_tick += period
                snap = self.state.snapshot()
                age = time.monotonic() - snap.last_update_monotonic
                if age > self.cfg.stale_warn_s:
                    print(f"[bridge] warning: lowstate {age*1000:.0f} ms stale")

                cmd = self.command.current(self._t_elapsed)
                obs = self._build_obs(snap, cmd)
                action_isaac, self.hidden = self.policy.act(obs, self.hidden)
                np.clip(action_isaac, -self.cfg.clip_action, self.cfg.clip_action, out=action_isaac)
                q_target_isaac = action_isaac * self.cfg.action_scale + DEFAULT_QPOS
                q_target_sdk = q_target_isaac[self.isaac2sdk]

                # Gain scale = soft-start ramp × safety taper. Both default 1.0
                # (no effect); either can clamp kp/kd down.
                soft = max(1e-3, float(self.cfg.soft_start_s))
                soft_ramp = min(1.0, (time.monotonic() - self._start_t) / soft)
                safety_scale = 1.0
                if self._safety is not None:
                    root_z = float(snap.root_pos_world[2]) if snap.has_odo else float("nan")
                    self._safety.update(step, root_z, float(snap.quat_wxyz[0]), snap.has_odo)
                    safety_scale = self._safety.gain_scale(step)
                scale = soft_ramp * safety_scale

                for i in range(29):
                    m = self._lowcmd_msg.motor_cmd[i]
                    m.q = float(q_target_sdk[i])
                    if scale != 1.0:
                        m.kp = float(self._kp_sdk[i] * scale)
                        m.kd = float(self._kd_sdk[i] * scale)
                    else:
                        # restore full gains after taper/ramp expired
                        m.kp = float(self._kp_sdk[i])
                        m.kd = float(self._kd_sdk[i])
                self._lowcmd_msg.crc = self._crc.Crc(self._lowcmd_msg)
                self._lowcmd_pub.Write(self._lowcmd_msg)

                self.prev_action_isaac = action_isaac.copy()
                step += 1
                self._t_elapsed += period
                self._render_frame()
                if step % 50 == 0:  # once per sim second
                    root_z = snap.root_pos_world[2] if snap.has_odo else float("nan")
                    print(
                        f"[bridge] step={step:4d} root_z={root_z:.3f} "
                        f"|a|={float(np.linalg.norm(action_isaac)):.2f} "
                        f"w_quat={float(snap.quat_wxyz[0]):.3f}"
                    )

                # Sleep to next tick. If we drifted, just run the next step.
                sleep_for = next_tick - time.monotonic()
                if sleep_for > 0:
                    time.sleep(sleep_for)
                else:
                    next_tick = time.monotonic()
        finally:
            # Order matters: damp BEFORE closing DDS resources so the motors
            # see the damp command. Then clean up I/O.
            self._shutdown_damp()
            self.command.close()
            self.height_scan.close()
            self._close_video()


def _rot_inverse_quat(quat_wxyz: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate ``v`` by the inverse of the quaternion (world → body)."""
    w, x, y, z = quat_wxyz
    # Conjugate quat.
    q_conj = np.array([w, -x, -y, -z], dtype=np.float32)
    wc, xc, yc, zc = q_conj
    return np.array([
        v[0] * (wc * wc + xc * xc - yc * yc - zc * zc)
        + v[1] * 2 * (xc * yc - wc * zc)
        + v[2] * 2 * (xc * zc + wc * yc),
        v[0] * 2 * (xc * yc + wc * zc)
        + v[1] * (wc * wc - xc * xc + yc * yc - zc * zc)
        + v[2] * 2 * (yc * zc - wc * xc),
        v[0] * 2 * (xc * zc - wc * yc)
        + v[1] * 2 * (yc * zc + wc * xc)
        + v[2] * (wc * wc - xc * xc - yc * yc + zc * zc),
    ], dtype=np.float32)
