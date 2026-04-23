# wbc_sim2sim

Sonic-side harness for running LeggedLab-wbc finalized RL policies in MuJoCo
against sonic's G1 29-DoF MJCF. This is the Python-layer scaffold that sim2real
work will eventually slot into.

## What this does today

- Loads the sonic robot MJCF (`gear_sonic/data/robots/g1/scene_29dof_sim2sim.xml`,
  a flat wrapper over `g1_29dof_old.xml` that adds a 200 Hz `<option>` and the
  phantom `raycast_floor` in group 5 that the height scanner needs).
- Reuses `LeggedLab-wbc/sim2sim`'s obs builder (308-d), PD loop (implicit at
  50 Hz policy / 200 Hz sim), and checkpoint loader (RSL-RL `ActorCriticRecurrent`,
  LSTM + MLP + obs normalizer).
- Patches the upstream mesh loader so it tolerates sonic's `meshes/` subdirs.

## Usage

```bash
conda activate sonic

# From the repo root:
python -m wbc_sim2sim \
    --ckpt /home/weirui/codes/LeggedLab-wbc/finalized_models/walk_manipulation.pt \
    --profile walk_demo --steps 1000 \
    --device cuda --headless \
    --video /tmp/wbc_walk.mp4
```

All LeggedLab `run_mujoco.py` flags pass through unchanged (`--push`,
`--terrain bumps`, `--vx 0.5`, `--record-dir`, `--no-height-scan`, ...).

## Architecture (current)

```
┌────────────────────────────────────────────────────────────────┐
│   wbc_sim2sim (this package)                                   │
│   ┌─────────────┐   ┌──────────────┐   ┌─────────────────────┐ │
│   │  __main__   │──▶│  runner.run  │──▶│  scene.build_...    │ │
│   └─────────────┘   └──────┬───────┘   └─────────────────────┘ │
│                            │                                   │
│                            ▼                                   │
│             ┌──────────────────────────┐                       │
│             │ monkey-patch LeggedLab   │                       │
│             │ sim2sim: scene_xml +     │                       │
│             │ mesh loader              │                       │
│             └──────────┬───────────────┘                       │
└────────────────────────┼───────────────────────────────────────┘
                         ▼
           ┌──────────────────────────────────┐
           │  LeggedLab-wbc/sim2sim           │
           │  (imported via sys.path)         │
           │                                  │
           │  - scripts/run_mujoco.py (CLI)   │
           │  - obs_builder  (308-d)          │
           │  - policy       (RSL-RL loader)  │
           │  - mujoco_runner (PD + scan)     │
           └──────────────────────────────────┘
```

## What this does *not* do yet (i.e. the sim2real gap)

The goal is to replace sonic's deploy policy with the LeggedLab-wbc policy.
Sonic's deploy is a C++ / TensorRT pipeline using a 436-d observation with:
- 64-d encoder token (separate ONNX network that consumes motion reference data)
- 10-frame history of base ang vel / gravity / joint pos / joint vel / actions

The LeggedLab-wbc policy consumes 308-d **single-frame** observations that
include an **EE pose target block (24-d)** and a **187-d height scan**. These
obs spaces are disjoint — a direct "swap the .pt for an .onnx" deployment
does not exist.

Roadmap (rough):

1. **[DONE] sim2sim smoke test on sonic MJCF**
   — this package, option-1 path. Confirms kinematics + physics parity.

2. **[TODO] Python-side sim2real adapter**
   - Lift `obs_builder`/`policy`/`MujocoRunner` out of LeggedLab-wbc and into
     this package (stop depending on sys.path injection).
   - Wire a ZMQ/ROS command source so EE targets + vx/vy/wz are not hardcoded.
   - Keep the 308-d obs. Document which signals need to come from the real
     robot's state publisher (base quat, joint states, EE FK) and which need
     to come from an upstream task planner (EE targets, base-height goal).
   - Swap in a sonic-style MuJoCo simulator (or keep LeggedLab's — decide).

3. **[TODO] C++ deploy integration**
   - Author a new C++ inference node in `gear_sonic_deploy/src/g1/` that
     consumes 308-d obs, runs the WBC policy ONNX, emits 29-d position targets.
   - Export the `.pt` actor (LSTM + MLP + obs normalizer) to ONNX. LSTM hidden
     state has to be exposed as named I/O tensors.
   - Replace (or parallel-run alongside) sonic's encoder+decoder+planner chain.
   - Rebuild with `deploy.sh sim` to run loopback MuJoCo before touching real.

## Files

- `scene.py` — constants + MJCF patcher (inline include, Isaac-gain actuators,
  STL-only mesh loader).
- `runner.py` — `run()` that monkey-patches LeggedLab sim2sim and delegates
  to its CLI.
- `__main__.py` — `python -m wbc_sim2sim <same-args-as-run_mujoco.py>`.

## Known hazards

- `LEGGED_SIM2SIM_DIR` is hardcoded to
  `/home/weirui/codes/LeggedLab-wbc/sim2sim`. When the sim2sim code lands in
  this repo directly, delete the hardcode + the sys.path injection in
  `runner.py`.
- `meshes/` is read from `gear_sonic/data/robots/g1/meshes/`. If sonic ships a
  new robot revision that drops an STL the WBC policy's 29-DoF model depends
  on, you'll see a compile error from MjModel (not a quiet drift).
- Sonic's robot defaults (damping=0.05, frictionloss=0.2) are slightly
  different from LeggedLab's scene defaults (damping=0.1, no frictionloss).
  Rollouts match LeggedLab to 3 decimals on stand/walk plane, but if you push
  into long-horizon bumps/push regimes this may diverge.
