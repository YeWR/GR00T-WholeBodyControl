"""``python -m wbc_sim2sim --ckpt ... --steps ...`` — same args as
LeggedLab sim2sim's ``run_mujoco.py``."""
from .runner import run

if __name__ == "__main__":
    run()
