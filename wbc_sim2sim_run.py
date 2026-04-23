"""Backwards-compat shim. Prefer ``python -m wbc_sim2sim`` going forward."""
from wbc_sim2sim.runner import run

if __name__ == "__main__":
    run()
