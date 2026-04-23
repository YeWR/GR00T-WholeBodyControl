"""wbc_sim2sim — sonic-side harness for running LeggedLab-wbc finalized
policies in MuJoCo using sonic's G1 MJCF.

Current state: thin shim over ``LeggedLab-wbc/sim2sim``. Adds
  - sonic-rooted MJCF (``gear_sonic/data/robots/g1/scene_29dof_sim2sim.xml``)
  - mesh-loader patch that tolerates the subdirs in sonic's ``meshes/``

See ``README.md`` for the full architecture sketch and the (substantial) TODO
list that stands between this and sim2real.
"""
from .runner import run
from .scene import DEFAULT_SCENE_XML, LEGGED_SIM2SIM_DIR

__all__ = ["run", "DEFAULT_SCENE_XML", "LEGGED_SIM2SIM_DIR"]
