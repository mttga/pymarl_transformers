from functools import partial
import sys
import os
import warnings

from .mpe.mpe_wrapper import MPEWrapper
from .multiagentenv import MultiAgentEnv
try:
    from .starcraft import StarCraft2Env
    include_sc2 = True
except:
    warnings.warn("Impossible to import SMAC, verify your installation.")
    include_sc2 = False
from .matrix_game import OneStepMatrixGame

def env_fn(env, **kwargs) -> MultiAgentEnv:
    return env(**kwargs)

REGISTRY = {}
if include_sc2: # include starcraft only if correctly installed
    REGISTRY["sc2"] = partial(env_fn, env=StarCraft2Env)
REGISTRY["mpe"] = partial(env_fn, env=MPEWrapper)
REGISTRY["one_step_matrix_game"] = partial(env_fn, env=OneStepMatrixGame)

if sys.platform == "linux":
    os.environ.setdefault("SC2PATH", "~/StarCraftII")