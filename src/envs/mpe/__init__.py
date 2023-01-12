from . import core, scenario

from gym.envs.registration import register
from . import scenarios

import importlib
# Multiagent envs
# ----------------------------------------

_particles = {
    "multi_speaker_listener": "MultiSpeakerListener-v0",
    "simple_adversary": "SimpleAdversary-v0",
    "simple_crypto": "SimpleCrypto-v0",
    "simple_push": "SimplePush-v0",
    "simple_reference": "SimpleReference-v0",
    "simple_speaker_listener": "SimpleSpeakerListener-v0",
    "simple_spread": "SimpleSpread-v0",
    "simple_tag": "SimpleTag-v0",
    "simple_world_comm": "SimpleWorldComm-v0",
    "climbing_spread": "ClimbingSpread-v0",
}

for scenario_name, gymkey in _particles.items():
    scenario_module = importlib.import_module("envs.mpe.scenarios."+scenario_name)
    scenario_aux = scenario_module.Scenario()
    world = scenario_aux.make_world()

    # Registers multi-agent particle environments:
    register(
        gymkey,
        entry_point="envs.mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario_aux.reset_world,
            "reward_callback": scenario_aux.reward,
            "observation_callback": scenario_aux.observation,
            "world_info_callback": getattr(scenario_aux, "world_benchmark_data", None)
        },
    )

# Registers the custom double spread environment:

for N in range(2, 11, 2):
    scenario_name = "simple_doublespread"
    gymkey = f"DoubleSpread-{N}ag-v0"
    scenario_module = importlib.import_module("envs.mpe.scenarios."+scenario_name)
    scenario_aux = scenario_module.Scenario()
    world = scenario_aux.make_world(N)

    register(
        gymkey,
        entry_point="envs.mpe.environment:MultiAgentEnv",
        kwargs={
            "world": world,
            "reset_callback": scenario_aux.reset_world,
            "reward_callback": scenario_aux.reward,
            "observation_callback": scenario_aux.observation,
            "world_info_callback": getattr(scenario_aux, "world_benchmark_data", None)
        },
    )