from .coma import COMACritic
from .centralV import CentralVCritic
from .ac import ACCritic
REGISTRY = {}

REGISTRY["coma_critic"] = COMACritic
REGISTRY["cv_critic"] = CentralVCritic
REGISTRY["ac_critic"] = ACCritic