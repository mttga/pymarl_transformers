REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent
from .noisy_agents import NoisyRNNAgent
from .n_transf_agent import TransformerAgent
from .n_transf_agent_smac import TransformerAgentSmac
from .updet_smac import UpdetSmac

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent
REGISTRY["n_transf"] = TransformerAgent
REGISTRY["n_transf_smac"] = TransformerAgentSmac
REGISTRY["updet_smac"] = UpdetSmac