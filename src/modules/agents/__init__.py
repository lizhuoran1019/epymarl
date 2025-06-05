from .rnn_agent import RNNAgent
from .rnn_agent_2 import RNNAgent2 
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent
from .cnn_rnn_agent import CNNRNNAgent

REGISTRY = {}
REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_2"] = RNNAgent2
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["cnn_rnn"] = CNNRNNAgent