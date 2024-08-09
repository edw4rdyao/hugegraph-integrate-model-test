"""
Jumping Knowledge Network (JKNet)

References
----------
Paper: https://arxiv.org/abs/1806.03536
DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/jknet
"""

import dgl.function as fn
import torch.nn as nn
import torch.nn.functional as F
from dgl.nn.pytorch import GraphConv, JumpingKnowledge


class JKNet(nn.Module):
    """
    Jumping Knowledge Network (JKNet) model.

    Parameters
    ----------
    n_in_feats : int
        Number of input features.
    n_hidden : int
        Number of hidden units.
    n_out_feats : int
        Number of output features.
    n_layers : int, optional
        Number of GNN layers, by default 1.
    mode : str, optional
        Jumping Knowledge mode ('cat', 'max', 'lstm'), by default "cat".
    dropout : float, optional
        Dropout rate, by default 0.0.
    """

    def __init__(self, n_in_feats, n_hidden, n_out_feats, n_layers=1, mode="cat", dropout=0.0):
        super(JKNet, self).__init__()
        self.mode = mode
        self.dropout = nn.Dropout(dropout)  # Dropout layer

        self.layers = nn.ModuleList()  # List of GraphConv layers
        self.layers.append(GraphConv(n_in_feats, n_hidden, activation=F.relu))  # Input layer
        for _ in range(n_layers):
            self.layers.append(GraphConv(n_hidden, n_hidden, activation=F.relu))  # Hidden layers

        if self.mode == "lstm":
            self.jump = JumpingKnowledge(mode, n_hidden, n_layers)  # Jumping Knowledge with LSTM
        else:
            self.jump = JumpingKnowledge(mode)  # Jumping Knowledge without LSTM

        if self.mode == "cat":
            n_hidden = n_hidden * (n_layers + 1)  # Adjust hidden size for concatenation mode

        self.output_layer = nn.Linear(n_hidden, n_out_feats)  # Output layer
        self.reset_params()

    def reset_params(self):
        for layer in self.layers:
            layer.reset_parameters()  # Reset parameters of GraphConv layers
        self.jump.reset_parameters()  # Reset parameters of Jumping Knowledge
        self.output_layer.reset_parameters()  # Reset parameters of output layer

    def forward(self, graph, feats):
        hidden_representations = []
        for layer in self.layers:
            feats = self.dropout(layer(graph, feats))  # Apply GraphConv and dropout
            hidden_representations.append(feats)  # Collect hidden representations

        if self.mode == "lstm":
            self.jump.lstm.flatten_parameters()  # Flatten LSTM parameters for efficiency
        # Apply Jumping Knowledge to aggregate hidden representations
        graph.ndata["h"] = self.jump(hidden_representations)
        # Message passing: copy node data to messages, then sum messages
        graph.update_all(fn.copy_u("h", "m"), fn.sum("m", "h"))
        h = self.output_layer(graph.ndata["h"])  # Apply the output layer
        return h  # Return the final node representations
