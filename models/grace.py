"""
GRACE (Graph Contrastive Learning)

References
----------
Papers: https://arxiv.org/abs/2006.04131
Author's code: https://github.com/CRIPAC-DIG/GRACE
DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/grace
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv


class GRACE(nn.Module):
    """
    GRACE model for graph representation learning via contrastive learning.

    Parameters
    ----------
    n_in_feats : int
        Number of input features.
    n_hidden : int
        Size of hidden layer.
    n_out_feats : int
        Size of output feature.
    n_layers : int
        Number of GNN layers.
    act_fn : nn.Module
        Activation function.
    temp : float
        Temperature parameter for contrastive loss.
    """

    def __init__(self, n_in_feats, n_hidden, n_out_feats, n_layers, act_fn, temp):
        super(GRACE, self).__init__()
        self.encoder = GCN(n_in_feats, n_hidden, act_fn, n_layers)  # Initialize the GCN encoder
        self.proj = MLP(n_hidden, n_out_feats)  # Initialize the MLP projector
        self.temp = temp  # Set the temperature for contrastive loss

    @staticmethod
    def sim(z1, z2):
        z1 = F.normalize(z1)  # Normalize the features
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())  # Compute cosine similarity

    def sim_loss(self, z1, z2):
        f = lambda x: torch.exp(x / self.temp)  # Exponential function scaled by temperature
        refl_sim = f(self.sim(z1, z1))  # Similarity within the same view
        between_sim = f(self.sim(z1, z2))  # Similarity between different views
        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()  # Denominator of contrastive loss
        loss = -torch.log(between_sim.diag() / x1)  # Numerator of contrastive loss
        return loss

    def loss(self, z1, z2):
        l1 = self.sim_loss(z1, z2)  # Calculate loss for the first projection
        l2 = self.sim_loss(z2, z1)  # Calculate loss for the second projection
        return (l1 + l2).mean() * 0.5  # Average the losses for symmetry

    def get_embedding(self, graph, feat):
        h = self.encoder(graph, feat)  # Get embeddings from the GCN
        return h.detach()  # Detach embeddings from the graph for evaluation

    def forward(self, graph, feat):
        h = self.encoder(graph, feat)  # Encode graph features using GCN
        z = self.proj(h)  # Project graph embeddings using MLP
        return z


class GCN(nn.Module):
    """
    Graph Convolutional Network (GCN) for node feature transformation.

    Parameters
    ----------
    n_in_feats : int
        Number of input features per node.
    n_out_feats : int
        Number of output features per node.
    act_fn : nn.Module
        Activation function.
    n_layers : int
        Number of GCN layers.
    """

    def __init__(self, n_in_feats, n_out_feats, act_fn, n_layers=2):
        super(GCN, self).__init__()
        assert n_layers >= 2, "Number of layers should be at least 2."
        self.n_layers = n_layers
        self.n_hidden = n_out_feats * 2  # Set hidden dimension as twice the output dimension
        self.input_layer = GraphConv(n_in_feats, self.n_hidden, activation=act_fn)  # Define the input layer
        self.hidden_layers = nn.ModuleList([
            GraphConv(self.n_hidden, self.n_hidden, activation=act_fn) for _ in range(n_layers - 2)
        ])  # Define the hidden layers
        self.output_layer = GraphConv(self.n_hidden, n_out_feats, activation=act_fn)  # Define the output layer

    def forward(self, graph, feat):
        feat = self.input_layer(graph, feat)  # Apply graph convolution to the input layer
        for hidden_layer in self.hidden_layers:
            feat = hidden_layer(graph, feat)  # Apply graph convolution to each hidden layer
        return self.output_layer(graph, feat)  # Apply graph convolution to the output layer


class MLP(nn.Module):
    """
    A simple Multi-Layer Perceptron (MLP) for projecting node embeddings to a new space.

    Parameters
    ----------
    n_in_feats : int
        Number of input features.
    n_out_feats : int
        Number of output features.
    """

    def __init__(self, n_in_feats, n_out_feats):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in_feats, n_out_feats)  # Define the first fully connected layer
        self.fc2 = nn.Linear(n_out_feats, n_out_feats)  # Define the second fully connected layer

    def forward(self, x):
        z = F.elu(self.fc1(x))  # Apply ELU activation after the first layer
        return self.fc2(z)  # Return the output of the second layer
