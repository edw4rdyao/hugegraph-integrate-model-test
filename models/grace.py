"""
GRACE (Graph Contrastive Learning)

References
----------
Papers: https://arxiv.org/abs/2006.04131
Author's code: https://github.com/CRIPAC-DIG/GRACE
DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/grace
"""

import torch as th
import torch.nn as nn
import torch.nn.functional as F

from dgl.nn.pytorch import GraphConv


class GCN(nn.Module):
    r"""
    A Graph Convolutional Network (GCN) module for graph node feature transformation.

    Parameters
    -----------
    n_in_feats: int
        Number of input features per node.
    n_out_feats: int
        Number of output features per node.
    act_fn: nn.Module
        Activation function to use after each convolution.
    n_layers: int
        Number of layers in the GCN, at least 2.
    """

    def __init__(self, n_in_feats, n_out_feats, act_fn, n_layers=2):
        super(GCN, self).__init__()
        assert n_layers >= 2, "Number of layers should be at least 2."

        self.n_layers = n_layers
        self.n_hidden = n_out_feats * 2  # Set hidden dimension as twice the output dimension.
        self.input_layer = GraphConv(n_in_feats, self.n_hidden, activation=act_fn)
        self.hidden_layers = nn.ModuleList()
        for _ in range(n_layers - 2):
            self.hidden_layers.append(GraphConv(self.n_hidden, self.n_hidden, activation=act_fn))
        self.output_layer = GraphConv(self.n_hidden, n_out_feats, activation=act_fn)

    def forward(self, graph, feat):
        # Apply graph convolutions
        feat = self.input_layer(graph, feat)
        for hidden_layer in self.hidden_layers:
            feat = hidden_layer(graph, feat)
        feat = self.output_layer(graph, feat)
        return feat


class MLP(nn.Module):
    r"""
    A simple Multi-Layer Perceptron (MLP) for projecting node embeddings to a new space.

    Parameters
    -----------
    n_in_feats: int
        Number of input features.
    n_out_feats: int
        Number of output features.
    """

    def __init__(self, n_in_feats, n_out_feats):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(n_in_feats, n_out_feats)  # First fully connected layer.
        self.fc2 = nn.Linear(n_out_feats, n_in_feats)  # Second fully connected layer.

    def forward(self, x):
        z = F.elu(self.fc1(x))  # Apply ELU activation after the first layer.
        return self.fc2(z)  # Return the output of the second layer.


class Grace(nn.Module):
    r"""
    Implementation of the GRACE model for graph representation learning via contrastive learning.

    Parameters
    -----------
    n_in_feats: int
        Number of input features.
    n_hidden: int
        Hidden layer size.
    n_out_feats: int
        Output feature size.
    n_layers: int
        Number of GNN layers.
    act_fn: nn.Module
        Activation function used in the GCN.
    temp: float
        Temperature parameter used in contrastive loss.
    """

    def __init__(self, n_in_feats, n_hidden, n_out_feats, n_layers, act_fn, temp):
        super(Grace, self).__init__()
        self.encoder = GCN(n_in_feats, n_hidden, act_fn, n_layers)  # GCN encoder.
        self.proj = MLP(n_hidden, n_out_feats)  # MLP projector.
        self.temp = temp  # Temperature for scaling contrastive loss.

    @staticmethod
    def sim(z1, z2):
        z1 = F.normalize(z1)  # Normalize the features.
        z2 = F.normalize(z2)
        s = th.mm(z1, z2.t())  # Compute cosine similarity.
        return s

    def get_loss(self, z1, z2):
        f = lambda x: th.exp(x / self.temp)  # Exponential function scaled by temperature.

        refl_sim = f(self.sim(z1, z1))  # Similarity within the same view.
        between_sim = f(self.sim(z1, z2))  # Similarity between different views.

        x1 = refl_sim.sum(1) + between_sim.sum(1) - refl_sim.diag()  # Denominator of contrastive loss.
        loss = -th.log(between_sim.diag() / x1)  # Numerator of contrastive loss.

        return loss

    def get_embedding(self, graph, feat):
        h = self.encoder(graph, feat)  # Get embeddings from the GCN.
        return h.detach()  # Detach embeddings from the graph for evaluation.

    def forward(self, graph1, graph2, feat1, feat2):
        h1 = self.encoder(graph1, feat1)  # Encode first graph features.
        h2 = self.encoder(graph2, feat2)  # Encode second graph features.
        z1 = self.proj(h1)  # Project first graph embeddings.
        z2 = self.proj(h2)  # Project second graph embeddings.
        l1 = self.get_loss(z1, z2)  # Calculate loss for the first projection.
        l2 = self.get_loss(z2, z1)  # Calculate loss for the second projection.
        ret = (l1 + l2) * 0.5  # Average the losses for symmetry.
        return ret.mean()  # Return the mean of the loss.
