"""
Deep Graph Infomax(DGI)

References
----------
Papers: https://arxiv.org/abs/1809.10341
Author's code: https://github.com/PetarV-/DGI
DGL code: https://github.com/dmlc/dgl/tree/master/examples/pytorch/dgi
"""

import math

import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv


class DGI(nn.Module):
    r"""
    Deep Graph Infomax (DGI) model that maximizes mutual information between node embeddings and a graph summary.

    Parameters
    -----------
    n_in_feats: int
        Input feature size.
    n_hidden: int
        Hidden feature size.
    n_layers: int
        Number of the GNN encoder layers.
    p_drop: float
        Dropout rate for regularization.
    """

    def __init__(self, n_in_feats, n_hidden=512, n_layers=1, p_drop=0):
        super(DGI, self).__init__()
        self.encoder = GCNEncoder(n_in_feats, n_hidden, n_layers, p_drop)  # Initialize the encoder
        self.discriminator = Discriminator(n_hidden)  # Initialize the discriminator
        self.loss = nn.BCEWithLogitsLoss()  # Initialize the loss function for binary cross-entropy

    def forward(self, graph, feat):
        positive = self.encoder(graph, feat, corrupt=False)  # Encode positive samples
        negative = self.encoder(graph, feat, corrupt=True)  # Encode negative samples
        summary = torch.sigmoid(positive.mean(dim=0))  # Compute summary vector
        positive = self.discriminator(positive, summary)  # Discriminate positive samples
        negative = self.discriminator(negative, summary)  # Discriminate negative samples
        l1 = self.loss(positive, torch.ones_like(positive))  # Calculate loss for positive samples
        l2 = self.loss(negative, torch.zeros_like(negative))  # Calculate loss for negative samples
        return l1 + l2


class GCNEncoder(nn.Module):
    r"""
    A GCN-based encoder module which applies graph convolutions to input features.

    Parameters
    -----------
    n_in_feats: int
        Input feature size.
    n_hidden: int
        Hidden feature size.
    n_layers: int
        Number of the GNN encoder layers (>=2).
    p_drop: float
        Dropout rate for regularization.
    """

    def __init__(self, n_in_feats, n_hidden, n_layers, p_drop):
        super(GCNEncoder, self).__init__()
        # Input layer with PReLU activation function
        self.input_layer = GraphConv(n_in_feats, n_hidden, activation=nn.PReLU(n_hidden))
        # Hidden layers within a ModuleList
        self.hidden_layers = nn.ModuleList()
        assert n_layers >= 2
        for _ in range(n_layers - 2):
            # Hidden layers with PReLU activation function
            self.hidden_layers.append(GraphConv(n_hidden, n_hidden, activation=nn.PReLU(n_hidden)))
        self.output_layer = GraphConv(n_hidden, n_hidden)  # Output layer without activation function
        self.dropout = nn.Dropout(p=p_drop)  # Dropout module

    def forward(self, graph, feat, corrupt=False):
        if corrupt:
            perm = torch.randperm(graph.num_nodes())  # Corrupt features by random permutation
            feat = feat[perm]
        feat = self.input_layer(graph, feat)  # Apply input layer
        feat = self.dropout(feat)  # Apply dropout
        for hidden_layer in self.hidden_layers:
            feat = hidden_layer(graph, feat)  # Apply hidden layers
            feat = self.dropout(feat)  # Apply dropout
        feat = self.output_layer(graph, feat)  # Apply output layer
        return feat


class Discriminator(nn.Module):
    r"""
    A discriminator module for distinguishing between real and corrupted embeddings.

    Parameters
    -----------
    n_hidden: int
        Hidden dimension size used for bilinear transformation.
    """

    def __init__(self, n_hidden):
        super(Discriminator, self).__init__()
        # Weight parameters for bilinear transformation
        self.weight = nn.Parameter(torch.Tensor(n_hidden, n_hidden))
        self.uniform_weight()

    def uniform_weight(self):
        bound = 1.0 / math.sqrt(self.weight.size(0))  # Uniformly initialize weights within a certain bound
        self.weight.data.uniform_(-bound, bound)

    def forward(self, feat, summary):
        # Apply bilinear transformation to the feature and summary vector
        feat = torch.matmul(feat, torch.matmul(self.weight, summary))
        return feat
